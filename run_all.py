"""SOLAR SWARM — one command for the whole real-data pipeline.

  python run_all.py probe                      # check dataset/parameter names on CDAWeb
  python run_all.py fetch --start 2012-01-01 --end 2026-09-30
  python run_all.py analyze                    # all analyses -> results/, ui/results.js, ui/event_priors.js
  python run_all.py priors [--mock]            # only rebuild ui/event_priors.js from results/*.csv
  python run_all.py montecarlo [--n 1000]      # v11 replay statistics -> results/montecarlo.json
         [--home 3000] [--budget 20] [--out montecarlo_L3000.json]   # S5 home fixed (default: from the priors), Δv budget
  python run_all.py all  --start ... --end ... # fetch + analyze
  python run_all.py mock                       # pipeline test on synthetic data (NOT results)
  python run_all.py offline                    # only the steps that need no data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ssw import config, fetch, events, validate_dbm, lead_time, shock_detector, structure_scale, planner, reliability, priors, montecarlo

ROOT = config.ROOT


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ fetch
def cmd_probe(_):
    for ds, params in [(config.L1_DATASET, list(config.L1_PARAMS.values()))] + [(d, [p]) for d, p in config.HIRES_MAG]:
        try:
            info = fetch.hapi_info(ds)
            have = {p["name"] for p in info["parameters"]}
            miss = [p for p in params if p not in have]
            print(f"{ds}: {'OK' if not miss else 'MISSING ' + str(miss)}  range {info.get('startDate')} .. {info.get('stopDate')}")
            if miss:
                print("   available:", sorted(have))
        except Exception as e:
            print(f"{ds}: FAILED {e}")


def cmd_fetch(a):
    raw = ROOT / "data" / "raw"
    log("DONKI CME + IPS (monthly files)")
    for kind in ["CME", "IPS"]:
        fetch.fetch_donki(kind, a.start, a.end, raw, source=a.source, key=a.key)
    log(f"L1 1-min solar wind: {config.L1_DATASET}")
    fetch.fetch_hapi_months(config.L1_DATASET, list(config.L1_PARAMS.values()), a.start, a.end, raw)
    log("High-res magnetometer windows behind each shock")
    _fetch_hires(raw)
    log("fetch done")


def _event_table(raw: Path, l1: pd.DataFrame) -> pd.DataFrame:
    ev = events.build_events(fetch.load_donki(raw, "CME"), fetch.load_donki(raw, "IPS"))
    if ev.empty:
        raise SystemExit("No CME->Earth shock pairs found. Did the DONKI fetch work?")
    return events.add_insitu(ev, l1)


def _load_l1(raw: Path) -> pd.DataFrame:
    df = fetch.load_hapi_months(raw, config.L1_DATASET)
    if df.empty:
        return df
    return df.rename(columns={v: k for k, v in config.L1_PARAMS.items()})


def _fetch_hires(raw: Path):
    ev = events.build_events(fetch.load_donki(raw, "CME"), fetch.load_donki(raw, "IPS"))
    ok, done = 0, set()
    for ds, p in config.HIRES_MAG:
        try:
            info = fetch.hapi_info(ds)
            start = pd.Timestamp(info["startDate"])
            stop = pd.Timestamp(info["stopDate"])
        except Exception as e:
            log(f"  {ds}: info failed ({e})")
            continue
        for r in ev.itertuples():
            if r.ips_id in done or not (start <= r.t_obs <= stop):
                continue
            tag = r.ips_id.replace(":", "")
            try:
                fetch.fetch_hapi_window(ds, [p], r.t_obs - pd.Timedelta(minutes=10),
                                        r.t_obs + pd.Timedelta(hours=3, minutes=10), raw, tag, info)
                ok += 1
                done.add(r.ips_id)
            except Exception as e:
                log(f"  {ds} {tag}: {e}")
    log(f"  high-res windows available: {ok}")


# ------------------------------------------------------------------ analyze
def analyze(raw: Path, out: Path, ui_js: Path | None, mode: str, priors_js: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    R = dict(meta=dict(generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), data_mode=mode))

    log("reliability (no data needed)")
    R["reliability"] = reliability.run(config.SAT)

    l1 = _load_l1(raw)
    log(f"L1 rows: {len(l1):,}")
    ev = _event_table(raw, l1)
    ev.to_csv(out / "events.csv", index=False)
    log(f"events: {len(ev)} CME->Earth pairs ({int(ev.plausible.sum())} plausible)")
    R["events"] = dict(n_pairs=int(len(ev)), n_plausible=int(ev.plausible.sum()),
                       n_multi_cme=int((ev.n_linked > 1).sum()),
                       period=[str(ev.t21.min().date()), str(ev.t21.max().date())],
                       with_enlil=int(ev.enlil_arr.notna().sum()))

    log("DBM validation (M0 / M0w / M1 / M2 / Enlil)")
    res, per, live, gstar = validate_dbm.run(ev)
    validate_dbm.save(res, per, live, out, gstar)
    R["dbm"], R["dbm_live"] = res, live

    log("lead time on real shocks")
    R["lead_time"], lt = lead_time.run(ev, live["gamma"], live["test_mae_h"])
    lt.to_csv(out / "lead_time_per_event.csv", index=False)

    log("structure size from high-res sheath data")
    lengths, tau_rows = structure_per_event(raw, ev)
    tau_rows.to_csv(out / "structure_per_event.csv", index=False)
    R["structure_scale"] = structure_scale.summarize(lengths)
    s = R["structure_scale"]["by_detrend_min"].get(str(structure_scale.DEFAULT_D))

    log("scout optimizer (weight sweep)")
    if s:
        R["planner"] = planner.sweep(config.SCOUT, s["median_km"], s["sigma_ln"], "real sheath data")
    else:
        R["planner"] = planner.sweep(config.SCOUT, config.SCOUT["lam_prior_km"], config.SCOUT["lam_sigma_ln"], "ASSUMED (no data)")

    log("shock detector (ML vs rule)")
    if len(l1) > 1440 * 365:
        shocks = pd.to_datetime([i["eventTime"] for i in fetch.load_donki(raw, "IPS")
                                 if str(i.get("location", "")).strip().lower() == "earth"], utc=True)
        shocks = shocks[(shocks >= l1.index.min()) & (shocks <= l1.index.max())]
        try:
            R["shock_detector"], _ = shock_detector.run(l1, pd.Series(shocks), config.SHOCK_TRAIN_END)
        except RuntimeError as e:
            R["shock_detector"] = dict(error=str(e))
    else:
        R["shock_detector"] = dict(error="need at least a year of L1 data")

    (out / "summary.json").write_text(json.dumps(R, indent=2, default=_json))
    if priors_js:
        P = priors.from_results(out, mode)
        priors.write(P, priors_js, out / "priors.json")
        priors.write_triple_table(pd.read_csv(out / "events.csv", parse_dates=["t21", "t_obs"]), out)
        log(f"wrote {priors_js.relative_to(ROOT)} ({len(P['historical'])} historical events)")
    if ui_js:
        ui_js.parent.mkdir(parents=True, exist_ok=True)
        ui_js.write_text("// generated by run_all.py — do not edit by hand\nwindow.SS_RESULTS = "
                         + json.dumps(R, default=_json) + ";\n", encoding="utf-8")
        log(f"wrote {ui_js.relative_to(ROOT)}")
    log("analyze done")
    return R


def structure_per_event(raw: Path, ev: pd.DataFrame):
    """Sheath correlation time and length for every event with a high-res window."""
    lengths = {d: [] for d in structure_scale.DETREND_MIN}
    rows, used = [], set()
    for ds, p in config.HIRES_MAG:
        wdir = raw / "hapi" / ds / "windows"
        for r in ev[ev.plausible & ev.v_shock.notna()].itertuples():
            f = wdir / f"{r.ips_id.replace(':', '')}.csv.gz"
            if r.ips_id in used or not f.exists():
                continue
            used.add(r.ips_id)
            b = fetch.load_csv(f)[r.t_obs + pd.Timedelta(minutes=5): r.t_obs + pd.Timedelta(hours=3)]
            for d in lengths:
                tau = structure_scale.corr_time_s(b, d)
                lam = None if tau is None else float(r.v_shock * tau)
                lengths[d].append(lam)
                rows.append(dict(ips_id=r.ips_id, dataset=ds, v_shock=r.v_shock, detrend_min=d, tau_s=tau, lambda_km=lam))
    cols = ["ips_id", "dataset", "v_shock", "detrend_min", "tau_s", "lambda_km"]
    return lengths, pd.DataFrame(rows, columns=cols)


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def cmd_analyze(_):
    analyze(ROOT / "data" / "raw", ROOT / "results", ROOT / "ui" / "results.js", "real", ROOT / "ui" / "event_priors.js")


def cmd_all(a):
    cmd_fetch(a)
    cmd_analyze(a)


def cmd_offline(_):
    """Parts that need no downloaded data: reliability + planner on the assumed prior."""
    R = dict(meta=dict(generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       data_mode="partial — data steps not run yet (run: python run_all.py all)"),
             reliability=reliability.run(config.SAT),
             planner=planner.sweep(config.SCOUT, config.SCOUT["lam_prior_km"], config.SCOUT["lam_sigma_ln"], "ASSUMED (no data)"))
    js = ROOT / "ui" / "results.js"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text("// generated by run_all.py — do not edit by hand\nwindow.SS_RESULTS = " + json.dumps(R, default=_json) + ";\n", encoding="utf-8")
    log(f"wrote {js.relative_to(ROOT)}")


def cmd_mock(_):
    sys.path.insert(0, str(ROOT / "tests"))
    import mock_data
    raw = ROOT / "data_mock" / "raw"
    mock_data.build(raw)
    analyze(raw, ROOT / "results_mock", ROOT / "ui" / "results_mock.js", "MOCK — synthetic test data, not results",
            ROOT / "ui" / "event_priors_mock.js")


def cmd_priors(a):
    out, js = ((ROOT / "results_mock", ROOT / "ui" / "event_priors_mock.js") if a.mock
               else (ROOT / "results", ROOT / "ui" / "event_priors.js"))
    P = priors.from_results(out, "MOCK — synthetic test data, not results" if a.mock else "real")
    priors.write(P, js, out / "priors.json")
    priors.write_triple_table(pd.read_csv(out / "events.csv", parse_dates=["t21", "t_obs"]), out)
    log(f"wrote {js.relative_to(ROOT)} ({len(P['historical'])} historical events)")
    for k in ["v0", "half_angle", "lon", "w", "gamma", "gamma_m1", "bz_sheath", "tau_c"]:
        v = P[k]
        print(f"  {k:10s} {v['kind']:9s} median {v.get('median', v.get('value', '—'))!s:>22}  "
              f"{('sigma_ln %.2f' % v['sigma_ln']) if 'sigma_ln' in v else '':14s} {v['source']}")


def cmd_montecarlo(a):
    js = ROOT / "ui" / ("event_priors_mock.js" if a.mock else "event_priors.js")
    out_f = ROOT / ("results_mock" if a.mock else "results") / (a.out or "montecarlo.json")
    if a.home is not None:
        montecarlo.A["homeL"] = a.home
    if a.budget is not None:
        montecarlo.A["budget"] = a.budget
    log(f"Monte Carlo: {a.n} events, priors {js.relative_to(ROOT)}, S5 home {montecarlo.A['homeL']}, budget {montecarlo.A['budget']} m/s")
    out, _ = montecarlo.run(a.n, priors_path=js)
    montecarlo.save(out, out_f)
    g, e, s5 = out["gain_min"], out["eta_err_h"], out["s5"]
    log(f"wrote {out_f.relative_to(ROOT)} in {out['meta']['runtime_s']} s")
    print(f"  gain {g['median']:.1f} min (10-90 %: {g['p10']:.1f}-{g['p90']:.1f})")
    print(f"  ETA error median: T+0.3 h {e['T0']['median']:.2f} h, T+12 h {e['T12']['median']:.2f} h, "
          f"T+24 h {e['T24']['median']:.2f} h, after S1 {out['eta_err_after_S1_min']['median']:.1f} min")
    print(f"  2-sigma coverage {out['coverage_2sigma']}")
    print(f"  S5 home {s5['home_km']:.0f} km: moved in {s5['moved_pct']:.1f} % (in {s5['inward']}, out {s5['outward']}), "
          f"mean dv {s5['mean_dv_ms']:.1f} m/s (max {s5['max_dv_ms']:.1f}), final separations {s5['final_separation_km']}")
    print(f"  U before contact median {s5['U_before_contact']['median']:.3f}, below {montecarlo.A['uThr']}: {s5['U_below_thr_pct']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["probe", "fetch", "analyze", "all", "mock", "offline", "priors", "montecarlo"]:
        p = sub.add_parser(name)
        if name in ("priors", "montecarlo"):
            p.add_argument("--mock", action="store_true", help="use the mock priors / results_mock/ instead")
        if name == "montecarlo":
            p.add_argument("--n", type=int, default=1000)
            p.add_argument("--home", type=float, default=None, help="fixed S5 home separation, km (default: optimum from the priors)")
            p.add_argument("--budget", type=float, default=None, help="S5 event-time dv budget, m/s (default 40)")
            p.add_argument("--out", default=None, help="output file name in results/ (default montecarlo.json)")
        if name in ("fetch", "all"):
            p.add_argument("--start", default="2012-01-01")
            p.add_argument("--end", default=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
            p.add_argument("--source", choices=["ccmc", "nasa"], default="ccmc",
                           help="ccmc = kauai.ccmc.gsfc.nasa.gov (no key); nasa = api.nasa.gov (needs --key)")
            p.add_argument("--key", default="DEMO_KEY")
    a = ap.parse_args()
    {"probe": cmd_probe, "fetch": cmd_fetch, "analyze": cmd_analyze, "all": cmd_all, "mock": cmd_mock, "offline": cmd_offline,
     "priors": cmd_priors, "montecarlo": cmd_montecarlo}[a.cmd](a)


if __name__ == "__main__":
    main()
