"""Event priors for the v11 replay: what a "typical Earth-impacting CME" looks like.

Every replay in ui/solar-swarm-v11.html draws its hidden truth from these
distributions, so they come from the real event table wherever possible:

  v0          log-normal fit to DONKI speeds at 21.5 R_sun (plausible pairs)
  half_angle  empirical quantiles of the DONKI half-angle
  lon         empirical quantiles of the DONKI longitude
  w           empirical quantiles of the real 24-h ambient wind before launch (w_obs)
  gamma       log-normal fit to the per-event gamma that reproduces each training arrival
  gamma_m1    the calibrated global gamma (M1)
  bz_sheath   empirical quantiles of the most southward Bz, shock to +6 h (OMNI)
  tau_c       log-normal fit to the sheath correlation time (lambda / v_shock, D = 30 min)
  historical  up to 40 real events that the UI can replay

Anything that has no data falls back to the ASSUMED values below, and is marked
source = "assumed" so the UI can label it.

Log-normal fits are robust (median and IQR), because a few DONKI link errors give
absurd per-event gammas that would otherwise dominate sigma.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import dbm, structure_scale

QS = np.linspace(0, 100, 21)          # quantile grid handed to the UI (inverse-CDF sampling)
N_HIST = 40

FALLBACK = dict(
    v0=dict(kind="lognormal", median=800.0, sigma_ln=0.40, clip=[450.0, 3000.0]),
    half_angle=dict(kind="uniform", lo=25.0, hi=60.0),
    lon=dict(kind="normal", mean=0.0, sd=15.0),
    w=dict(kind="normal", mean=430.0, sd=90.0, clip=[300.0, 750.0]),
    gamma=dict(kind="lognormal", median=2e-8, sigma_ln=0.7),
    gamma_m1=dict(kind="value", value=2e-8),
    bz_sheath=dict(kind="normal", mean=-6.0, sd=7.0),
    tau_c=dict(kind="lognormal", median=15.0, sigma_ln=0.5),
)


def _assumed(key):
    return dict(FALLBACK[key], source="assumed", n=0)


def lognormal_fit(x, min_n=10):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v) and v > 0], float)
    if x.size < min_n:
        return None
    ln = np.log(x)
    q25, q50, q75 = np.percentile(ln, [25, 50, 75])
    return dict(kind="lognormal", median=float(math.exp(q50)), sigma_ln=float((q75 - q25) / 1.349),
                clip=[float(x.min()), float(x.max())], n=int(x.size))


def quantile_fit(x, min_n=10):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if x.size < min_n:
        return None
    return dict(kind="quantiles", p=QS.tolist(), q=np.percentile(x, QS).round(4).tolist(),
                median=float(np.median(x)), mean=float(x.mean()), sd=float(x.std()), n=int(x.size))


def _pick(key, fit, label):
    return dict(fit, source=f"data: {label} (n={fit['n']})") if fit else _assumed(key)


def _iso(t):
    t = pd.to_datetime(t, utc=True, errors="coerce")
    return None if pd.isna(t) else t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x, nd=None):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd) if nd is not None else float(x)


def _dbm_gamma(r):
    """Gamma that makes the drag model reproduce the real L1 arrival with the DONKI speed and
    the measured wind — None if no gamma can (arrival slower than the wind allows, or faster
    than a drag-free CME). Those events cannot be replayed honestly and are skipped."""
    if not np.isfinite(r.get("r_target", np.nan)):
        return None
    g = dbm.gamma_for_arrival(r.v0, r.w_obs, r.transit_h * 3600, r.r_target)
    return g if g is not None and 1e-10 < g < 1e-6 else None


def historical(ev: pd.DataFrame, n=N_HIST) -> list[dict]:
    """Up to n real events with everything the replay needs: the 10 fastest, then the rest
    spread evenly over the period (so both solar cycles are represented). Only events whose
    arrival the drag model can reproduce (see _dbm_gamma) are eligible."""
    need = ["v0", "half_angle", "lon", "w_obs", "v_shock", "t_obs", "t21"]
    if ev.empty or any(c not in ev for c in need):
        return []
    ok = ev[ev.plausible].dropna(subset=need).sort_values("t21").reset_index(drop=True)
    ok["g_fit"] = [_dbm_gamma(r) for _, r in ok.iterrows()]
    ok = ok[ok.g_fit.notna()].reset_index(drop=True)
    if ok.empty:
        return []
    fast = list(ok.v0.nlargest(min(10, n)).index)
    rest = [i for i in ok.index if i not in fast]
    k = max(0, min(n - len(fast), len(rest)))
    pick = sorted(fast + [rest[int(j)] for j in np.linspace(0, len(rest) - 1, k).round()] if k else fast)
    out = []
    for i in dict.fromkeys(pick):
        r = ok.loc[i]
        out.append(dict(cme_id=str(r.cme_id), t21=_iso(r.t21), t_obs=_iso(r.t_obs),
                        transit_h=_num(r.transit_h, 3), v0=_num(r.v0, 1), half_angle=_num(r.half_angle, 1),
                        lon=_num(r.lon, 1), lat=_num(r.get("lat", np.nan), 1), w_obs=_num(r.w_obs, 1),
                        v_shock=_num(r.v_shock, 1), bz_min_sheath=_num(r.get("bz_min_sheath", np.nan), 2),
                        r_target_km=_num(r.get("r_target", np.nan), 0), enlil_arr=_iso(r.get("enlil_arr")),
                        g_fit_axis=_num(r.g_fit)))
    return out


def build(ev: pd.DataFrame, gstar=None, tau_rows=None, live=None, mode="real") -> dict:
    """ev = events.csv table; gstar = per-event training gammas; tau_rows = per-event
    sheath correlation times; live = dbm_live_params."""
    pl = ev[ev.plausible] if (not ev.empty and "plausible" in ev) else ev
    col = lambda c: pl[c].values if c in pl else []
    P = dict(meta=dict(data_mode=mode, generated_utc=pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
                       n_events=int(len(pl))))
    P["v0"] = _pick("v0", lognormal_fit(col("v0")), "DONKI speed at 21.5 R_sun")
    P["half_angle"] = _pick("half_angle", quantile_fit(col("half_angle")), "DONKI half-angle, deg")
    P["lon"] = _pick("lon", quantile_fit(col("lon")), "DONKI longitude, deg")
    P["w"] = _pick("w", quantile_fit(col("w_obs")), "L1 wind, 24-h median before launch")
    g = gstar["g_star"].values if gstar is not None and "g_star" in gstar else []
    P["gamma"] = _pick("gamma", lognormal_fit(g), "per-event gamma reproducing each training arrival")
    gm1 = (live or {}).get("gamma")
    P["gamma_m1"] = (dict(kind="value", value=float(gm1), source="data: M1 calibration", n=int(P["gamma"]["n"]))
                     if gm1 else _assumed("gamma_m1"))
    P["bz_sheath"] = _pick("bz_sheath", quantile_fit(col("bz_min_sheath")), "OMNI min Bz, shock to +6 h, nT")
    tau = []
    if tau_rows is not None and len(tau_rows):
        tr = tau_rows[tau_rows.detrend_min == structure_scale.DEFAULT_D]
        tau = tr.tau_s.values
    P["tau_c"] = _pick("tau_c", lognormal_fit(tau),
                       f"sheath correlation time, detrend {structure_scale.DEFAULT_D} min, s")
    P["historical"] = historical(ev)
    P["meta"]["historical_note"] = (f"chosen among {_n_fit(ev)} of {int(len(pl))} events whose real L1 arrival the "
                                    "drag model can reproduce with the DONKI speed and the measured wind")
    return P


def _n_fit(ev):
    if ev.empty or "w_obs" not in ev:
        return 0
    ok = ev[ev.plausible].dropna(subset=["v0", "w_obs", "transit_h"])
    return int(sum(_dbm_gamma(r) is not None for _, r in ok.iterrows()))


def from_results(out_dir: Path, mode="real") -> dict:
    """Rebuild the priors from the CSVs written by `run_all.py analyze`."""
    out_dir = Path(out_dir)
    f = out_dir / "events.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found — run `python run_all.py analyze` first")
    ev = pd.read_csv(f, parse_dates=["t21", "t_obs", "enlil_arr"])
    rd = lambda n: pd.read_csv(out_dir / n) if (out_dir / n).exists() else None
    live_f = out_dir / "dbm_live_params.json"
    live = json.loads(live_f.read_text()) if live_f.exists() else None
    return build(ev, rd("dbm_train_gstar.csv"), rd("structure_per_event.csv"), live, mode)


def write(P: dict, js_path: Path, json_path: Path | None = None):
    js_path = Path(js_path)
    js_path.parent.mkdir(parents=True, exist_ok=True)
    js_path.write_text("// generated by run_all.py priors — do not edit by hand\nwindow.SS_PRIORS = "
                       + json.dumps(P) + ";\n", encoding="utf-8")
    if json_path:
        Path(json_path).write_text(json.dumps(P, indent=1), encoding="utf-8")


def read_js(js_path: Path) -> dict:
    """Parse ui/event_priors.js back into a dict (used by ssw/montecarlo.py)."""
    s = Path(js_path).read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1: s.rstrip().rindex(";")])
