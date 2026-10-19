"""Synthetic data in the SAME file formats as the real DONKI / CDAWeb downloads.

Purpose: prove the pipeline runs end to end. Numbers produced from this are NOT
scientific results — they only show that each step computes what it should
(e.g. calibration recovers the gamma that was planted).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ssw import dbm  # noqa: E402

START, END = "2019-01-01", "2023-12-31"
TRUE_GAMMA_MEDIAN = 6e-8          # planted; calibration should land near this
TRUE_LAMBDA_KM = 60_000.0         # planted sheath correlation length


def _iso(t):
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%MZ")


def build(raw: Path, seed: int = 7, n_cme: int = 140):
    rng = np.random.default_rng(seed)
    raw = Path(raw)
    if (raw / "MOCK_DONE").exists():
        return
    t0, t1 = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC")
    span = (t1 - t0).total_seconds()

    # ---- ambient wind: slow 27-day-ish structure
    minutes = pd.date_range(t0, t1 + pd.Timedelta(days=1), freq="1min")
    tm = np.arange(minutes.size)
    w_amb = 420 + 70 * np.sin(2 * np.pi * tm / (27 * 1440)) + 40 * np.sin(2 * np.pi * tm / (9.1 * 1440) + 1)

    cmes, ips, arrivals = {}, {}, []
    times = np.sort(t0.value + rng.uniform(0.02, 0.97, n_cme) * span * 1e9)
    for k, tv in enumerate(times):
        t21 = pd.Timestamp(int(tv), tz="UTC").floor("min")
        v0 = float(np.clip(np.exp(rng.normal(np.log(750), 0.35)), 350, 2800))
        ha = float(np.clip(rng.normal(35, 12), 10, 80))
        lon, lat = float(rng.normal(0, 18)), float(rng.normal(0, 9))
        i_w = min(int((t21 - t0).total_seconds() // 60), tm.size - 1)
        w = float(w_amb[i_w])
        g = TRUE_GAMMA_MEDIAN * np.exp(0.45 * rng.standard_normal() + 0.01 * (ha - 35))
        r_t = dbm.l1_km(t21 + pd.Timedelta(days=2))
        tt = dbm.arrival_time(v0, w, g, r_t)
        if tt is None:
            continue
        t_arr = t21 + pd.Timedelta(seconds=tt + rng.normal(0, 2.5 * 3600))
        if t_arr > t1:
            continue
        v_arr = float(dbm.v_of_t(tt, v0, w, g))
        cid = f"{t21:%Y-%m-%dT%H:%M}:00-CME-001"
        iid = f"{t_arr:%Y-%m-%dT%H:%M}:00-IPS-001"
        en_done = t21 + pd.Timedelta(hours=rng.uniform(4, 12))
        en_arr = t_arr + pd.Timedelta(hours=rng.normal(1.5, 9))
        start = t21 - pd.Timedelta(minutes=90)
        cmes[cid] = dict(activityID=cid, startTime=_iso(start), linkedEvents=[{"activityID": iid}],
                         cmeAnalyses=[dict(time21_5=_iso(t21), speed=round(v0), halfAngle=round(ha),
                                           latitude=round(lat), longitude=round(lon), isMostAccurate=True,
                                           enlilList=[dict(modelCompletionTime=_iso(en_done),
                                                           estimatedShockArrivalTime=_iso(en_arr),
                                                           impactList=[{"location": "Earth", "arrivalTime": _iso(en_arr)}])])])
        ips[iid] = dict(activityID=iid, location="Earth", eventTime=_iso(t_arr), linkedEvents=[{"activityID": cid}])
        arrivals.append((t_arr, v_arr))
    # a few uncatalogued (real-but-unlabelled) jumps and some non-Earth IPS noise
    extra = [pd.Timestamp(int(t0.value + rng.uniform(0, 1) * span * 1e9), tz="UTC") for _ in range(25)]

    for kind, recs in [("CME", cmes), ("IPS", ips)]:
        d = raw / "donki" / kind
        d.mkdir(parents=True, exist_ok=True)
        by_month = {}
        for r in recs.values():
            key = (r.get("startTime") or r.get("eventTime"))[:7]
            by_month.setdefault(key, []).append(r)
        for m in pd.period_range(START, END, freq="M"):
            (d / f"{m}.json").write_text(json.dumps(by_month.get(str(m), [])))

    # ---- L1 1-min (OMNI-like, shifted ~40 min to the bow shock)
    V = w_amb + rng.normal(0, 6, tm.size)
    N = 6.0 * (420 / w_amb) ** 2 * np.exp(rng.normal(0, 0.15, tm.size))
    B = 5.5 + rng.normal(0, 0.5, tm.size)
    BZ = rng.normal(0, 2.5, tm.size)
    for t_arr, v_arr in arrivals + [(t, 0) for t in extra]:
        i = int((t_arr - t0).total_seconds() // 60) + 40
        if i >= tm.size - 1:
            continue
        j = np.arange(i, min(i + 18 * 60, tm.size))
        decay = np.exp(-(j - i) / 600.0)
        vj = (v_arr if v_arr else V[i] + rng.uniform(60, 120))
        V[j] = V[j] + max(vj - V[i], 40) * decay
        N[j] *= 1 + 1.6 * decay
        B[j] *= 1 + 1.4 * decay
        BZ[j] += rng.normal(-4, 4) * decay
    gaps = rng.uniform(0, 1, tm.size) < 0.01
    V[gaps] = 99999.9
    l1 = pd.DataFrame({"flow_speed": V, "proton_density": N, "BZ_GSM": BZ, "F": B}, index=minutes)
    l1.index.name = "time"
    l1.loc[l1.flow_speed > 9e4, "flow_speed"] = np.nan
    out = raw / "hapi" / "OMNI_HRO_1MIN"
    out.mkdir(parents=True, exist_ok=True)
    for m, g in l1.groupby(l1.index.to_period("M")):
        g.to_csv(out / f"{m}.csv.gz", compression="gzip")

    # ---- high-res magnetometer windows (3-s, OU process with planted correlation length)
    wdir = raw / "hapi" / "DSCOVR_H0_MAG" / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    dt = 3.0
    for (iid, rec), (t_arr, v_arr) in zip(ips.items(), arrivals):
        lam = TRUE_LAMBDA_KM * np.exp(0.4 * rng.standard_normal())
        tau = lam / max(v_arr, 300)
        n = int(3.4 * 3600 / dt)
        a = np.exp(-dt / tau)
        x = np.zeros((n, 3))
        for k in range(1, n):
            x[k] = a * x[k - 1] + np.sqrt(1 - a * a) * rng.normal(0, 3, 3)
        idx = pd.date_range(t_arr - pd.Timedelta(minutes=10), periods=n, freq=f"{int(dt)}s")
        df = pd.DataFrame(x + [2, -3, 1], index=idx, columns=["B1GSE_0", "B1GSE_1", "B1GSE_2"])
        df.index.name = "time"
        df.to_csv(wdir / f"{iid.replace(':', '')}.csv.gz", compression="gzip")
    (raw / "MOCK_DONE").write_text("synthetic")
    print(f"mock: {len(cmes)} CMEs, {len(arrivals)} arrivals, {len(extra)} unlabelled jumps")
