"""Build the CME -> Earth-arrival event table from DONKI records.

Observed arrival = a DONKI interplanetary shock (IPS) at location "Earth" that DONKI
links to the CME. Enlil baseline = the latest WSA-Enlil prediction completed BEFORE
the observed arrival (so the comparison is causal and fair).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dbm


def _t(x):
    return pd.to_datetime(x, utc=True, errors="coerce") if x else pd.NaT


def _ids(rec):
    return {e.get("activityID") for e in (rec.get("linkedEvents") or []) if e.get("activityID")}


def _best_analysis(cme):
    an = [a for a in (cme.get("cmeAnalyses") or []) if a.get("speed") and a.get("time21_5")]
    if not an:
        return None
    best = [a for a in an if a.get("isMostAccurate")]
    return (best or an)[-1]


def _enlil_before(cme, t_obs):
    best = None
    for a in cme.get("cmeAnalyses") or []:
        for en in a.get("enlilList") or []:
            done = _t(en.get("modelCompletionTime"))
            arr = None
            for imp in en.get("impactList") or []:
                if "earth" in str(imp.get("location", "")).lower() and imp.get("arrivalTime"):
                    arr = _t(imp["arrivalTime"])
            if arr is None or pd.isna(arr):
                arr = _t(en.get("estimatedShockArrivalTime"))
            if pd.isna(done) or pd.isna(arr) or done >= t_obs:
                continue
            if best is None or done > best[0]:
                best = (done, arr)
    return best


def _angle(x, lim):
    """DONKI writes 999 for an unknown direction: treat anything outside +-lim as missing."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if abs(v) <= lim else np.nan


def build_events(cmes: list[dict], ips: list[dict]) -> pd.DataFrame:
    cme_by_id = {c["activityID"]: c for c in cmes if c.get("activityID")}
    earth_ips = [i for i in ips if str(i.get("location", "")).strip().lower() == "earth"]
    # IPS -> linked CMEs, using links from both sides
    links = {i["activityID"]: {x for x in _ids(i) if "-CME-" in x} for i in earth_ips}
    for cid, c in cme_by_id.items():
        for x in _ids(c):
            if x in links:
                links[x].add(cid)

    rows = []
    for i in earth_ips:
        t_obs = _t(i.get("eventTime"))
        cands = [cme_by_id[c] for c in links.get(i["activityID"], ()) if c in cme_by_id]
        cands = [(c, _best_analysis(c)) for c in cands]
        cands = [(c, a) for c, a in cands if a is not None and _t(a["time21_5"]) < t_obs]
        if not cands or pd.isna(t_obs):
            continue
        c, a = max(cands, key=lambda ca: float(ca[1]["speed"]))   # fastest CME drives the shock
        t21 = _t(a["time21_5"])
        en = _enlil_before(c, t_obs)
        rows.append(dict(
            cme_id=c["activityID"], ips_id=i["activityID"], n_linked=len(cands),
            t21=t21, t_obs=t_obs, transit_h=(t_obs - t21).total_seconds() / 3600,
            v0=float(a["speed"]), half_angle=float(a.get("halfAngle") or np.nan),
            lon=_angle(a.get("longitude"), 180), lat=_angle(a.get("latitude"), 90),
            r_target=dbm.l1_km(t_obs),
            enlil_done=en[0] if en else pd.NaT, enlil_arr=en[1] if en else pd.NaT,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("t21").drop_duplicates("ips_id").reset_index(drop=True)
    # physically implausible pairings (DONKI link errors): transit outside 0.8-6 days
    df["plausible"] = df.transit_h.between(19, 144)
    return df


def add_insitu(df: pd.DataFrame, l1: pd.DataFrame) -> pd.DataFrame:
    """Attach real L1 measurements: ambient wind before launch (causal), the
    post-shock speed after arrival, and the most southward sheath Bz."""
    if l1.empty or df.empty:
        df["w_obs"] = np.nan
        df["v_shock"] = np.nan
        df["bz_min_sheath"] = np.nan
        return df
    V = l1["V"].dropna()
    BZ = l1["BZ"].dropna() if "BZ" in l1 else pd.Series(dtype=float)
    w_obs, v_sh, v_pre, bz_min = [], [], [], []
    for _, r in df.iterrows():
        pre = V[r.t21 - pd.Timedelta(hours=24): r.t21]                 # known at forecast time
        w_obs.append(pre.median() if len(pre) > 60 else np.nan)
        # OMNI is shifted to the bow shock (~30-60 min after L1), so look 0-3 h after
        post = V[r.t_obs + pd.Timedelta(minutes=5): r.t_obs + pd.Timedelta(hours=3)]
        before = V[r.t_obs - pd.Timedelta(hours=3): r.t_obs - pd.Timedelta(minutes=5)]
        v_sh.append(post.quantile(0.75) if len(post) > 30 else np.nan)
        v_pre.append(before.median() if len(before) > 30 else np.nan)
        # sheath Bz: shock to +6 h (same 5-min guard as v_shock against the OMNI shift)
        sh = BZ[r.t_obs + pd.Timedelta(minutes=5): r.t_obs + pd.Timedelta(hours=6)]
        bz_min.append(sh.min() if len(sh) > 60 else np.nan)
    df["w_obs"] = w_obs
    df["v_shock"] = v_sh
    df["v_pre"] = v_pre
    df["bz_min_sheath"] = bz_min
    return df
