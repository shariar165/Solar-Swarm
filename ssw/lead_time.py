"""Weakness #1 — what the sub-L1 sentinel adds, measured on REAL storms.

The replay UI shows one number per made-up CME. Here every real CME-driven shock in
the event table gets its own number, using the SHOCK speed measured at L1. The shock
moves faster than the plasma behind it, so the post-shock plasma speed would overstate
the travel time (and the gain). The shock speed comes from mass-flux conservation
across the jump in OMNI density and speed (events.shock_mass_flux):

  V_sh = (n2 v2 - n1 v1) / (n2 - n1),  upstream [-30, -5] min, downstream [+5, +30] min

Events with a weak density jump (n2/n1 < 1.2) or V_sh outside 250-3000 km/s are dropped.


  extra warning (sub-L1 vs L1)  = travel time of the shock over the 750,000 km offset
  baseline warning (L1 only)    = travel time from L1 to Earth's bow shock

And what the in-situ contact does to the arrival forecast:
  before contact: the DBM test error (from validate_dbm, real)
  after contact : the remaining distance / measured speed, with an ASSUMED 3%
                  speed-measurement error and the DBM deceleration over the gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dbm
from .config import SUB_L1_OFFSET_KM

OFFSETS_KM = [250e3, 500e3, 750e3, 1.5e6, 3.0e6]
SPEED_ERR = 0.03       # ASSUMED in-situ speed measurement error (fractional)
MIN_N_RATIO = 1.2      # weakest density jump accepted as a shock
V_RANGE = (250.0, 3000.0)


def _travel_min(d_km, v, w, g):
    """Minutes for a front moving at v (decelerating toward w) to cover d_km."""
    t = dbm.arrival_time(v, w, g, r_target=d_km, r0=0.0)
    return np.nan if t is None else t / 60.0


def _summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p10=float(np.percentile(x, 10)),
                p90=float(np.percentile(x, 90)), min=float(x.min()), max=float(x.max()))


def run(ev: pd.DataFrame, gamma: float, dbm_test_mae_h: float | None) -> tuple[dict, pd.DataFrame]:
    cand = ev[ev.plausible].copy()
    if "v_shock_mf" not in cand:
        cand["v_shock_mf"], cand["n_ratio"] = np.nan, np.nan
    keep = cand.v_shock_mf.between(*V_RANGE) & (cand.n_ratio >= MIN_N_RATIO)
    dropped = dict(n_candidates=int(len(cand)), no_jump_found=int(cand.v_shock_mf.isna().sum()),
                   weak_density_jump=int((cand.n_ratio < MIN_N_RATIO).sum()),
                   speed_out_of_range=int((cand.n_ratio >= MIN_N_RATIO).sum() - keep.sum()))
    ev = cand[keep].copy()
    w = ev.w_obs.fillna(400.0)
    rows = []
    for (_, r), wi in zip(ev.iterrows(), w):
        v = float(r.v_shock_mf)
        gain = _travel_min(SUB_L1_OFFSET_KM, v, wi, gamma)
        base = _travel_min(dbm.L1_OFFSET_KM - dbm.BOW_SHOCK_KM, v, wi, gamma)
        # forecast error after S1 contact: remaining distance at a mis-measured speed
        rem = SUB_L1_OFFSET_KM + dbm.L1_OFFSET_KM - dbm.BOW_SHOCK_KM
        t_true = _travel_min(rem, v, wi, gamma)
        t_meas = _travel_min(rem, v * (1 + SPEED_ERR), wi, gamma)
        rows.append(dict(cme_id=r.cme_id, t_obs=r.t_obs, v_shock=v, v_plasma=r.v_shock, n_ratio=r.n_ratio, w_obs=wi,
                         gain_min=gain, l1_warning_min=base, total_min=gain + base,
                         eta_err_after_contact_min=abs(t_true - t_meas)))
    per = pd.DataFrame(rows)
    sens = {}
    for d in OFFSETS_KM:
        sens[f"{int(d/1e3)}k_km"] = _summ([_travel_min(d, v, wi, gamma) for v, wi in zip(ev.v_shock_mf, w)])
    res = dict(
        n_events=int(len(per)),
        shock_speed_kms=_summ(per.v_shock) if len(per) else None,
        post_shock_plasma_speed_kms=_summ(per.v_plasma) if len(per) else None,
        density_jump=_summ(per.n_ratio) if len(per) else None,
        selection=dropped,
        shock_speed_method="mass flux: (n2 v2 - n1 v1)/(n2 - n1), OMNI medians [-30,-5] / [+5,+30] min around the jump; n2/n1 >= 1.2, 250-3000 km/s",
        gain_min=_summ(per.gain_min) if len(per) else None,
        l1_only_warning_min=_summ(per.l1_warning_min) if len(per) else None,
        with_sentinel_warning_min=_summ(per.total_min) if len(per) else None,
        eta_error_before_contact_h=dbm_test_mae_h,
        eta_error_after_contact_min=_summ(per.eta_err_after_contact_min) if len(per) else None,
        offset_sensitivity_min=sens,
        assumption="in-situ speed error 3%; shock assumed to keep its L1-measured speed profile over the last 2.2 million km",
    )
    if len(per):
        fastest = per.sort_values("v_shock", ascending=False).head(5)
        res["fastest_events"] = fastest[["cme_id", "v_shock", "gain_min"]].round(1).to_dict("records")
    return res, per
