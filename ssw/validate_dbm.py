"""Weakness #4 — real-data validation of CME arrival forecasts.

Models (all causal: only information available at forecast time):
  M0   literature DBM: gamma = 2e-8 /km, w = 400 km/s
  M0w  same gamma, but w = real ambient wind measured at L1 in the 24 h before launch
  M1   gamma and w fitted on the training years
  M2   per-event gamma predicted by ridge regression from CME features (v0, width,
       direction, ambient wind) — the "ML-assisted" model
  ENLIL  NASA's operational WSA-Enlil prediction recorded in DONKI (reference)

Split: chronological (first 70% train, last 30% test). Uncertainty: ensemble sigma,
rescaled on the training set so that +/-1 sigma covers 68% of training errors.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import dbm
from .config import GAMMA0, W0, SPLIT_FRACTION

FEATS = ["log_v0", "half_angle", "abs_lon", "abs_lat", "w_used"]


def _pred(v0, w, g, r):
    t = dbm.arrival_time(v0, w, g, r)
    return np.nan if t is None else t / 3600.0


def _metrics(err_h, sig_h=None):
    e = np.asarray(err_h, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return None
    m = dict(n=int(e.size), mae_h=float(np.mean(np.abs(e))), rmse_h=float(np.sqrt(np.mean(e ** 2))),
             bias_h=float(np.mean(e)), median_abs_h=float(np.median(np.abs(e))))
    if sig_h is not None:
        s = np.asarray(sig_h, float)
        ok = np.isfinite(np.asarray(err_h, float)) & np.isfinite(s)
        if ok.any():
            m["coverage_1sigma"] = float(np.mean(np.abs(np.asarray(err_h)[ok]) <= s[ok]))
            m["mean_sigma_h"] = float(np.mean(s[ok]))
    return m


def _boot_diff(a, b, n=4000, seed=1):
    """Paired bootstrap 95% CI of MAE(a) - MAE(b) on events where both exist."""
    a, b = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 5:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (n, a.size))
    d = a[idx].mean(1) - b[idx].mean(1)
    return dict(n=int(a.size), mae_diff_h=float(a.mean() - b.mean()),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p_better=float(np.mean(d < 0)))


def _fit_global(tr, use_wobs):
    best = (1e9, None, None)
    for g in np.logspace(-9, -6.3, 28):
        for w in ([None] if use_wobs else np.arange(300, 601, 25)):
            p = [_pred(r.v0, (r.w_used if w is None else w), g, r.r_target) for r in tr.itertuples()]
            e = np.array(p) - tr.transit_h.values
            mae = np.nanmean(np.abs(e)) + 50 * np.mean(~np.isfinite(e))   # penalise non-arrivals
            if mae < best[0]:
                best = (mae, g, w)
    return best[1], best[2]


def _sigma_raw(r, w, g):
    _, s, _ = dbm.ensemble(r.v0, w, g, r.r_target, n=80, seed=int(r.v0))
    return np.nan if s is None else s / 3600.0


def run(ev: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    ev = ev[ev.plausible].copy().sort_values("t21").reset_index(drop=True)
    have_w = ev["w_obs"].notna().mean() > 0.7 if "w_obs" in ev else False
    ev["w_used"] = ev["w_obs"].fillna(W0) if "w_obs" in ev else W0
    ev["log_v0"] = np.log(ev.v0)
    ev["abs_lon"] = ev.lon.abs()
    ev["abs_lat"] = ev.lat.abs()
    for c in ["half_angle", "abs_lon", "abs_lat"]:
        ev[c] = ev[c].fillna(ev[c].median())
    n_tr = int(len(ev) * SPLIT_FRACTION)
    tr, te = ev.iloc[:n_tr].copy(), ev.iloc[n_tr:].copy()

    # --- M1: global calibration
    g1, w1 = _fit_global(tr, use_wobs=have_w)
    w1_txt = "real ambient (24 h before launch)" if w1 is None else f"{w1:.0f} km/s"

    # --- M2: per-event gamma -> ridge regression on log gamma
    tr["g_star"] = [dbm.gamma_for_arrival(r.v0, r.w_used, r.transit_h * 3600, r.r_target) for r in tr.itertuples()]
    fit = tr.dropna(subset=["g_star"])
    X = fit[FEATS].values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(fit)), (X - mu) / sd]
    y = np.log(fit.g_star.values)
    lam = 3.0
    beta = np.linalg.solve(Xs.T @ Xs + lam * np.diag([0] + [1] * len(FEATS)), Xs.T @ y)
    # residual spread of log-gamma becomes the ensemble width for M2
    sg2 = float(np.std(y - Xs @ beta))

    def g2(df):
        Z = np.c_[np.ones(len(df)), (df[FEATS].values - mu) / sd]
        return np.exp(np.clip(Z @ beta, math.log(1e-10), math.log(5e-6)))

    models = {
        "M0": lambda d: [(r.v0, W0, GAMMA0) for r in d.itertuples()],
        "M0w": lambda d: [(r.v0, r.w_used, GAMMA0) for r in d.itertuples()],
        "M1": lambda d: [(r.v0, (r.w_used if w1 is None else w1), g1) for r in d.itertuples()],
        "M2": lambda d: [(r.v0, r.w_used, g) for r, g in zip(d.itertuples(), g2(d))],
    }

    out, per_event, params = {}, te[["cme_id", "t21", "t_obs", "transit_h", "v0"]].copy(), {}
    for name, f in models.items():
        p_tr = np.array([_pred(v, w, g, r.r_target) for (v, w, g), r in zip(f(tr), tr.itertuples())])
        p_te = np.array([_pred(v, w, g, r.r_target) for (v, w, g), r in zip(f(te), te.itertuples())])
        # calibrate sigma: k such that 68% of |train error| <= k * raw sigma
        s_tr = np.array([_sigma_raw(r, w, g) for (v, w, g), r in zip(f(tr), tr.itertuples())])
        s_te = np.array([_sigma_raw(r, w, g) for (v, w, g), r in zip(f(te), te.itertuples())])
        ratio = np.abs(p_tr - tr.transit_h.values) / s_tr
        k = float(np.nanpercentile(ratio, 68)) if np.isfinite(ratio).sum() > 5 else 1.0
        err = p_te - te.transit_h.values
        out[name] = dict(test=_metrics(err, k * s_te), train=_metrics(p_tr - tr.transit_h.values),
                         sigma_scale=k, no_arrival_test=int(np.sum(~np.isfinite(p_te))))
        per_event[f"pred_{name}_h"] = p_te
        per_event[f"sig_{name}_h"] = k * s_te
        params[name] = k

    # --- Enlil reference (only events where DONKI has a causal Enlil run)
    en_err = ((te.enlil_arr - te.t_obs).dt.total_seconds() / 3600).values
    per_event["err_ENLIL_h"] = en_err
    out["ENLIL"] = dict(test=_metrics(en_err))

    best = min(["M0w", "M1", "M2"], key=lambda m: (out[m]["test"] or {"mae_h": 1e9})["mae_h"])
    errs = {m: per_event[f"pred_{m}_h"].values - te.transit_h.values for m in models}
    comparisons = {
        f"{best}_vs_M0": _boot_diff(errs[best], errs["M0"]),
        f"{best}_vs_ENLIL": _boot_diff(errs[best], en_err),
    }
    res = dict(
        n_events_total=int(len(ev)), n_train=int(len(tr)), n_test=int(len(te)),
        train_period=[str(tr.t21.min().date()), str(tr.t21.max().date())],
        test_period=[str(te.t21.min().date()), str(te.t21.max().date())] if len(te) else None,
        ambient_wind_source="real L1 (24 h median before launch)" if have_w else "fixed 400 km/s",
        m1=dict(gamma_per_km=g1, w=w1_txt),
        m2=dict(features=FEATS, ridge_lambda=lam, n_fit=int(len(fit)), log_gamma_resid_sd=sg2),
        models=out, best_model=best, comparisons=comparisons,
    )
    live = dict(gamma=float(g1), w_rule=("ambient" if w1 is None else float(w1)),
                sigma_scale=out["M1"]["sigma_scale"], test_mae_h=out["M1"]["test"]["mae_h"] if out["M1"]["test"] else None)
    return res, per_event, live


def save(res, per_event, live, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dbm_validation.json").write_text(json.dumps(res, indent=2, default=str))
    (out_dir / "dbm_live_params.json").write_text(json.dumps(live, indent=2))
    per_event.to_csv(out_dir / "dbm_test_predictions.csv", index=False)
