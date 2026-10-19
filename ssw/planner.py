"""Weakness #2 (second half) — the scout optimizer, now fed by data and stress-tested.

Same model as the UI: expected gradient error U(L) of the S2-S5 pair, averaged over
the structure-size prior lambda; candidate separations scored J = U + w * dv.
New: (1) lambda prior comes from structure_scale (real sheath data) when available;
(2) the dv weight w is swept, so the decision is shown to be robust (or not).
"""
from __future__ import annotations

import math

import numpy as np

Z = np.linspace(-2.5, 2.5, 41)
GW = np.exp(-Z ** 2 / 2)
GW /= GW.sum()


def eps(L, lam, noise):
    x = 2 * math.pi * L / lam
    return min(1.5, math.hypot(math.sqrt(2) * noise / x, x * x / 6))


def expected_error(L, lam_med, sig_ln, noise):
    return float(sum(w * eps(L, lam_med * math.exp(sig_ln * z), noise) for z, w in zip(Z, GW)))


def plan(cfg: dict, lam_med: float, sig_ln: float, weight: float) -> dict:
    win = cfg["window_h"] * 3600
    u0 = expected_error(cfg["L0"], lam_med, sig_ln, cfg["noise_ratio"])
    rows = []
    for L in cfg["candidates"]:
        dv = 4 * abs(cfg["L0"] - L) * 1e3 / win            # bang-bang transfer, accelerate + brake
        u = expected_error(L, lam_med, sig_ln, cfg["noise_ratio"])
        rows.append(dict(L_km=L, U=u, dv_ms=dv, J=u + weight * dv, feasible=dv <= cfg["budget_ms"]))
    if u0 <= cfg["u_threshold"]:
        pick = dict(L_km=cfg["L0"], U=u0, dv_ms=0.0, action="hold (U within limit)")
    else:
        best = min((r for r in rows if r["feasible"]), key=lambda r: r["J"])
        pick = dict(L_km=best["L_km"], U=best["U"], dv_ms=best["dv_ms"],
                    action="hold (no better feasible option)" if best["L_km"] == cfg["L0"] else "trim")
    return dict(U0=u0, weight=weight, candidates=rows, decision=pick)


def sweep(cfg: dict, lam_med: float, sig_ln: float, source: str) -> dict:
    runs = [plan(cfg, lam_med, sig_ln, w) for w in cfg["weights"]]
    chosen = sorted({r["decision"]["L_km"] for r in runs})
    # unconstrained optimum, to show what the budget costs
    grid = np.geomspace(200, 2e6, 400)
    errs = [expected_error(L, lam_med, sig_ln, cfg["noise_ratio"]) for L in grid]
    i = int(np.argmin(errs))
    return dict(
        lambda_source=source, lambda_median_km=lam_med, lambda_sigma_ln=sig_ln,
        U0=runs[0]["U0"], threshold=cfg["u_threshold"],
        decisions_by_weight=[dict(weight=r["weight"], **r["decision"]) for r in runs],
        robust=len(chosen) == 1, separations_chosen_km=chosen,
        unconstrained_best_L_km=float(grid[i]), unconstrained_best_U=float(errs[i]),
        reachable_with_budget=bool(abs(grid[i] - cfg["L0"]) * 4e3 / (cfg["window_h"] * 3600) <= cfg["budget_ms"]),
        table_default_weight=plan(cfg, lam_med, sig_ln, 0.004)["candidates"],
    )
