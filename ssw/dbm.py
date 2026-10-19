"""Drag-Based Model (Vrsnak et al. 2013), closed form.

dv/dt = -gamma (v - w) |v - w|
Units: r [km], v [km/s], gamma [1/km], t [s].
"""
from __future__ import annotations

import math
import numpy as np

RSUN_KM = 695_700.0
R0_KM = 21.5 * RSUN_KM            # DONKI CME analyses report speed at 21.5 solar radii
AU_KM = 149_597_870.7
L1_OFFSET_KM = 1.5e6              # L1 is ~1.5 million km sunward of Earth
BOW_SHOCK_KM = 9.0e4              # rough Earth bow-shock standoff (~14 R_E)
ECC = 0.0167


def sun_earth_km(t) -> float:
    """Sun-Earth distance on a given date (first-order Kepler; good to ~0.01%).

    Earth's distance changes by ~5 million km over the year, which is ~2-3 h of
    CME travel time, so a fixed 1 AU would add a seasonal error to every hindcast.
    """
    import pandas as pd
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    perihelion = pd.Timestamp("2000-01-03T05:00Z")
    m = 2 * math.pi * ((ts - perihelion).total_seconds() / 86400.0) / 365.2596
    return AU_KM * (1 - ECC * math.cos(m))


def l1_km(t) -> float:
    return sun_earth_km(t) - L1_OFFSET_KM


def r_of_t(t, v0, w, gamma, r0=R0_KM):
    """Heliocentric distance of the CME front after time t (closed form)."""
    t = np.asarray(t, float)
    dv = v0 - w
    s = 1.0 if dv >= 0 else -1.0
    return r0 + w * t + s / gamma * np.log1p(s * gamma * dv * t)


def v_of_t(t, v0, w, gamma):
    t = np.asarray(t, float)
    dv = v0 - w
    s = 1.0 if dv >= 0 else -1.0
    return dv / (1.0 + s * gamma * dv * t) + w


def arrival_time(v0, w, gamma, r_target, r0=R0_KM, t_max=20 * 86400.0):
    """Transit time [s] from r0 to r_target. None if not reached within t_max."""
    if v0 <= 0 or w <= 0 or gamma <= 0:
        return None
    if r_of_t(t_max, v0, w, gamma, r0) < r_target:
        return None
    lo, hi = 0.0, t_max
    for _ in range(80):                      # bisection: r(t) is monotonic
        mid = 0.5 * (lo + hi)
        if r_of_t(mid, v0, w, gamma, r0) < r_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def arrival_speed(v0, w, gamma, r_target, r0=R0_KM):
    t = arrival_time(v0, w, gamma, r_target, r0)
    return None if t is None else float(v_of_t(t, v0, w, gamma))


def gamma_for_arrival(v0, w, t_obs, r_target, r0=R0_KM, lo=1e-11, hi=1e-5):
    """Per-event gamma that reproduces an observed transit time (None if impossible).

    For v0 > w a larger gamma means more braking (later arrival); for v0 < w the
    wind accelerates the CME, so a larger gamma means earlier arrival.
    """
    def f(g):
        t = arrival_time(v0, w, g, r_target, r0)
        return (t if t is not None else 1e12) - t_obs
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    a, b = math.log(lo), math.log(hi)
    for _ in range(70):
        m = 0.5 * (a + b)
        fm = f(math.exp(m))
        if (fm > 0) == (flo > 0):
            a, flo = m, fm
        else:
            b = m
    return math.exp(0.5 * (a + b))


def ensemble(v0, w, gamma, r_target, n=200, sv=0.10, sw=50.0, sg=0.5, seed=0):
    """DBEM-style ensemble: perturb v0 (relative), w (absolute), gamma (log-normal).

    Returns (mean transit s, std s, fraction of members that arrive).
    """
    rng = np.random.default_rng(seed)
    vs = v0 * (1 + sv * rng.standard_normal(n))
    ws = np.clip(w + sw * rng.standard_normal(n), 200, 900)
    gs = gamma * np.exp(sg * rng.standard_normal(n))
    ts = [arrival_time(a, b, c, r_target) for a, b, c in zip(vs, ws, gs)]
    ts = np.array([x for x in ts if x is not None])
    if ts.size < 0.5 * n:
        return None, None, ts.size / n
    return float(ts.mean()), float(ts.std()), ts.size / n
