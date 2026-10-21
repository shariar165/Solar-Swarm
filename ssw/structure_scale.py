"""Weakness #2 — replace the ASSUMED structure-size prior (lambda = 9,000 km) with data.

For each real CME-driven shock, take high-resolution magnetometer data from the
sheath behind it (5 min to 3 h after the shock) and measure the correlation length
of the magnetic fluctuations:

  1. detrend each B component with a running mean of D minutes
  2. autocorrelation of the residual; tau_c = first lag where ACF < 1/e
  3. Taylor's frozen-in hypothesis: L = V_sheath * tau_c

The detrending window D caps the largest scale you can see, so results are reported
for several D (sensitivity). The optimizer uses D = 30 min: on synthetic data with a
known 60,000 km length it recovers ~47,000 km (shorter windows underestimate more).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DETREND_MIN = [5, 10, 30]
DEFAULT_D = 30


def corr_length_km(b: pd.DataFrame, v_kms: float, detrend_min: int) -> float | None:
    if not np.isfinite(v_kms):
        return None
    tau = corr_time_s(b, detrend_min)
    return None if tau is None else float(v_kms * tau)


def corr_time_s(b: pd.DataFrame, detrend_min: int) -> float | None:
    """Median over the B components of the ACF 1/e crossing, in seconds (tau_c)."""
    if b.empty:
        return None
    b = b.dropna()
    if len(b) < 600:
        return None
    dt = float(np.median(np.diff(b.index.values).astype("timedelta64[ms]").astype(float))) / 1000.0
    if not (0 < dt <= 10):
        return None
    b = b.resample(f"{max(int(round(dt)), 1)}s").mean().interpolate(limit=10).dropna()
    win = max(int(detrend_min * 60 / max(dt, 1)), 5)
    taus = []
    for c in b.columns:
        x = (b[c] - b[c].rolling(win, center=True, min_periods=win // 2).mean()).dropna().values
        if x.size < 4 * win:
            continue
        x = x - x.mean()
        n = int(2 ** np.ceil(np.log2(2 * x.size)))
        fx = np.fft.rfft(x, n)
        acf = np.fft.irfft(fx * np.conj(fx), n)[: x.size]
        acf /= acf[0]
        below = np.where(acf < 1 / np.e)[0]
        if below.size:
            taus.append(below[0] * max(dt, 1))
    if not taus:
        return None
    return float(np.median(taus))


def summarize(lengths: dict[int, list[float]]) -> dict:
    res = {}
    for d, L in lengths.items():
        L = np.array([x for x in L if x and np.isfinite(x) and x > 0])
        if L.size < 3:
            res[str(d)] = None
            continue
        ln = np.log(L)
        res[str(d)] = dict(n=int(L.size), median_km=float(np.exp(np.median(ln))),
                           sigma_ln=float(np.std(ln)), p10_km=float(np.percentile(L, 10)),
                           p90_km=float(np.percentile(L, 90)))
    return dict(by_detrend_min=res, default_detrend_min=DEFAULT_D,
                method="sheath (shock+5 min to +3 h), ACF 1/e crossing x sheath speed (Taylor hypothesis)",
                caveat="Correlation length is a proxy for the structure size the scout pair must resolve; "
                       "the detrending window caps the largest scale measured.")
