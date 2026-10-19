"""Weakness #5 — a real ML model with enough data: shock detection in L1 solar wind.

Data   : 1-min L1 solar wind (OMNI via CDAWeb HAPI): speed V, density N, |B|, Bz.
Labels : DONKI interplanetary shocks at Earth (NASA's own catalogue).
Model  : gradient-boosted trees on upstream/downstream window features.
Split  : chronological (train <= SHOCK_TRAIN_END, test after) — no leakage.
Baseline: the rule currently in the live UI (>= 50 km/s jump vs 15-min mean).

Label alignment: DONKI shock times refer to the L1 spacecraft, while OMNI is shifted
to the bow shock (~30-60 min later). Each label is therefore moved to the minute with
the strongest speed jump inside [t_DONKI - 30 min, t_DONKI + 90 min].

Caveat reported with the results: DONKI's shock list is not complete, so some
"false alarms" may be real, uncatalogued shocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

UP, DN = 20, 8          # upstream / downstream window, minutes (decision delay = DN)
TOL = pd.Timedelta(minutes=20)
DEAD = pd.Timedelta(hours=3)


def _grid(l1: pd.DataFrame) -> pd.DataFrame:
    df = l1[["V", "N", "B", "BZ"]].copy()
    df = df.resample("1min").mean()
    return df.interpolate(limit=5)


def features(l1: pd.DataFrame) -> pd.DataFrame:
    d = _grid(l1)
    f = pd.DataFrame(index=d.index)
    for c in ["V", "N", "B"]:
        up = d[c].rolling(UP, min_periods=UP // 2).median().shift(1)
        dn = d[c][::-1].rolling(DN, min_periods=DN // 2).median()[::-1].shift(-1)
        f[f"{c}_up"] = up
        f[f"d{c}"] = dn - up
        f[f"r{c}"] = dn / up.where(up.abs() > 1e-3)
        f[f"{c}_sd_up"] = d[c].rolling(UP, min_periods=UP // 2).std().shift(1)
    f["dV_2min"] = d.V.shift(-2) - d.V.rolling(15, min_periods=8).mean().shift(1)
    f["dBz"] = d.BZ[::-1].rolling(DN, min_periods=4).median()[::-1].shift(-1) - d.BZ.rolling(UP, min_periods=10).median().shift(1)
    f = f.astype("float32")
    f["valid"] = d[["V", "N", "B"]].notna().all(axis=1)
    return f


def align_labels(shocks: pd.Series, f: pd.DataFrame) -> pd.DatetimeIndex:
    out = []
    for t in pd.to_datetime(shocks, utc=True):
        w = f.loc[t - pd.Timedelta(minutes=30): t + pd.Timedelta(minutes=90), "dV"]
        if w.notna().sum() > 30:
            out.append(w.idxmax())
    return pd.DatetimeIndex(out)


def _events_from_score(score: pd.Series, thr: float) -> pd.DatetimeIndex:
    hit = score[score >= thr]
    ev, last = [], None
    for t, s in hit.items():
        if last is None or t - last > DEAD:
            ev.append(t)
            last = t
    return pd.DatetimeIndex(ev)


def _evaluate(det: pd.DatetimeIndex, truth: pd.DatetimeIndex, days: float) -> dict:
    tp, dt = 0, []
    used = set()
    for t in truth:
        m = [d for d in det if abs(d - t) <= TOL and d not in used]
        if m:
            d = min(m, key=lambda x: abs(x - t))
            used.add(d)
            tp += 1
            dt.append((d - t).total_seconds() / 60)
    fp = len(det) - len(used)
    return dict(n_true=int(len(truth)), detected=int(tp), pod=tp / max(len(truth), 1),
                false_alarms=int(fp), false_alarms_per_30d=fp / max(days, 1) * 30,
                precision=tp / max(len(det), 1),
                median_timing_err_min=float(np.median(np.abs(dt))) if dt else None)


def rule_baseline(l1: pd.DataFrame) -> pd.Series:
    """Exactly the live-UI rule: speed 2 min later minus previous 15-min mean >= 50 km/s."""
    d = _grid(l1)
    return (d.V.shift(-2) - d.V.rolling(15, min_periods=8).mean().shift(1)) / 50.0


def run(l1: pd.DataFrame, shock_times: pd.Series, train_end: str, seed: int = 0):
    f = features(l1)
    f = f[f.valid]
    labels = align_labels(shock_times, f)
    y = pd.Series(0, index=f.index)
    for t in labels:
        y.loc[t - pd.Timedelta(minutes=2): t + pd.Timedelta(minutes=2)] = 1
    X = f.drop(columns="valid")
    cut = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    tr, te = X.index < cut, X.index >= cut

    # training set: all positives + 40 random negatives per positive, excluding +/-3 h of shocks
    rng = np.random.default_rng(seed)
    near = pd.Series(False, index=X.index)
    for t in labels:
        near.loc[t - pd.Timedelta(hours=3): t + pd.Timedelta(hours=3)] = True
    pos = np.where(tr & (y.values == 1))[0]
    neg_pool = np.where(tr & (y.values == 0) & ~near.values)[0]
    hard = np.where(tr & (y.values == 0) & near.values)[0]
    neg = rng.choice(neg_pool, size=min(len(neg_pool), 40 * max(len(pos), 1)), replace=False)
    idx = np.r_[pos, neg, hard[:: max(1, len(hard) // (10 * max(len(pos), 1)))]]
    if len(pos) < 10:
        raise RuntimeError(f"only {len(pos)} positive minutes in training data — fetch more years")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(X.iloc[idx], y.iloc[idx])

    # pick the event threshold on the last training year (validation slice)
    score_all = pd.Series(clf.predict_proba(X)[:, 1], index=X.index)
    val = X.index[tr][X.index[tr] >= cut - pd.Timedelta(days=365)]
    val_truth = labels[(labels >= val.min()) & (labels < cut)] if len(val) else labels[:0]
    best_thr, best_f1 = 0.5, -1
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        m = _evaluate(_events_from_score(score_all.loc[val], thr), val_truth, len(val) / 1440)
        f1 = 2 * m["pod"] * m["precision"] / max(m["pod"] + m["precision"], 1e-9)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1

    test_idx = X.index[te]
    days = len(test_idx) / 1440
    truth_te = labels[labels >= cut]
    ml = _evaluate(_events_from_score(score_all.loc[test_idx], best_thr), truth_te, days)
    rb = rule_baseline(l1).reindex(test_idx)
    rule = _evaluate(_events_from_score(rb, 1.0), truth_te, days)

    imp = {}
    try:
        from sklearn.inspection import permutation_importance
        sub = np.r_[np.where(te & (y.values == 1))[0], rng.choice(np.where(te)[0], 3000, replace=False)]
        pi = permutation_importance(clf, X.iloc[sub], y.iloc[sub], n_repeats=3, random_state=seed, scoring="average_precision")
        imp = dict(sorted(zip(X.columns, map(float, pi.importances_mean)), key=lambda kv: -kv[1])[:6])
    except Exception:
        pass

    res = dict(
        data="L1 1-min (OMNI), labels = DONKI interplanetary shocks at Earth",
        train_period=[str(X.index[tr].min().date()), str(X.index[tr].max().date())],
        test_period=[str(test_idx.min().date()), str(test_idx.max().date())] if len(test_idx) else None,
        n_shocks_train=int((labels < cut).sum()), n_shocks_test=int(len(truth_te)),
        test_days=round(days, 1), decision_delay_min=DN, threshold=best_thr,
        ml=ml, rule_baseline=rule, top_features=imp,
        caveat="DONKI's shock list is incomplete, so some false alarms may be real shocks.",
    )
    return res, clf
