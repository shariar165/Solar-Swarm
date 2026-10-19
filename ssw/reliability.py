"""Weakness #3 — what a 5th spacecraft is worth, as a number.

Exact enumeration (no sampling noise) over every survive/fail combination of the
craft. "3D capability" = at least one surviving 4-craft subset whose tetrahedron
quality Q exceeds 0.05 (Q = 1 for a regular tetrahedron, 0 for flat).

Failure model (ASSUMED, swept): each craft independently has an early-failure chance
p0 at launch, then a constant annual failure probability p.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

Q_MIN = 0.05
P_ANNUAL = [0.05, 0.10, 0.20]
P_EARLY = [0.0, 0.10]
YEARS = [1, 3, 5]


def tet_q(pts) -> float:
    p = np.asarray(pts, float)
    vol = abs(np.linalg.det(np.c_[p[1] - p[0], p[2] - p[0], p[3] - p[0]])) / 6
    s = sum(np.sum((p[i] - p[j]) ** 2) for i in range(4) for j in range(i + 1, 4))
    L = math.sqrt(s / 6)
    return float(6 * math.sqrt(2) * vol / L ** 3) if L > 0 else 0.0


def capable(alive: tuple, pos: dict) -> bool:
    return any(tet_q([pos[k] for k in sub]) > Q_MIN for sub in itertools.combinations(alive, 4))


def p_capable(pos: dict, p_fail: float) -> float:
    names = list(pos)
    tot = 0.0
    for state in itertools.product([0, 1], repeat=len(names)):
        alive = tuple(n for n, s in zip(names, state) if s)
        if len(alive) < 4:
            continue
        pr = math.prod((1 - p_fail) if s else p_fail for s in state)
        if capable(alive, pos):
            tot += pr
    return tot


def single_loss_tolerance(pos: dict) -> dict:
    out = {}
    for k in pos:
        rest = tuple(n for n in pos if n != k)
        out[k] = len(rest) >= 4 and capable(rest, pos)
    return out


def best_spare_position(base: dict) -> tuple:
    """Grid search for a 5th-craft position that tolerates any single loss."""
    best, best_score = None, -1
    for x in np.arange(-700e3, 101e3, 100e3):
        for y in np.arange(-300e3, 301e3, 100e3):
            for z in np.arange(-300e3, 301e3, 100e3):
                pos = dict(base, S5=(x, y, z))
                score = min(max((tet_q([pos[n] for n in sub])
                                 for sub in itertools.combinations([n for n in pos if n != k], 4)), default=0)
                            for k in pos)
                if score > best_score:
                    best, best_score = (float(x), float(y), float(z)), score
    return best, best_score


def run(sat: dict) -> dict:
    four = {k: sat[k] for k in ["S1", "S2", "S3", "S4"]}
    scout = dict(sat)
    spare_pos, spare_score = best_spare_position(four)
    spare = dict(four, S5=spare_pos)
    configs = {"4 craft": four, "5 craft — S5 as scout (current)": scout,
               "5 craft — S5 as spare (best spot)": spare}
    table = []
    for p in P_ANNUAL:
        for p0 in P_EARLY:
            row = dict(p_annual=p, p_early=p0)
            for name, pos in configs.items():
                row[name] = {f"y{y}": p_capable(pos, 1 - (1 - p0) * (1 - p) ** y) for y in YEARS}
            table.append(row)
    return dict(
        q_min=Q_MIN,
        single_loss_tolerated={n: single_loss_tolerance(pos) for n, pos in configs.items()},
        spare_position_km=[v / 1e3 for v in spare_pos], spare_worst_case_q=spare_score,
        tetra_q={"S1-S4": tet_q(list(four.values()))},
        table=table,
        assumption="independent failures; p0 at launch then constant annual probability p (swept)",
        trade_off="S5 can be a scout (3,000 km from S2, measures small scales) or a spare "
                  "(placed to replace any craft) — not both.",
    )
