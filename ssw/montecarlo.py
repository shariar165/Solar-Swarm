"""Monte Carlo of the v11 replay — the same model as ui/ss-physics-v11.js, in Python.

  python run_all.py montecarlo [--n 1000]      -> results/montecarlo.json

One event = hidden truth drawn from the event priors (ui/event_priors.js, the file the
UI reads) -> noisy observations -> 200-member particle filter with L5-imager updates
-> S5 re-planning from the forecast's uncertainty -> contacts with a tilted planar shock.

This is a line-by-line port: the same random streams (bit-exact mulberry32 and seed
hash), the same draw order and the same formulas, so a seed gives the same event in
both languages up to floating-point rounding (tests/test_montecarlo.py checks that).
Only what the statistics need is ported: random events, no timing solution, no scene.
The member arithmetic is vectorised with numpy; everything that consumes random
numbers stays sequential, in the JS order.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import priors as priors_mod

D2R, H, DAY = math.pi / 180, 3600.0, 86400.0
AU, RSUN = 149.6e6, 695700.0
R0, L1OFF, SUBOFF = 21.5 * RSUN, 1.5e6, .75e6
MP, KB, MU0, GAD = 1.6726e-27, 1.380649e-23, 4e-7 * math.pi, 5 / 3
AMB = dict(n=5.0, B=5.0, T=1e5)

# ASSUMED — must match `A` in ui/ss-physics-v11.js
A = dict(haloS1=(120e3, 80e3), haloMon=(250e3, 150e3), haloForm=(50e3, 30e3), haloPeriod=180 * DAY,
         obsV0=.10, obsHalf=5.0, obsW=40.0, tiltSd=25.0, tiltMax=60.0,
         imgErr=.05, imgDt=6 * H, imgMaxR=.7 * AU, insituErr=.03,
         nEns=200, essFrac=.5, lwA=.97,
         budget=40.0, wDv=.004, uThr=.25, uHyst=.03, noise=.075, L0=3000.0, Lmin=500.0, Lmax=15000.0, nCand=12,
         tFirst=.8 * H, tDec=6 * H, lastBefore=3 * H, arriveBefore=2 * H, minTransfer=1800.0,
         transit=(.8 * DAY, 5 * DAY), machMin=1.2)
OMH = 2 * math.pi / A["haloPeriod"]
SAT0 = [(-750e3, 0.0, 0.0), (-20e3, -240e3, -60e3), (-20e3, 240e3, -60e3), (30e3, 0.0, 250e3), (-20e3, -237e3, -60e3)]
GH = [-2.5 + i * .125 for i in range(41)]
GW = [math.exp(-z * z / 2) for z in GH]
GS = sum(GW)
CAND = [math.floor(A["Lmin"] * (A["Lmax"] / A["Lmin"]) ** (i / (A["nCand"] - 1)) + .5) for i in range(A["nCand"])]

# ---------------------------------------------------------------- random numbers (bit-exact with JS)
M32 = 0xFFFFFFFF


def _imul(a, b):
    return (a * b) & M32


def mulberry32(a):
    a &= M32

    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r


def seed_hash(a, b, c):
    h = (a & M32) ^ 0x9E3779B9
    h = _imul(h ^ (h >> 16), 0x85EBCA6B)
    h ^= _imul((b + 0x632BE5AB) & M32, 0xC2B2AE35)
    h = _imul(h ^ (h >> 13), 0x27D4EB2F)
    h ^= _imul((c + 0x165667B1) & M32, 0x9E3779B1)
    h ^= h >> 15
    return h & M32


class Stream:
    """One independent random stream per purpose (same salts as the JS)."""
    def __init__(self, seed, salt, k=0):
        self.r = mulberry32(seed_hash(seed, salt, k))

    def __call__(self):
        return self.r()

    def g(self):
        u = self.r()
        return math.sqrt(-2 * math.log(u + 1e-12)) * math.cos(2 * math.pi * self.r())


# ---------------------------------------------------------------- drag-based model (scalar and vectorised)
class Traj:
    def __init__(self, v0, w, g, rE=AU, r0=R0):
        self.v0, self.w, self.g, self.rE, self.r0 = v0, w, g, rE, r0
        self.dv = v0 - w
        self.s = 1.0 if self.dv >= 0 else -1.0
        self.k = self.s * g * self.dv
        self.te = self.tAt(rE)

    def r(self, t):
        return self.r0 + self.w * t + self.s / self.g * math.log1p(self.k * t)

    def v(self, t):
        return self.dv / (1 + self.k * t) + self.w

    def tAt(self, R):
        if R <= self.r0:
            return 0.0
        t = (R - self.r0) / max(self.v0, self.w)
        for _ in range(50):
            d = (self.r(t) - R) / self.v(t)
            t -= d
            if abs(d) < 1e-4:
                break
        return math.inf if t > 25 * DAY else t


class TrajV:
    """The same closed form for many members at once."""
    def __init__(self, v0, w, g, rE=AU, r0=R0):
        self.v0, self.w, self.g, self.rE, self.r0 = v0, w, g, rE, r0
        self.dv = v0 - w
        self.s = np.where(self.dv >= 0, 1.0, -1.0)
        self.k = self.s * g * self.dv

    def r(self, t):
        return self.r0 + self.w * t + self.s / self.g * np.log1p(self.k * t)

    def v(self, t):
        return self.dv / (1 + self.k * t) + self.w

    def tAt(self, R):
        R = np.broadcast_to(np.asarray(R, float), self.v0.shape)
        t = (R - self.r0) / np.maximum(self.v0, self.w)
        done = R <= self.r0
        t = np.where(done, 0.0, t)
        for _ in range(50):
            if done.all():
                break
            d = (self.r(t) - R) / self.v(t)
            t = np.where(done, t, t - d)
            done = done | (np.abs(d) < 1e-4)
        return np.where(t > 25 * DAY, np.inf, t)


def cms(n, B, T):
    rho = n * 1e6 * MP
    vA = B * 1e-9 / math.sqrt(MU0 * rho)
    cs = math.sqrt(GAD * 2 * KB * T / MP)
    return math.hypot(vA, cs) / 1e3


CMS1 = cms(AMB["n"], AMB["B"], AMB["T"])


def mach_l1(T, rL1):
    return (T.v(T.tAt(rL1)) - T.w) / CMS1


def eps(L, lam):
    x = 2 * math.pi * L / lam
    return min(1.5, math.hypot(math.sqrt(2) * A["noise"] / x, x * x / 6))


def eeps_p(L, lam, sig):
    s = 0.0
    for z, w in zip(GH, GW):
        s += w * eps(L, lam * math.exp(sig * z))
    return s / GS


# ---------------------------------------------------------------- priors
def load_priors(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "ui" / "event_priors.js"
    P = priors_mod.read_js(path) if path.exists() else {}
    out = dict(meta=P.get("meta", {"data_mode": "assumed"}), historical=P.get("historical", []))
    for k, fb in priors_mod.FALLBACK.items():
        out[k] = P.get(k) or dict(fb, source="assumed", n=0)
    return out


def _clip(x, c):
    return min(max(x, c[0]), c[1]) if c else x


def draw(pr, rnd):
    k = pr["kind"]
    if k == "lognormal":
        return _clip(pr["median"] * math.exp(pr["sigma_ln"] * rnd.g()), pr.get("clip"))
    if k == "normal":
        return _clip(pr["mean"] + pr["sd"] * rnd.g(), pr.get("clip"))
    if k == "uniform":
        return pr["lo"] + (pr["hi"] - pr["lo"]) * rnd()
    if k == "quantiles":
        u, p, q = rnd() * 100, pr["p"], pr["q"]
        j = 0
        while j < len(p) - 2 and u > p[j + 1]:
            j += 1
        return q[j] + (q[j + 1] - q[j]) * (u - p[j]) / (p[j + 1] - p[j])
    return pr["value"]


def _cdf_q(pr, x):
    p, q = pr["p"], pr["q"]
    if x <= q[0]:
        return 0.0
    if x >= q[-1]:
        return 1.0
    j = 0
    while q[j + 1] < x:
        j += 1
    w = q[j + 1] - q[j]
    return (p[j] + ((p[j + 1] - p[j]) * (x - q[j]) / w if w > 0 else 0)) / 100


def pdf(pr, x):
    c = pr.get("clip")
    if c and (x < c[0] or x > c[1]):
        return 0.0
    k = pr["kind"]
    if k == "lognormal":
        if x <= 0:
            return 0.0
        z = math.log(x / pr["median"]) / pr["sigma_ln"]
        return math.exp(-z * z / 2) / x
    if k == "normal":
        z = (x - pr["mean"]) / pr["sd"]
        return math.exp(-z * z / 2)
    if k == "uniform":
        return 1.0 if pr["lo"] <= x <= pr["hi"] else 0.0
    if k == "quantiles":
        h = (pr["q"][-1] - pr["q"][0]) / 40
        return (_cdf_q(pr, x + h) - _cdf_q(pr, x - h)) / (2 * h)
    return 1.0


def ln_spread(pr):
    return pr["sigma_ln"] if pr["kind"] == "lognormal" else .5


# ---------------------------------------------------------------- one event
def _accept(t):
    if not abs(t["lon"]) < t["half"]:
        return False
    T = Traj(t["v0"], t["w"], t["g"], t["rE"])
    if not (A["transit"][0] <= T.te <= A["transit"][1]):
        return False
    return mach_l1(T, t["rE"] - L1OFF) > A["machMin"]


def draw_truth(seed, P):
    for k in range(500):
        rnd = Stream(seed, 1, k)
        t = dict(seed=seed, attempt=k)
        t["v0"] = draw(P["v0"], rnd); t["w"] = draw(P["w"], rnd); t["g"] = draw(P["gamma"], rnd)
        t["half"] = draw(P["half_angle"], rnd); t["lon"] = draw(P["lon"], rnd)
        t["tilt"] = min(max(A["tiltSd"] * rnd.g(), -A["tiltMax"]), A["tiltMax"]) * D2R
        t["phiN"] = rnd() * 2 * math.pi
        t["bz"] = draw(P["bz_sheath"], rnd); t["tau"] = draw(P["tau_c"], rnd)
        t["ph1"] = rnd() * 2 * math.pi; t["ph2"] = rnd() * 2 * math.pi; t["ph3"] = rnd() * 2 * math.pi
        t["rE"] = AU
        if _accept(t):
            return t
    raise RuntimeError(f"no Earth-impacting event found for seed {seed}")


def _s1_yz(t, tt):
    return A["haloS1"][0] * np.sin(t["ph1"] + OMH * tt), A["haloS1"][1] * np.cos(t["ph1"] + OMH * tt)


def _mon_yz(t, tt):
    return A["haloMon"][0] * np.sin(t["ph2"] + OMH * tt), A["haloMon"][1] * np.cos(t["ph2"] + OMH * tt)


def _form_drift(t, tt):
    return (A["haloForm"][0] * (math.sin(t["ph3"] + OMH * tt) - math.sin(t["ph3"])),
            A["haloForm"][1] * (math.cos(t["ph3"] + OMH * tt) - math.cos(t["ph3"])))


def contact_v(T, rL1, n, x, yz):
    """Planar shock (normal n = (nx, ny, nz), arrays or scalars) reaches a craft at x and yz(tt)."""
    nx, ny, nz = n
    y, z = yz(0.0)
    tt = T.tAt(rL1 + (nx * x + ny * y + nz * z) / nx)
    for _ in range(3):
        y, z = yz(tt)
        tt = T.tAt(rL1 + (nx * x + ny * y + nz * z) / nx)
    return tt


def _wstat(x, w):
    m = float(np.dot(w, x))
    return m, math.sqrt(float(np.dot(w, (x - m) ** 2)))


def _norm_w(w):
    s = float(w.sum())
    if not (s > 0) or not math.isfinite(s):
        return False
    w /= s
    return True


def _ess(w):
    return 1.0 / float(np.dot(w, w))


def forecast(E, P):
    N, ob, t, rE, rL1 = A["nEns"], E["obs"], E["truth"], E["rE"], E["rL1"]
    rnd = Stream(t["seed"], 3)
    lnV, sv = math.log(ob["v0"]), A["obsV0"]

    def ok(v, w, g):
        T = Traj(v, w, g, rE)
        return A["transit"][0] <= T.te <= A["transit"][1] and mach_l1(T, rL1) > A["machMin"]

    lv, ww, lg, th, ph, wt = (np.zeros(N) for _ in range(6))
    for i in range(N):
        k = 0
        while True:
            v = math.exp(lnV + sv * rnd.g()); w = ob["w"] + A["obsW"] * rnd.g(); g = draw(P["gamma"], rnd)
            if ok(v, w, g):
                break
            k += 1
            if k >= 60:
                break
        zl, zq = (ob["v0"] - v) / (sv * v), (math.log(v) - lnV) / sv
        lv[i], ww[i], lg[i] = math.log(v), w, math.log(g)
        th[i] = min(max(A["tiltSd"] * rnd.g(), -A["tiltMax"]), A["tiltMax"]) * D2R
        ph[i] = rnd() * 2 * math.pi
        wt[i] = pdf(P["v0"], v) * math.exp(-zl * zl / 2) / math.exp(-zq * zq / 2) * pdf(P["w"], w) if k < 60 else 0.0
    if not _norm_w(wt):
        wt[:] = 1 / N
    M = dict(lv=lv, w=ww, lg=lg, th=th, ph=ph, wt=wt)

    def derive():
        T = TrajV(np.exp(M["lv"]), M["w"], np.exp(M["lg"]), rE)
        n = (np.cos(M["th"]), np.sin(M["th"]) * np.cos(M["ph"]), np.sin(M["th"]) * np.sin(M["ph"]))
        M["T"] = T
        M["te"] = T.tAt(rE)
        M["vL1"] = T.v(T.tAt(rL1))
        a = contact_v(T, rL1, n, -750e3, lambda tt: _s1_yz(t, tt))
        b = contact_v(T, rL1, n, 0.0, lambda tt: _mon_yz(t, tt))
        M["tS1"], M["gain"] = a, (b - a) / 60

    def stats(tk, kind, extra):
        w = M["wt"]
        e, v, gn, s1 = _wstat(M["te"], w), _wstat(M["vL1"], w), _wstat(M["gain"], w), _wstat(M["tS1"], w)
        h = dict(t=tk, kind=kind, eta_m=e[0], eta_s=e[1], vL1_m=v[0], vL1_s=v[1], gain_m=gn[0], gain_s=gn[1],
                 tS1_m=s1[0], tS1_s=s1[1], g_m=math.exp(_wstat(M["lg"], w)[0]), w_m=_wstat(M["w"], w)[0], ess=_ess(w), **extra)
        E["hist"].append(h)
        return h

    E["hist"] = []
    derive()
    stats(.3 * H, "initial", {})
    ri, rj, T = Stream(t["seed"], 4), Stream(t["seed"], 5), E["T"]
    tk = A["imgDt"]
    while T.r(tk) < A["imgMaxR"] and tk < T.tAt(rL1 - SUBOFF) - H:
        ro = T.r(tk) * (1 + A["imgErr"] * ri.g())
        r = M["T"].r(tk)
        sg = A["imgErr"] * r
        z = (ro - r) / sg
        M["wt"] *= np.exp(-z * z / 2) / sg
        if not _norm_w(M["wt"]):
            M["wt"][:] = 1 / N
        es = _ess(M["wt"])
        res = es < A["essFrac"] * N
        if res:
            _resample(M, rj, ok)
            derive()
        stats(tk, "imager", dict(r_obs=ro, ess_before=es, resampled=res))
        tk += A["imgDt"]


def _resample(M, rnd, ok):
    N, a = len(M["wt"]), A["lwA"]
    hs = math.sqrt(1 - a * a)
    mean, sd = {}, {}
    for k in ("lv", "w", "lg"):
        mean[k], sd[k] = _wstat(M[k], M["wt"])
    wt = M["wt"].tolist()
    idx, c, j = [0] * N, wt[0], 0
    u0 = rnd() / N
    for i in range(N):
        u = u0 + i / N
        while u > c and j < N - 1:
            j += 1
            c += wt[j]
        idx[i] = j
    old = {k: M[k].copy() for k in ("lv", "w", "lg", "th", "ph")}
    for i in range(N):
        p, k = idx[i], 0
        while True:
            v = a * old["lv"][p] + (1 - a) * mean["lv"] + hs * sd["lv"] * rnd.g()
            w = a * old["w"][p] + (1 - a) * mean["w"] + hs * sd["w"] * rnd.g()
            g = a * old["lg"][p] + (1 - a) * mean["lg"] + hs * sd["lg"] * rnd.g()
            if ok(math.exp(v), w, math.exp(g)):
                break
            k += 1
            if k >= 20:
                break
        if k >= 20:
            v, w, g = old["lv"][p], old["w"][p], old["lg"][p]
        M["lv"][i], M["w"][i], M["lg"][i], M["th"][i], M["ph"][i] = v, w, g, old["th"][p], old["ph"][p]
    M["wt"][:] = 1 / N


# ---------------------------------------------------------------- S5 re-planning
def track_at(seg, t):
    s = seg[0]
    for x in seg:
        if t < x["t0"]:
            break
        s = x
    d = min(max(t, s["t0"]), s["t1"]) - s["t0"]
    L = s["L0"] + s["v0"] * d + .5 * s["a"] * d * d
    v = s["v0"] + s["a"] * d
    return (L + v * (t - s["t1"]), v) if t > s["t1"] else (L, v)


def dv_used(seg, t):
    dv = 0.0
    for s in seg:
        if t <= s["t0"]:
            break
        dv += abs(s["a"]) * (min(t, s["t1"]) - s["t0"])
    return dv * 1e3


def hist_at(E, t):
    h = E["hist"][0]
    for x in E["hist"]:
        if x["t"] <= t:
            h = x
    return h


def plan(E, P):
    tau, L0 = P["tau_c"], A["L0"]
    s_tau = ln_spread(tau)
    E["seg"] = [dict(t0=-1e9, t1=1e9, L0=L0, v0=0.0, a=0.0)]
    E["dec"], E["target"] = [], L0

    def decide(td):
        h = hist_at(E, td)
        V, sV = h["vL1_m"], h["vL1_s"]
        lam, sig = V * tau["median"], math.hypot(s_tau, sV / V)
        tArr = h["tS1_m"] - 2 * h["tS1_s"] - A["arriveBefore"]
        L, v = track_at(E["seg"], td)
        used = dv_used(E["seg"], td)
        left = A["budget"] - used
        commit = dv_used(E["seg"], 1e9) - used
        Uh = eeps_p(E["target"], lam, sig)
        Jh = Uh + A["wDv"] * commit
        aNow = next((s["a"] for s in E["seg"] if s["t0"] <= td < s["t1"]), 0.0)
        tStop = abs(v / aNow) if v != 0 and aNow != 0 else 0.0
        Lstop, T2 = L + v * tStop / 2, tArr - td - tStop
        cands = []
        for Lc in CAND:
            dv = abs(v) * 1e3 + 4 * abs(Lc - Lstop) * 1e3 / T2
            U = eeps_p(Lc, lam, sig)
            cands.append(dict(L=Lc, U=U, dv=dv, J=U + A["wDv"] * dv, ok=T2 > A["minTransfer"] and dv <= left))
        best = None
        for c in cands:
            if c["ok"] and (best is None or c["J"] < best["J"]):
                best = c
        D = dict(t=td, V=V, sV=sV, lam=lam, sig=sig, Uh=Uh, Jh=Jh, target=E["target"], left=left)
        if Uh <= A["uThr"] or best is None or best["L"] == E["target"] or Uh - best["U"] <= A["uHyst"] or not best["J"] < Jh:
            D["action"] = "hold"
        else:
            D["action"], D["pick"] = "retarget", best
            keep = [dict(s) for s in E["seg"] if s["t0"] < td]
            keep[-1]["t1"] = td
            t0 = td
            if tStop > 0:
                a = -math.copysign(1, v) * abs(aNow)
                keep.append(dict(t0=t0, t1=t0 + tStop, L0=L, v0=v, a=a))
                t0 += tStop
                L, v = Lstop, 0.0
            dL = best["L"] - L
            aT = 4 * dL / (T2 * T2)
            tm = t0 + T2 / 2
            keep.append(dict(t0=t0, t1=tm, L0=L, v0=0.0, a=aT))
            keep.append(dict(t0=tm, t1=tArr, L0=L + dL / 2, v0=aT * T2 / 2, a=-aT))
            keep.append(dict(t0=tArr, t1=1e9, L0=best["L"], v0=0.0, a=0.0))
            E["seg"], E["target"] = keep, best["L"]
        E["dec"].append(D)

    times = [A["tFirst"]] + [h["t"] for h in E["hist"] if h["kind"] == "imager"]
    t = times[-1] + A["tDec"]
    while True:
        if t >= hist_at(E, t)["tS1_m"] - A["lastBefore"]:
            break
        times.append(t)
        t += A["tDec"]
    tLast = hist_at(E, times[-1])["tS1_m"] - A["lastBefore"]
    if tLast > times[-1] + 600:
        times.append(tLast)
    tSubAxis = E["T"].tAt(E["rL1"] - SUBOFF)
    for td in times:
        if td < tSubAxis:
            decide(td)
    E["burns"] = sum(d["action"] == "retarget" for d in E["dec"])


def contacts(E):
    t, T, rL1 = E["truth"], E["T"], E["rL1"]
    n = (math.cos(t["tilt"]), math.sin(t["tilt"]) * math.cos(t["phiN"]), math.sin(t["tilt"]) * math.sin(t["phiN"]))
    E["tSub"] = contact_v(T, rL1, n, -750e3, lambda tt: _s1_yz(t, tt))
    E["tMon"] = contact_v(T, rL1, n, 0.0, lambda tt: _mon_yz(t, tt))

    def s5_yz(tt):
        dy, dz = _form_drift(t, tt)
        return SAT0[1][1] + dy + track_at(E["seg"], tt)[0], SAT0[1][2] + dz
    E["tc5"] = contact_v(T, rL1, n, SAT0[1][0], s5_yz)
    E["gain"] = (E["tMon"] - E["tSub"]) / 60
    E["tL1axis"] = T.tAt(rL1)
    E["tSubAxis"] = T.tAt(rL1 - SUBOFF)


def after_contact(E):
    T, t = E["T"], E["truth"]
    rnd = Stream(t["seed"], 7)
    hp = hist_at(E, E["tSub"] - 1)
    Vm = T.v(E["tSub"]) * (1 + A["insituErr"] * rnd.g())
    rS1 = E["rL1"] - SUBOFF
    a = Traj(Vm, hp["w_m"], hp["g_m"], E["rE"], rS1)
    b = Traj(Vm * (1 + A["insituErr"]), hp["w_m"], hp["g_m"], E["rE"], rS1)
    E["hist"].append(dict(t=E["tSub"], kind="insitu", eta_m=E["tSub"] + a.te, eta_s=abs(b.te - a.te)))
    E["lamTrue"] = T.v(E["tL1axis"]) * t["tau"]
    Lc, vc = track_at(E["seg"], E["tc5"])
    E["Lfinal"], E["movingAtContact"] = Lc, abs(vc) > 1e-9
    E["errL"], E["errL0"] = eps(Lc, E["lamTrue"]), eps(A["L0"], E["lamTrue"])


def summarize(E):
    tE, tSub = E["T"].te, E["tSub"]
    pre = [h for h in E["hist"] if h["kind"] != "insitu"]

    def at(tt):
        h = hist_at(E, tt)
        return h if h["t"] <= tt and h["kind"] != "insitu" and tt < tSub else None

    err = lambda h: abs(h["eta_m"] - tE) if h else math.nan
    z = lambda h: (h["eta_m"] - tE) / h["eta_s"] if h else math.nan
    t = E["truth"]
    return dict(seed=t["seed"], v0=t["v0"], w=t["w"], g=t["g"], tilt=t["tilt"] / D2R, transit_h=tE / H,
                gain=E["gain"], gainAxis=(E["tL1axis"] - E["tSubAxis"]) / 60, vL1=E["T"].v(E["tL1axis"]),
                gainExp=E["hist"][0]["gain_m"], gainExpS=E["hist"][0]["gain_s"],
                err0=err(E["hist"][0]), err12=err(at(12 * H)), err24=err(at(24 * H)), errLast=err(pre[-1]),
                errPost=abs(E["hist"][-1]["eta_m"] - tE), z0=z(E["hist"][0]), zLast=z(pre[-1]), zs=[z(h) for h in pre],
                sig=[h["eta_s"] for h in pre], nImg=len(pre) - 1, dv=dv_used(E["seg"], 1e9), burns=E["burns"],
                moved=E["burns"] > 0, Lfinal=E["Lfinal"], lamTrue=E["lamTrue"], errL=E["errL"], errL0=E["errL0"],
                movingAtContact=E["movingAtContact"])


def make_event(seed, P):
    t = draw_truth(seed, P)
    rnd = Stream(seed, 2)
    obs = dict(v0=t["v0"] * (1 + A["obsV0"] * rnd.g()), half=t["half"] + A["obsHalf"] * rnd.g(), w=t["w"] + A["obsW"] * rnd.g())
    E = dict(truth=t, T=Traj(t["v0"], t["w"], t["g"], t["rE"]), rE=t["rE"], rL1=t["rE"] - L1OFF, obs=obs)
    forecast(E, P)
    plan(E, P)
    contacts(E)
    after_contact(E)
    return summarize(E)


# ---------------------------------------------------------------- the batch and its statistics
def _q(a, p):
    a = np.asarray([x for x in a if x is not None and math.isfinite(x)], float)
    return float(np.percentile(a, p * 100)) if a.size else None


def _st(a):
    return dict(median=_q(a, .5), p10=_q(a, .1), p90=_q(a, .9), n=int(sum(1 for x in a if x is not None and math.isfinite(x))))


def stats(S):
    cov = lambda zs: float(np.mean([abs(z) <= 2 for z in zs if math.isfinite(z)]))
    allz = [z for s in S for z in s["zs"]]
    logs = [math.log(s["sig"][k] / s["sig"][k - 1]) for s in S for k in range(1, len(s["sig"]))]
    final = {}
    for s in S:
        k = str(int(round(s["Lfinal"])))
        final[k] = final.get(k, 0) + 1
    return dict(
        n_events=len(S),
        gain_min=_st([s["gain"] for s in S]),
        gain_on_axis_min=_st([s["gainAxis"] for s in S]),
        shock_speed_L1_kms=_st([s["vL1"] for s in S]),
        gain_expected_at_T0_min=_st([s["gainExp"] for s in S]),
        eta_err_h=dict(T0=_st([s["err0"] / H for s in S]), T12=_st([s["err12"] / H for s in S]),
                       T24=_st([s["err24"] / H for s in S]), last_before_contact=_st([s["errLast"] / H for s in S])),
        eta_err_after_S1_min=_st([s["errPost"] / 60 for s in S]),
        coverage_2sigma=dict(initial=cov([s["z0"] for s in S]), last_before_contact=cov([s["zLast"] for s in S]), all_updates=cov(allz)),
        sigma_eta_mean_log_ratio_per_update=float(np.mean(logs)) if logs else None,
        s5=dict(moved_pct=100 * float(np.mean([s["moved"] for s in S])), dv_ms=_st([s["dv"] for s in S]),
                mean_dv_ms=float(np.mean([s["dv"] for s in S])), burns=_st([s["burns"] for s in S]),
                burns_count={str(b): int(sum(s["burns"] == b for s in S)) for b in sorted({s["burns"] for s in S})},
                final_separation_km=final, moving_at_contact=int(sum(s["movingAtContact"] for s in S)),
                small_scale_error_final=_st([s["errL"] for s in S]), small_scale_error_if_held=_st([s["errL0"] for s in S])))


def run(n=1000, seed0=1, priors_path=None, P=None):
    P = P or load_priors(priors_path)
    t0 = time.time()
    S = [make_event(seed0 + i, P) for i in range(n)]
    out = dict(meta=dict(model="ui/ss-physics-v11.js (Python port: ssw/montecarlo.py)", seeds=[seed0, seed0 + n - 1],
                         priors=P["meta"].get("data_mode", "assumed"), runtime_s=round(time.time() - t0, 1),
                         assumed={k: (list(v) if isinstance(v, tuple) else v) for k, v in A.items()}),
               **stats(S))
    return out, S


def save(out, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1), encoding="utf-8")
