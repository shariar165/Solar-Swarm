// =====================================================================================
//  SOLAR SWARM v11 — physics part (no DOM, no particles).
//  Used by ui/solar-swarm-v11.html (window.SSP), by the Monte Carlo batch, and by the
//  Node tests (module.exports). ssw/montecarlo.py is a line-by-line Python port.
//
//  One event = hidden truth (drawn from the event priors) → what the forecaster sees
//  (noisy observations + an L5 imager) → a 200-member particle-filter forecast → S5
//  re-planning from the forecast's uncertainty → contacts with a tilted planar shock.
// =====================================================================================
(function (root) {
function SSPhysFactory() {
'use strict';
const D2R = Math.PI / 180, H = 3600, DAY = 86400;
const AU = 149.6e6, RSUN = 695700, R0 = 21.5 * RSUN, L1OFF = 1.5e6, SUBOFF = .75e6;
const OMEGA = 2 * Math.PI / (25.38 * DAY);
const MP = 1.6726e-27, KB = 1.380649e-23, MU0 = 4e-7 * Math.PI, GAD = 5 / 3;
const AMB = {n: 5, B: 5, T: 1e5};                       // ambient wind at 1 AU (ASSUMED typical)

// Everything below is ASSUMED (not from data). The UI lists this object in the ⓘ panel.
const A = {
  haloS1: [120e3, 80e3],        // S1 halo around sub-L1, y and z amplitude, km
  haloMon: [250e3, 150e3],      // L1 reference monitor (DSCOVR-like) halo, km
  haloForm: [50e3, 30e3],       // S2–S5 formation centre halo, km (drifts a few 1,000 km per event)
  haloPeriod: 180 * DAY,
  obsV0: .10, obsHalf: 5, obsW: 40,                     // what the forecaster sees: 10 %, 5°, 40 km/s
  tiltSd: 25, tiltMax: 60,                              // shock-normal tilt prior, deg
  imgErr: .05, imgDt: 6 * H, imgMaxR: .7 * AU,          // Vigil-like L5 imager: 5 % range error, every 6 h
  insituErr: .03, timingErr: 2,                         // in-situ speed 3 %, contact timing ±2 s
  nEns: 200, essFrac: .5, lwA: .97,                     // particle filter: members, resample at ESS < 50 %, Liu–West shrinkage
  budget: 40, wDv: .004, uThr: .25, uHyst: .03,         // S5: Δv budget m/s, Δv weight, U limit, hysteresis
  noise: .075, L0: 3000, Lmin: 500, Lmax: 15000, nCand: 12,
  tFirst: .8 * H, tDec: 6 * H, lastBefore: 3 * H, arriveBefore: 2 * H, minTransfer: 1800,
  transit: [.8 * DAY, 5 * DAY], machMin: 1.2,
  wCustom: 400, gCustom: 2e-8,                          // the v10 defaults, used only for "Custom" events
};

// ---------------- small helpers ----------------
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const sub = (a, b) => a.map((x, i) => x - b[i]), add = (a, b) => a.map((x, i) => x + b[i]), mul = (a, s) => a.map(x => x * s);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = a => {const m = Math.hypot(...a); return a.map(x => x / m)};
const angDeg = (a, b) => Math.acos(clamp(dot(nrm(a), nrm(b)), -1, 1)) / D2R;
const km = n => Math.round(n).toLocaleString('en-US') + ' km';
const hh = t => (t >= 0 ? '+' : '−') + (Math.abs(t) / H).toFixed(1) + 'h';

function mulberry32(a) {return () => {a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296}}
// one independent stream per purpose, so changing one part never shifts another
function hash(a, b, c) {
  let h = (a | 0) ^ 0x9E3779B9; h = Math.imul(h ^ h >>> 16, 0x85EBCA6B);
  h ^= Math.imul((b | 0) + 0x632BE5AB | 0, 0xC2B2AE35); h = Math.imul(h ^ h >>> 13, 0x27D4EB2F);
  h ^= Math.imul((c | 0) + 0x165667B1 | 0, 0x9E3779B1); h ^= h >>> 15; return h >>> 0}
function stream(seed, salt, k = 0) {const rnd = mulberry32(hash(seed, salt, k)); rnd.g = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); return rnd}
const strSeed = s => {let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return h >>> 0};

// ---------------- Drag-Based Model, exact closed form ----------------
// dv/dt = −γ (v−w)|v−w|  →  r(t) = r0 + w t + (s/γ) ln(1 + s γ (v0−w) t),  s = sign(v0−w)
// r(t) is monotonic and concave (v0 > w) or convex (v0 < w): Newton from the lower bound
// t = (R−r0)/max(v0, w) converges monotonically, ~5 steps instead of 64 bisections.
function traj(v0, w, g, rE = AU, r0 = R0) {
  const dv = v0 - w, s = dv >= 0 ? 1 : -1, k = s * g * dv;
  const r = t => r0 + w * t + s / g * Math.log1p(k * t), v = t => dv / (1 + k * t) + w;
  const tAt = R => {
    if (R <= r0) return 0;
    let t = (R - r0) / Math.max(v0, w);
    for (let i = 0; i < 50; i++) {const d = (r(t) - R) / v(t); t -= d; if (Math.abs(d) < 1e-4) break}
    return t > 25 * DAY ? Infinity : t};
  return {v0, w, g, rE, r0, r, v, tAt, te: tAt(rE)}}

// ---------------- plasma ----------------
function cms(n, B, T) {const rho = n * 1e6 * MP, vA = B * 1e-9 / Math.sqrt(MU0 * rho), cs = Math.sqrt(GAD * 2 * KB * T / MP); return Math.hypot(vA, cs) / 1e3}
const CMS1 = cms(AMB.n, AMB.B, AMB.T);
function shockJump(Vsh, w, up = AMB) {           // gas-dynamic Rankine–Hugoniot, upstream = ambient wind
  const M = Math.max(1.01, (Vsh - w) / cms(up.n, up.B, up.T)), X = (GAD + 1) * M * M / ((GAD - 1) * M * M + 2);
  const p21 = (2 * GAD * M * M - (GAD - 1)) / (GAD + 1);
  return {M, X, v2: Vsh - (Vsh - w) / X, n2: up.n * X, T2: up.T * p21 / X}}
const machL1 = (T, rL1) => (T.v(T.tAt(rL1)) - T.w) / CMS1;

// ---------------- multi-spacecraft timing (least squares) ----------------
function det3(a) {return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])}
function solve3(A, b) {const D = det3(A); return [0, 1, 2].map(k => det3(A.map((row, i) => row.map((x, j) => j === k ? b[i] : x))) / D)}
function timingSolve(P, t) {
  const N = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
  for (let k = 1; k < P.length; k++) {const a = sub(P[k], P[0]), b = t[k] - t[0]; for (let i = 0; i < 3; i++) {r[i] += a[i] * b; for (let j = 0; j < 3; j++) N[i][j] += a[i] * a[j]}}
  const m = solve3(N, r), mm = Math.hypot(...m); return {n: m.map(x => x / mm), V: 1 / mm}}
function tetQ(ps) {const V = Math.abs(dot(sub(ps[1], ps[0]), cross(sub(ps[2], ps[0]), sub(ps[3], ps[0])))) / 6; let s = 0; for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {const d = sub(ps[i], ps[j]); s += dot(d, d)} const L = Math.sqrt(s / 6); return 6 * Math.SQRT2 * V / L ** 3}

// ---------------- scout error model (same as v10 and ssw/planner.py) ----------------
const GH = [...Array(41)].map((_, i) => -2.5 + i * .125), GW = GH.map(z => Math.exp(-z * z / 2)), GS = GW.reduce((a, b) => a + b);
const eps = (L, lam) => {const x = 2 * Math.PI * L / lam; return Math.min(1.5, Math.hypot(Math.SQRT2 * A.noise / x, x * x / 6))};
const EepsP = (L, lam, sig) => GH.reduce((s, z, i) => s + GW[i] * eps(L, lam * Math.exp(sig * z)), 0) / GS;
const CAND = [...Array(A.nCand)].map((_, i) => Math.round(A.Lmin * Math.pow(A.Lmax / A.Lmin, i / (A.nCand - 1))));

// ---------------- swarm geometry (km from L1; x Sun→Earth, z ecliptic north) ----------------
const SAT0 = [{id: 'S1', p: [-750e3, 0, 0], tag: 'sail hub'}, {id: 'S2', p: [-20e3, -240e3, -60e3], tag: ''}, {id: 'S3', p: [-20e3, 240e3, -60e3], tag: ''},
              {id: 'S4', p: [30e3, 0, 250e3], tag: ''}, {id: 'S5', p: [-20e3, -237e3, -60e3], tag: 'scout'}];
const OMH = 2 * Math.PI / A.haloPeriod;

// ---------------- event priors ----------------
const FALLBACK = {
  v0: {kind: 'lognormal', median: 800, sigma_ln: .40, clip: [450, 3000]}, half_angle: {kind: 'uniform', lo: 25, hi: 60},
  lon: {kind: 'normal', mean: 0, sd: 15}, w: {kind: 'normal', mean: 430, sd: 90, clip: [300, 750]},
  gamma: {kind: 'lognormal', median: 2e-8, sigma_ln: .7}, gamma_m1: {kind: 'value', value: 2e-8},
  bz_sheath: {kind: 'normal', mean: -6, sd: 7}, tau_c: {kind: 'lognormal', median: 15, sigma_ln: .5}};
function priors(P) {
  const o = {meta: (P && P.meta) || {data_mode: 'assumed'}, historical: (P && P.historical) || []};
  for (const k in FALLBACK) o[k] = P && P[k] ? P[k] : {...FALLBACK[k], source: 'assumed', n: 0};
  return o}
const clipTo = (x, c) => c ? clamp(x, c[0], c[1]) : x;
function draw(pr, rnd) {
  switch (pr.kind) {
    case 'lognormal': return clipTo(pr.median * Math.exp(pr.sigma_ln * rnd.g()), pr.clip);
    case 'normal': return clipTo(pr.mean + pr.sd * rnd.g(), pr.clip);
    case 'uniform': return pr.lo + (pr.hi - pr.lo) * rnd();
    case 'quantiles': {const u = rnd() * 100, p = pr.p, q = pr.q; let j = 0; while (j < p.length - 2 && u > p[j + 1]) j++; return q[j] + (q[j + 1] - q[j]) * (u - p[j]) / (p[j + 1] - p[j])}
    default: return pr.value}}
function cdfQ(pr, x) {const p = pr.p, q = pr.q; if (x <= q[0]) return 0; if (x >= q[q.length - 1]) return 1;
  let j = 0; while (q[j + 1] < x) j++; const w = q[j + 1] - q[j]; return (p[j] + (w > 0 ? (p[j + 1] - p[j]) * (x - q[j]) / w : 0)) / 100}
function pdf(pr, x) {                                   // prior density (up to a constant)
  if (pr.clip && (x < pr.clip[0] || x > pr.clip[1])) return 0;
  switch (pr.kind) {
    case 'lognormal': {if (x <= 0) return 0; const z = Math.log(x / pr.median) / pr.sigma_ln; return Math.exp(-z * z / 2) / x}
    case 'normal': {const z = (x - pr.mean) / pr.sd; return Math.exp(-z * z / 2)}
    case 'uniform': return x >= pr.lo && x <= pr.hi ? 1 : 0;
    case 'quantiles': {const h = (pr.q[pr.q.length - 1] - pr.q[0]) / 40; return (cdfQ(pr, x + h) - cdfQ(pr, x - h)) / (2 * h)}
    default: return 1}}
const lnSpread = pr => pr.kind === 'lognormal' ? pr.sigma_ln : .5;

// =====================================================================================
//  ONE EVENT
// =====================================================================================
// spec: {seed} random · {hist: cme_id, seed?} historical · {custom: {v0, tilt (deg), bz}, seed?}
function makeEvent(spec, Pin, opt = {}) {
  const P = priors(Pin), seed = spec.seed != null ? spec.seed >>> 0 : spec.hist ? strSeed(String(spec.hist)) : 1;
  const tr = spec.hist ? histTruth(spec.hist, P, seed) : spec.custom ? customTruth(spec.custom, P, seed) : drawTruth(seed, P);
  const E = core(tr, P, opt);
  forecast(E, P);
  plan(E, P);
  contacts(E);
  afterContact(E);
  E.summary = summarize(E);
  return E}

// ---------- 2a. hidden truth ----------
function drawHalo(rnd, t) {t.ph1 = rnd() * 2 * Math.PI; t.ph2 = rnd() * 2 * Math.PI; t.ph3 = rnd() * 2 * Math.PI}
function drawTruth(seed, P) {
  for (let k = 0; k < 500; k++) {                       // "draw again with the next seed" (sub-seed k)
    const rnd = stream(seed, 1, k), t = {kind: 'random', seed, attempt: k};
    t.v0 = draw(P.v0, rnd); t.w = draw(P.w, rnd); t.g = draw(P.gamma, rnd);
    t.half = draw(P.half_angle, rnd); t.lon = draw(P.lon, rnd);
    t.tilt = clamp(A.tiltSd * rnd.g(), -A.tiltMax, A.tiltMax) * D2R; t.phiN = rnd() * 2 * Math.PI;
    t.bz = draw(P.bz_sheath, rnd); t.tau = draw(P.tau_c, rnd); drawHalo(rnd, t); t.rE = AU;
    if (accept(t)) return t}
  throw new Error('no Earth-impacting event found for seed ' + seed)}
function accept(t, why) {
  if (!(Math.abs(t.lon) < t.half)) return why ? 'misses Earth (|lon| ≥ half-angle)' : false;
  const T = traj(t.v0, t.w, t.g, t.rE);
  if (!(T.te >= A.transit[0] && T.te <= A.transit[1])) return why ? 'transit outside 0.8–5 days' : false;
  if (!(machL1(T, t.rE - L1OFF) > A.machMin)) return why ? 'no shock at L1 (Mach ≤ 1.2)' : false;
  return why ? '' : true}
function histTruth(id, P, seed) {
  const h = P.historical.find(x => x.cme_id === id);
  if (!h) throw new Error('historical event not found: ' + id);
  const rnd = stream(seed, 8), t = {kind: 'historical', seed, h};
  t.v0 = h.v0; t.w = h.w_obs; t.half = h.half_angle; t.lon = h.lon;
  t.tilt = clamp(A.tiltSd * rnd.g(), -A.tiltMax, A.tiltMax) * D2R; t.phiN = rnd() * 2 * Math.PI;
  t.bz = h.bz_min_sheath != null ? h.bz_min_sheath : draw(P.bz_sheath, rnd); t.tau = draw(P.tau_c, rnd); drawHalo(rnd, t);
  t.rE = h.r_target_km ? h.r_target_km + L1OFF : AU; t.t21 = Date.parse(h.t21); t.tObs = Date.parse(h.t_obs);
  t.tEnlil = h.enlil_arr ? (Date.parse(h.enlil_arr) - t.t21) / 1e3 : null;
  t.g = solveGamma(t, (t.tObs - t.t21) / 1e3);           // γ chosen so the L1 monitor sees the shock at the real time
  return t}
function customTruth(c, P, seed) {
  const rnd = stream(seed, 8), t = {kind: 'custom', seed};
  t.v0 = c.v0; t.w = A.wCustom; t.g = A.gCustom; t.half = 40; t.lon = 0;
  t.tilt = (c.tilt || 0) * D2R; t.phiN = 20 * D2R; t.bz = c.bz; t.tau = P.tau_c.median || draw(P.tau_c, rnd); drawHalo(rnd, t); t.rE = AU;
  return t}
// monitor contact time for a given γ (positions are known; the tilt is the event's own)
function monitorContact(t, g) {const T = traj(t.v0, t.w, g, t.rE); return contactT(T, t.rE - L1OFF, normal(t), tt => monPos(t, tt))}
function solveGamma(t, tTarget) {
  let lo = Math.log(1e-11), hi = Math.log(1e-5);
  const f = lg => monitorContact(t, Math.exp(lg)) - tTarget, flo = f(lo), fhi = f(hi);
  if (!(flo * fhi < 0)) {t.gammaFit = false; return Math.exp(Math.abs(flo) < Math.abs(fhi) ? lo : hi)}
  for (let i = 0; i < 80; i++) {const m = (lo + hi) / 2; if ((f(m) > 0) === (flo > 0)) lo = m; else hi = m}
  t.gammaFit = true; return Math.exp((lo + hi) / 2)}

// ---------- 2b. positions on their halo orbits ----------
const normal = t => [Math.cos(t.tilt), Math.sin(t.tilt) * Math.cos(t.phiN), Math.sin(t.tilt) * Math.sin(t.phiN)];
const s1Pos = (t, tt) => [-750e3, A.haloS1[0] * Math.sin(t.ph1 + OMH * tt), A.haloS1[1] * Math.cos(t.ph1 + OMH * tt)];
const monPos = (t, tt) => [0, A.haloMon[0] * Math.sin(t.ph2 + OMH * tt), A.haloMon[1] * Math.cos(t.ph2 + OMH * tt)];
const formDrift = (t, tt) => [0, A.haloForm[0] * (Math.sin(t.ph3 + OMH * tt) - Math.sin(t.ph3)), A.haloForm[1] * (Math.cos(t.ph3 + OMH * tt) - Math.cos(t.ph3))];
// a planar shock with normal n, nose on the Sun–Earth line at r(t), reaches point p when r(t) = rL1 + n·p / n_x
function contactT(T, rL1, n, posAt) {let tt = T.tAt(rL1 + dot(n, posAt(0)) / n[0]); for (let i = 0; i < 3; i++) tt = T.tAt(rL1 + dot(n, posAt(tt)) / n[0]); return tt}

function core(t, P, opt) {
  const T = traj(t.v0, t.w, t.g, t.rE), rL1 = t.rE - L1OFF, n = normal(t);
  const E = {truth: t, T, rE: t.rE, rL1, n, P, A, opt};
  E.o = {lon: t.lon * D2R, half: t.half * D2R, tilt: t.tilt, phiN: t.phiN, bz: t.bz, T0: -(R0 - RSUN) / t.v0 - 3600, live: false, t21: t.t21 || null};
  E.tOn = -(T.r0 - RSUN) / T.v0; E.tE = T.te; E.tEnter = T.tAt(rL1 - SUBOFF - 1.5e6); E.tEnd = E.tE + 900;
  E.onAxis = !!opt.onAxis;
  E.s1At = tt => E.onAxis ? [-750e3, 0, 0] : s1Pos(t, tt);
  E.monAt = tt => E.onAxis ? [0, 0, 0] : monPos(t, tt);
  E.driftAt = tt => formDrift(t, tt);
  E.lamTrue = null;                                     // set after contact (needs the true shock speed)
  // 2c. what the forecaster sees (historical: the real DONKI/L1 numbers ARE the observation)
  const rnd = stream(t.seed, 2);
  E.obs = t.kind === 'historical' ? {v0: t.v0, half: t.half, w: t.w}
    : {v0: t.v0 * (1 + A.obsV0 * rnd.g()), half: t.half + A.obsHalf * rnd.g(), w: t.w + A.obsW * rnd.g()};
  E.why = accept(t, true);                              // '' for every random event; may flag historical/custom ones
  E.log = [];
  return E}

// ---------- 2d. forecast: 200-member particle filter ----------
function forecast(E, P) {
  const N = A.nEns, ob = E.obs, rnd = stream(E.truth.seed, 3), rE = E.rE, rL1 = E.rL1;
  const lnV = Math.log(ob.v0), sv = A.obsV0;
  const M = {lv: new Float64Array(N), w: new Float64Array(N), lg: new Float64Array(N), th: new Float64Array(N), ph: new Float64Array(N), wt: new Float64Array(N)};
  const ok = (v, w, g) => {const T = traj(v, w, g, rE); return T.te >= A.transit[0] && T.te <= A.transit[1] && machL1(T, rL1) > A.machMin};
  // importance sampling: propose v0 ~ LogN(v0_obs, 10 %), w ~ N(w_obs, 40), γ ~ prior; weight = prior(v0)·L(v0_obs|v0)/q(v0) · prior(w)
  for (let i = 0; i < N; i++) {
    let v, w, g, k = 0;
    do {v = Math.exp(lnV + sv * rnd.g()); w = ob.w + A.obsW * rnd.g(); g = draw(P.gamma, rnd)} while (!ok(v, w, g) && ++k < 60);
    const zl = (ob.v0 - v) / (sv * v), zq = (Math.log(v) - lnV) / sv;
    M.lv[i] = Math.log(v); M.w[i] = w; M.lg[i] = Math.log(g);
    M.th[i] = clamp(A.tiltSd * rnd.g(), -A.tiltMax, A.tiltMax) * D2R; M.ph[i] = rnd() * 2 * Math.PI;
    M.wt[i] = k < 60 ? pdf(P.v0, v) * Math.exp(-zl * zl / 2) / Math.exp(-zq * zq / 2) * pdf(P.w, w) : 0;
  }
  // an observation far outside the prior can zero every weight: then trust the observation alone
  E.priorIgnored = !normW(M.wt); if (E.priorIgnored) M.wt.fill(1 / N);
  const derive = () => {                                // per-member predictions with the known halo positions
    M.T = []; M.te = new Float64Array(N); M.vL1 = new Float64Array(N); M.gain = new Float64Array(N); M.tS1 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const T = traj(Math.exp(M.lv[i]), M.w[i], Math.exp(M.lg[i]), rE), n = [Math.cos(M.th[i]), Math.sin(M.th[i]) * Math.cos(M.ph[i]), Math.sin(M.th[i]) * Math.sin(M.ph[i])];
      M.T.push(T); M.te[i] = T.te; M.vL1[i] = T.v(T.tAt(rL1));
      const a = contactT(T, rL1, n, E.s1At), b = contactT(T, rL1, n, E.monAt);
      M.tS1[i] = a; M.gain[i] = (b - a) / 60}};
  const stats = (t, kind, extra) => {
    const s = x => wstat(x, M.wt), e = s(M.te), v = s(M.vL1), gn = s(M.gain), s1 = s(M.tS1);
    const h = {t, kind, eta_m: e.m, eta_s: e.s, vL1_m: v.m, vL1_s: v.s, gain_m: gn.m, gain_s: gn.s, tS1_m: s1.m, tS1_s: s1.s,
      g_m: Math.exp(s(M.lg).m), w_m: s(M.w).m, ess: ess(M.wt), ...extra};
    E.hist.push(h); return h};
  E.hist = []; E.M = M;
  derive();
  E.ens0 = M.T.slice(0, 20);                            // a sample of the first-guess members, for the speed timeline
  const h0 = stats(.3 * H, 'initial', {});
  E.log.push([.3 * H, `① Forecast (200-member ensemble, observed v0 ${ob.v0.toFixed(0)} km/s, w ${ob.w.toFixed(0)} km/s): ETA ${(h0.eta_m / H).toFixed(1)} h ± ${(h0.eta_s / H).toFixed(1)} h · warning gain expected +${h0.gain_m.toFixed(0)} ± ${h0.gain_s.toFixed(0)} min`, 'hi']);
  // imager updates every 6 h until the front reaches 0.7 AU
  const ri = stream(E.truth.seed, 4), rj = stream(E.truth.seed, 5), T = E.T;
  for (let tk = A.imgDt; T.r(tk) < A.imgMaxR && tk < T.tAt(rL1 - SUBOFF) - H; tk += A.imgDt) {
    const rt = T.r(tk), ro = rt * (1 + A.imgErr * ri.g());
    for (let i = 0; i < N; i++) {const r = M.T[i].r(tk), sg = A.imgErr * r, z = (ro - r) / sg; M.wt[i] *= Math.exp(-z * z / 2) / sg}
    if (!normW(M.wt)) M.wt.fill(1 / N);
    const es = ess(M.wt), res = es < A.essFrac * N;
    if (res) {resample(M, rj, ok); derive()}
    const h = stats(tk, 'imager', {r_obs: ro, ess_before: es, resampled: res});
    E.log.push([tk, `L5 imager (assumed, Vigil-like): front at ${(ro / AU).toFixed(3)} AU ± 5 % → ETA ${(h.eta_m / H).toFixed(1)} h ± ${(h.eta_s / H).toFixed(1)} h, V̂(L1) ${h.vL1_m.toFixed(0)} ± ${h.vL1_s.toFixed(0)} km/s${res ? ' · resampled (ESS ' + (100 * es / N).toFixed(0) + ' %)' : ''}`, '']);
  }
}
function normW(w) {let s = 0; for (const x of w) s += x; if (!(s > 0) || !isFinite(s)) return false; for (let i = 0; i < w.length; i++) w[i] /= s; return true}
function ess(w) {let s = 0; for (const x of w) s += x * x; return 1 / s}
function wstat(x, w) {let m = 0; for (let i = 0; i < x.length; i++) m += w[i] * x[i]; let v = 0; for (let i = 0; i < x.length; i++) v += w[i] * (x[i] - m) ** 2; return {m, s: Math.sqrt(v)}}
// systematic resampling + Liu–West shrinkage jitter on (ln v0, w, ln γ)
function resample(M, rnd, ok) {
  const N = M.wt.length, a = A.lwA, hs = Math.sqrt(1 - a * a), keys = ['lv', 'w', 'lg'];
  const mean = {}, sd = {}; for (const k of keys) {const s = wstat(M[k], M.wt); mean[k] = s.m; sd[k] = s.s}
  const idx = new Int32Array(N); let c = M.wt[0], j = 0; const u0 = rnd() / N;
  for (let i = 0; i < N; i++) {const u = u0 + i / N; while (u > c && j < N - 1) c += M.wt[++j]; idx[i] = j}
  const old = {}; for (const k of ['lv', 'w', 'lg', 'th', 'ph']) old[k] = Float64Array.from(M[k]);
  for (let i = 0; i < N; i++) {
    const p = idx[i]; let v, w, g, k = 0;
    do {v = a * old.lv[p] + (1 - a) * mean.lv + hs * sd.lv * rnd.g(); w = a * old.w[p] + (1 - a) * mean.w + hs * sd.w * rnd.g(); g = a * old.lg[p] + (1 - a) * mean.lg + hs * sd.lg * rnd.g()}
    while (!ok(Math.exp(v), w, Math.exp(g)) && ++k < 20);
    if (k >= 20) {v = old.lv[p]; w = old.w[p]; g = old.lg[p]}
    M.lv[i] = v; M.w[i] = w; M.lg[i] = g; M.th[i] = old.th[p]; M.ph[i] = old.ph[p]}
  M.wt.fill(1 / N)}

// ---------- 3. S5 re-planning from the forecast's uncertainty ----------
// Track = piecewise-constant acceleration segments {t0, t1, L0 (km), v0 (km/s), a (km/s²)}.
function trackAt(seg, t) {
  let s = seg[0];
  for (const x of seg) {if (t < x.t0) break; s = x}
  const d = clamp(t, s.t0, s.t1) - s.t0, L = s.L0 + s.v0 * d + .5 * s.a * d * d, v = s.v0 + s.a * d;
  return t > s.t1 ? {L: L + v * (t - s.t1), v} : {L, v}}
function dvUsed(seg, t) {let dv = 0; for (const s of seg) {if (t <= s.t0) break; dv += Math.abs(s.a) * (Math.min(t, s.t1) - s.t0)} return dv * 1e3}
function histAt(E, t) {let h = E.hist[0]; for (const x of E.hist) if (x.t <= t) h = x; return h}
function plan(E, P) {
  const tau = P.tau_c, sTau = lnSpread(tau), L0 = A.L0;
  E.seg = [{t0: -1e9, t1: 1e9, L0, v0: 0, a: 0}]; E.dec = []; E.target = L0;
  const decide = td => {
    const h = histAt(E, td), V = h.vL1_m, sV = h.vL1_s, lam = V * tau.median, sig = Math.hypot(sTau, sV / V);
    const tArr = h.tS1_m - 2 * h.tS1_s - A.arriveBefore, st = trackAt(E.seg, td), used = dvUsed(E.seg, td), left = A.budget - used;
    // committed Δv of the plan already running = what is still to be burned on it
    const commit = dvUsed(E.seg, 1e9) - used, Uh = EepsP(E.target, lam, sig), Jh = Uh + A.wDv * commit;
    const aNow = (E.seg.find(s => td >= s.t0 && td < s.t1) || {a: 0}).a;
    const tStop = st.v !== 0 && aNow !== 0 ? Math.abs(st.v / aNow) : 0, Lstop = st.L + st.v * tStop / 2, T2 = tArr - td - tStop;
    const cands = CAND.map(L => {const dv = Math.abs(st.v) * 1e3 + 4 * Math.abs(L - Lstop) * 1e3 / T2, U = EepsP(L, lam, sig);
      return {L, U, dv, J: U + A.wDv * dv, ok: T2 > A.minTransfer && dv <= left}});
    const D = {t: td, V, sV, lam, sig, Uh, Jh, target: E.target, left, cands, Lnow: st.L, vnow: st.v, tArr};
    const best = cands.filter(c => c.ok).reduce((a, b) => !a || b.J < a.J ? b : a, null);
    const pre = `${hh(td)}: V̂(L1) ${V.toFixed(0)} ± ${sV.toFixed(0)} km/s → λ ≈ ${km(lam)} (×/÷${Math.exp(sig).toFixed(2)}); U(${km(E.target).replace(' km', '')}) ${Uh.toFixed(2)}`;
    if (Uh <= A.uThr) {D.action = 'hold'; D.why = `${pre} ≤ ${A.uThr} → hold`}
    else if (!best) {D.action = 'hold'; D.why = `${pre} > ${A.uThr} but no separation is reachable within the budget/time → hold`}
    else if (best.L === E.target) {D.action = 'hold'; D.why = `${pre} > ${A.uThr}; already heading to the best affordable separation ${km(best.L)} → hold`}
    else if (Uh - best.U <= A.uHyst || !(best.J < Jh)) {
      D.action = 'hold'; D.why = `${pre} > ${A.uThr}; best ${km(best.L)} gains ${(Uh - best.U).toFixed(2)}${Uh - best.U <= A.uHyst ? ' < ' + A.uHyst : ', but J ' + best.J.toFixed(2) + ' ≥ ' + Jh.toFixed(2)} → hold`}
    else {
      D.action = 'retarget'; D.pick = best;
      // stop at the current acceleration (speed stays continuous), then bang-bang to the target, arriving at tArr
      const keep = E.seg.filter(s => s.t0 < td).map(s => ({...s}));
      const last = keep[keep.length - 1]; last.t1 = td;
      const segs = keep; let t0 = td, L = st.L, v = st.v;
      if (tStop > 0) {const a = -Math.sign(v) * Math.abs(aNow); segs.push({t0, t1: t0 + tStop, L0: L, v0: v, a}); t0 += tStop; L = Lstop; v = 0}
      const dL = best.L - L, aT = 4 * dL / (T2 * T2), tm = t0 + T2 / 2;
      segs.push({t0, t1: tm, L0: L, v0: 0, a: aT}); segs.push({t0: tm, t1: tArr, L0: L + dL / 2, v0: aT * T2 / 2, a: -aT});
      segs.push({t0: tArr, t1: 1e9, L0: best.L, v0: 0, a: 0});
      E.seg = segs; E.target = best.L;
      D.why = `${pre} > ${A.uThr} → retarget ${km(best.L)}, Δv ${best.dv.toFixed(1)} m/s, ${(left - best.dv).toFixed(1)} m/s left (arrive ${hh(tArr)})`}
    E.dec.push(D);
    E.log.push([td, '② ' + D.why, D.action === 'retarget' ? 'alert' : 'hi'])};
  // decision times: T+0.8 h, each imager update, then every 6 h; the last one at t̂_S1 − 3 h
  const times = [A.tFirst, ...E.hist.filter(h => h.kind === 'imager').map(h => h.t)];
  for (let t = times[times.length - 1] + A.tDec; ; t += A.tDec) {const h = histAt(E, t); if (t >= h.tS1_m - A.lastBefore) break; times.push(t)}
  const hl = histAt(E, times[times.length - 1]), tLast = hl.tS1_m - A.lastBefore;
  if (tLast > times[times.length - 1] + 600) times.push(tLast);
  for (const td of times) if (td < E.T.tAt(E.rL1 - SUBOFF)) decide(td);
  E.burns = E.dec.filter(d => d.action === 'retarget').length;
  E.burnTimes = E.seg.filter(s => s.a !== 0).map(s => s.t0);
  E.s5At = t => trackAt(E.seg, t); E.dvAt = t => dvUsed(E.seg, t)}

// ---------- contacts with the tilted planar shock ----------
function contacts(E) {
  const t = E.truth, T = E.T, n = E.n, rL1 = E.rL1;
  const pos = [E.s1At, ...[1, 2, 3].map(i => tt => add(SAT0[i].p, E.driftAt(tt))), tt => add(add(SAT0[1].p, E.driftAt(tt)), [0, E.s5At(tt).L, 0])];
  E.satAt = (i, tt) => pos[i](tt);
  E.tc = pos.map(f => contactT(T, rL1, n, f));
  E.tMon = contactT(T, rL1, n, E.monAt);
  E.tSub = E.tc[0]; E.tL1 = E.tMon; E.gain = (E.tMon - E.tSub) / 60;
  E.tL1axis = T.tAt(rL1); E.tSubAxis = T.tAt(rL1 - SUBOFF);
  E.sats = SAT0.map((s, i) => ({...s, p: pos[i](E.tc[i])}));             // positions at their own contact
  E.monP = E.monAt(E.tMon); E.drift = Math.hypot(...E.driftAt(E.tc[1]));
  const rnd = stream(t.seed, 6), tn = E.tc.map(x => x + A.timingErr * rnd.g());
  E.sol4 = timingSolve(E.sats.slice(0, 4).map(s => s.p), tn.slice(0, 4)); E.sol5 = timingSolve(E.sats.map(s => s.p), tn);
  E.tSolve = Math.max(...E.tc.slice(0, 4)) + 60; E.Vn = T.v((E.tc[0] + Math.max(...E.tc.slice(1, 4))) / 2) * n[0];
  E.err4 = angDeg(E.sol4.n, n); E.err5 = angDeg(E.sol5.n, n);
  E.J = shockJump(T.v(E.tE), T.w); E.psi = Math.atan(OMEGA * rL1 / T.w);
  E.movingAtContact = Math.abs(E.s5At(E.tc[4]).v) > 1e-9}

// ---------- in-situ update after S1 contact, and the honest scorecard ----------
function afterContact(E) {
  const T = E.T, t = E.truth, rnd = stream(t.seed, 7), hp = histAt(E, E.tSub - 1);
  const Vm = T.v(E.tSub) * (1 + A.insituErr * rnd.g()), rS1 = E.rL1 - SUBOFF;
  const a = traj(Vm, hp.w_m, hp.g_m, E.rE, rS1), b = traj(Vm * (1 + A.insituErr), hp.w_m, hp.g_m, E.rE, rS1);
  const h = {t: E.tSub, kind: 'insitu', eta_m: E.tSub + a.te, eta_s: Math.abs(b.te - a.te), vL1_m: Vm, vL1_s: A.insituErr * Vm,
    gain_m: E.gain, gain_s: 0, tS1_m: E.tSub, tS1_s: 0, g_m: hp.g_m, w_m: hp.w_m, ess: hp.ess, v_meas: Vm};
  E.hist.push(h);
  E.lamTrue = T.v(E.tL1axis) * t.tau;
  E.errL = eps(E.s5At(E.tc[4]).L, E.lamTrue); E.errL0 = eps(A.L0, E.lamTrue);
  const s = E.sats;
  E.log.push([E.tSub, `③ S1 contact — shock ${T.v(E.tSub).toFixed(0)} km/s measured ${Vm.toFixed(0)} ± 3 % → ETA σ ${(h.eta_s / 60).toFixed(0)} min`, 'alert']);
  for (let i = 1; i < 5; i++) E.log.push([E.tc[i], `${s[i].id} contact ${mm(E.tc[i] - E.tc[0])} after S1`, '']);
  E.log.push([E.tSolve, `Timing solution (S1–S4): normal within ${E.err4.toFixed(2)}° of truth, V = ${E.sol4.V.toFixed(0)} km/s (true ${E.Vn.toFixed(0)})`, 'hi']);
  E.log.push([E.tMon, `L1 reference monitor (on its halo) sees the shock — swarm gained +${E.gain.toFixed(1)} min`, 'alert']);
  E.log.push([E.tE, `Shock reaches Earth — forecast error ${fmtErr(h.eta_m - E.tE)} after S1 contact, ${fmtErr(E.hist[0].eta_m - E.tE)} at T+0.3 h`, 'alert']);
  E.log.sort((x, y) => x[0] - y[0])}
const mm = dt => {const a = Math.abs(dt); return (dt < 0 ? '−' : '+') + String(Math.floor(a / 60)).padStart(2, '0') + ':' + String(Math.floor(a % 60)).padStart(2, '0')};
const fmtErr = d => Math.abs(d) >= H ? (d >= 0 ? '+' : '−') + (Math.abs(d) / H).toFixed(1) + ' h' : (d >= 0 ? '+' : '−') + (Math.abs(d) / 60).toFixed(1) + ' min';

// one row of numbers per event: what the Monte Carlo and the tests look at
function summarize(E) {
  const tE = E.tE, pre = E.hist.filter(h => h.kind !== 'insitu'), at = t => {const h = histAt(E, t); return h && h.t <= t && h.kind !== 'insitu' && t < E.tSub ? h : null};
  const err = h => h ? Math.abs(h.eta_m - tE) : NaN, z = h => h ? (h.eta_m - tE) / h.eta_s : NaN;
  const post = E.hist[E.hist.length - 1], last = pre[pre.length - 1];
  return {seed: E.truth.seed, kind: E.truth.kind, v0: E.truth.v0, w: E.truth.w, g: E.truth.g, tilt: E.truth.tilt / D2R, transit_h: tE / H,
    gain: E.gain, gainAxis: (E.tL1axis - E.tSubAxis) / 60, vL1: E.T.v(E.tL1axis), gainExp: E.hist[0].gain_m, gainExpS: E.hist[0].gain_s,
    err0: err(E.hist[0]), err12: err(at(12 * H)), err24: err(at(24 * H)), errLast: err(last), errPost: Math.abs(post.eta_m - tE),
    z0: z(E.hist[0]), zLast: z(last), zs: pre.map(z), sig: pre.map(h => h.eta_s), kinds: pre.map(h => h.kind),
    nImg: pre.length - 1, dv: E.dvAt(1e9), burns: E.burns, moved: E.burns > 0, Lfinal: E.s5At(E.tc[4]).L,
    lamTrue: E.lamTrue, errL: E.errL, errL0: E.errL0, movingAtContact: E.movingAtContact}}

function batch(n, P, seed0 = 1, opt = {}) {const out = []; for (let i = 0; i < n; i++) out.push(makeEvent({seed: seed0 + i}, P, opt).summary); return out}

return {D2R, H, DAY, AU, RSUN, R0, L1OFF, SUBOFF, OMEGA, MP, KB, MU0, GAD, AMB, A, SAT0, CAND, FALLBACK,
  clamp, sub, add, mul, dot, cross, nrm, angDeg, mulberry32, hash, stream, strSeed,
  traj, cms, shockJump, det3, solve3, timingSolve, tetQ, eps, EepsP, priors, draw, pdf, accept,
  makeEvent, trackAt, dvUsed, histAt, contactT, summarize, batch}
}
const SSP = SSPhysFactory(); SSP.factory = SSPhysFactory;
if (typeof module === 'object' && module.exports) module.exports = SSP; else root.SSP = SSP;
})(typeof self !== 'undefined' ? self : this);
