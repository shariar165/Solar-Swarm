// Phase 6 checks on the v11 physics part.  node tests/js/v11_checks.js [nEvents]
// Prints one JSON object; exit code 1 if any check fails. Wrapped by tests/test_v11.py.
const {SSP, loadPriors} = require('./load.js');
const P = loadPriors() || loadPriors('event_priors_mock.js');
const N = +(process.argv[2] || 1000), out = {checks: {}, numbers: {}};
const ok = (name, pass, info) => {out.checks[name] = {pass: !!pass, ...info}};
const H = 3600;

// 1. tilt 0, both craft on the axis: gain ≈ 750,000 km / V_shock(L1) within 3 %
{
  let worst = 0;
  for (const v0 of [500, 800, 1180, 1800, 2600]) {
    const E = SSP.makeEvent({custom: {v0, tilt: 0, bz: -10}, seed: 3}, P, {onAxis: true});
    const ref = 750e3 / E.T.v(E.tL1axis) / 60;
    worst = Math.max(worst, Math.abs(E.gain / ref - 1));
  }
  ok('axis_gain_within_3pct', worst < .03, {worst_rel_err: +worst.toFixed(4)});
}

// 2. same seed → identical event (gain, contacts, decisions)
{
  const sig = E => JSON.stringify([E.gain, E.tc, E.tMon, E.dec.map(d => [d.t, d.action, d.why]), E.hist.map(h => [h.t, h.eta_m, h.eta_s])]);
  let same = true;
  for (const s of [1, 7, 42, 1234, 99999]) same = same && sig(SSP.makeEvent({seed: s}, P)) === sig(SSP.makeEvent({seed: s}, P));
  const differ = sig(SSP.makeEvent({seed: 1}, P)) !== sig(SSP.makeEvent({seed: 2}, P));
  ok('same_seed_same_event', same && differ, {});
}

// 3–5. batch: filter behaviour, coverage, Δv budget, continuity, timing
const t0 = Date.now(), B = [], EV = [];
for (let i = 0; i < N; i++) {const E = SSP.makeEvent({seed: 1 + i}, P); B.push(E.summary); if (i < 200) EV.push(E)}
const ms = Date.now() - t0;
ok('batch_1000_under_10s', ms * 1000 / N < 10000, {ms_per_1000: Math.round(ms * 1000 / N)});
{
  // σ_ETA after an imager update vs the forecast just before it, averaged over all updates
  let up = 0, n = 0, grew = 0;
  for (const s of B) for (let k = 1; k < s.sig.length; k++) {const r = s.sig[k] / s.sig[k - 1]; up += Math.log(r); n++; if (r > 1) grew++}
  ok('sigma_eta_does_not_grow_on_average', up / n <= 0, {mean_log_ratio: +(up / n).toFixed(4), updates: n, frac_updates_that_grew: +(grew / n).toFixed(3)});
  const cov = arr => arr.filter(z => Math.abs(z) <= 2).length / arr.filter(Number.isFinite).length;
  const all = B.flatMap(s => s.zs).filter(Number.isFinite);
  const c0 = cov(B.map(s => s.z0)), cl = cov(B.map(s => s.zLast)), ca = cov(all);
  out.numbers.coverage_2sigma = {initial: +c0.toFixed(3), last_before_contact: +cl.toFixed(3), all_updates: +ca.toFixed(3), n_events: N};
  ok('coverage_2sigma_about_95pct', ca > .92 && ca < .985 && cl > .90, out.numbers.coverage_2sigma);
}
{
  let maxDv = 0, jumpL = 0, jumpV = 0;
  for (const E of EV) {
    maxDv = Math.max(maxDv, E.dvAt(1e9));
    // sample L(t), v(t) at 1 s around every segment boundary and every 60 s elsewhere
    const ts = new Set();
    for (const s of E.seg) for (const b of [s.t0, s.t1]) if (Math.abs(b) < 1e8) for (let d = -3; d <= 3; d++) ts.add(b + d);
    for (let t = 0; t < E.tc[4]; t += 60) ts.add(t);
    const T = [...ts].sort((a, b) => a - b);
    for (let i = 1; i < T.length; i++) {
      const a = E.s5At(T[i - 1]), b = E.s5At(T[i]), dt = T[i] - T[i - 1];
      const amax = Math.max(...E.seg.map(s => Math.abs(s.a)));
      jumpL = Math.max(jumpL, Math.abs(b.L - a.L) - (Math.max(Math.abs(a.v), Math.abs(b.v)) * dt + amax * dt * dt));
      jumpV = Math.max(jumpV, Math.abs(b.v - a.v) - amax * dt * 1.0001);
    }
  }
  ok('dv_budget_never_exceeded', maxDv <= SSP.A.budget + 1e-9, {max_dv_ms: +maxDv.toFixed(2)});
  ok('s5_position_velocity_continuous', jumpL < 1e-6 && jumpV < 1e-9, {max_excess_jump_L_km: jumpL, max_excess_jump_v_kms: jumpV});
}

// 6. historical: γ is solved so the L1 monitor sees the shock at the real time (within 1 min)
{
  const H0 = (P.historical || []);
  let worst = 0, nfit = 0;
  for (const h of H0) {const E = SSP.makeEvent({hist: h.cme_id}, P); if (!E.truth.gammaFit) continue; nfit++; worst = Math.max(worst, Math.abs(E.tMon - (Date.parse(h.t_obs) - Date.parse(h.t21)) / 1e3))}
  ok('historical_arrival_within_1min', H0.length === 0 || (nfit > 0 && worst < 60), {n_historical: H0.length, n_gamma_solved: nfit, worst_err_s: +worst.toFixed(2)});
}

// headline numbers
const q = (a, p) => {a = a.filter(Number.isFinite).sort((x, y) => x - y); const i = (a.length - 1) * p, lo = Math.floor(i); return a.length ? a[lo] + (a[Math.min(lo + 1, a.length - 1)] - a[lo]) * (i - lo) : NaN};
const st = a => ({median: q(a, .5), p10: q(a, .1), p90: q(a, .9)});
out.numbers.gain_min = st(B.map(s => s.gain));
out.numbers.eta_err_h = {T0: st(B.map(s => s.err0 / H)), T12: st(B.map(s => s.err12 / H)), T24: st(B.map(s => s.err24 / H)), last_before_contact: st(B.map(s => s.errLast / H)), after_S1_min: st(B.map(s => s.errPost / 60))};
out.numbers.s5 = {moved_pct: 100 * B.filter(s => s.moved).length / N, mean_dv: B.reduce((a, s) => a + s.dv, 0) / N, burns: st(B.map(s => s.burns)),
  final_L: B.reduce((o, s) => {const k = Math.round(s.Lfinal); o[k] = (o[k] || 0) + 1; return o}, {}), moving_at_contact: B.filter(s => s.movingAtContact).length};
out.priors = P && P.meta ? P.meta.data_mode : 'assumed';
out.pass = Object.values(out.checks).every(c => c.pass);
console.log(JSON.stringify(out, null, 1));
process.exitCode = out.pass ? 0 : 1;
