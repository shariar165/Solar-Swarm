// Fix 3: S5 home (pre-event) separation.   node tests/js/s5_home.js [n=1000]  ->  results/s5_home.json
// 1. the home separation from the priors (SSP.homeSep: minimum of the expected U over the population as the forecaster
//    sees it before an event), with the band where U stays within +0.02 of the minimum;
// 2. two cross-checks on the same n Monte Carlo events: the optimum if the forecaster's FIRST forecast of each event
//    were known in advance, and the optimum from the true λ of each event (neither is used by the model);
// 3. the between-storm Δv: the one-time move from the v10 start (3,000 km) to home, and the return to home after an
//    event, against the real gaps between CME-driven shocks at L1 (results/events.csv);
// 4. sensitivity of the home to the magnetometer noise ratio and to τ_c;
// 5. the rolling home: the optimum from the measured λ of the last 20 real shocks, every month (results/home_rolling.json);
// 6. the event-time re-planner from home: why it held at every decision time.
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');
global.window = {}; require(path.join(ROOT, 'ui', 'event_priors.js'));
const SSP = require(path.join(ROOT, 'ui', 'ss-physics-v11.js')), A = SSP.A, P = SSP.priors(window.SS_PRIORS), DAY = 86400;
const n = +process.argv[2] || 1000;
const q = (a, p) => {a = a.filter(Number.isFinite).sort((x, y) => x - y); const i = (a.length - 1) * p, lo = Math.floor(i); return a[lo] + (a[Math.min(lo + 1, a.length - 1)] - a[lo]) * (i - lo)};
const r3 = x => Math.round(x * 1000) / 1000, st = a => ({median: r3(q(a, .5)), p10: r3(q(a, .1)), p90: r3(q(a, .9))});

const home = SSP.homeSep(P);
// the same 1 % grid as homeSep, for the cross-checks
const grid = []; for (let L = A.Lmin; L <= A.Lmax * 1.000001; L *= 1.01) grid.push(L);
const optimum = U => {const u = grid.map(U), m = Math.min(...u), k = u.indexOf(m), f = grid.filter((_, i) => u[i] <= m + A.homeFlat);
  return {km: Math.round(grid[k]), U: r3(m), flat_km: [Math.round(f[0]), Math.round(f[f.length - 1])]}};
const E = []; for (let i = 1; i <= n; i++) E.push(SSP.makeEvent({seed: i}, window.SS_PRIORS));
const ff = E.filter(e => e.dec.length).map(e => [e.dec[0].lam, e.dec[0].sig]);
const firstForecast = optimum(L => ff.reduce((s, [l, sg]) => s + SSP.EepsP(L, l, sg), 0) / ff.length);
const truth = optimum(L => E.reduce((s, e) => s + SSP.eps(L, e.lamTrue), 0) / E.length);

// between storms: gaps between consecutive CME-driven shocks at L1 in the real catalogue
const csv = fs.readFileSync(path.join(ROOT, 'results', 'events.csv'), 'utf8').trim().split(/\r?\n/), hd = csv[0].split(',');
const it = hd.indexOf('t_obs'), tt = [...new Set(csv.slice(1).map(l => Date.parse(l.split(',')[it].replace(' ', 'T'))))].filter(Number.isFinite).sort((a, b) => a - b);
const gaps = tt.slice(1).map((t, i) => (t - tt[i]) / 1e3 / DAY).filter(g => g > 0);
// Δv for a transfer of dL km in T s: two impulses and a coast 2·dL/T (minimum); the planner's bang-bang profile 4·dL/T
const dv2 = (dL, T) => 2 * dL * 1e3 / T, dv4 = (dL, T) => 4 * dL * 1e3 / T;
const back = E.map(e => Math.abs(e.summary.Lfinal - e.home)).filter(d => d > 1), gMed = q(gaps, .5), g10 = q(gaps, .1);
const once = Math.abs(home.km - 3000);
const out = {
  label: 'S5 home separation, computed from the priors (not hard-coded)',
  home_from_priors: {km: home.km, U: r3(home.U), flat_km: home.flat.map(Math.round), flat_band: `U within +${A.homeFlat} of the minimum`, U_at_3000: r3(home.U3000),
    lambda_km: {median: Math.round(home.lam.median), p10: Math.round(home.lam.p10), p90: Math.round(home.lam.p90)}, n_prior_samples: home.n,
    method: 'λ = V_L1 · τ_c(median), V_L1 from the forecaster\'s joint prior (data triples + jitter + model-error term, accepted like a member), Gauss–Hermite over ln τ_c (σ ' + r3(SSP.lnSpread ? SSP.lnSpread(P.tau_c) : P.tau_c.sigma_ln) + ')'},
  cross_checks: {first_forecast_of_each_event: firstForecast, true_lambda_of_each_event: truth, n_events: n},
  between_storms: {
    real_gaps_days: {n: gaps.length, median: r3(gMed), p10: r3(g10), p90: r3(q(gaps, .9)), source: 'results/events.csv, consecutive CME-driven shock arrivals at L1'},
    once_3000_to_home: {dL_km: once, dv_ms: Object.fromEntries([30, 90, 180].map(d => [d + ' days', {two_impulse: r3(dv2(once, d * DAY)), bang_bang: r3(dv4(once, d * DAY))}]))},
    return_after_event: {events_moved: back.length, dL_km: st(back),
      dv_ms_median_move: {['median gap ' + gMed.toFixed(1) + ' d']: {two_impulse: r3(dv2(q(back, .5), gMed * DAY)), bang_bang: r3(dv4(q(back, .5), gMed * DAY))},
        ['10th-percentile gap ' + g10.toFixed(1) + ' d']: {two_impulse: r3(dv2(q(back, .5), g10 * DAY)), bang_bang: r3(dv4(q(back, .5), g10 * DAY))}},
      gaps_long_enough_for_1ms_two_impulse_pct: back.length ? r3(100 * gaps.filter(g => dv2(q(back, .5), g * DAY) < 1).length / gaps.length) : null},
    note: 'free-space transfer: the natural relative dynamics near L1 and formation station-keeping are outside the model'}};
// 4. sensitivity: noise ratio SN × τ_c scale
out.sensitivity = [];
for (const noise of [.03, .075, .15]) for (const ts of [.5, 1, 2]) {
  const h = SSP.homeSep(P, {noise, tauScale: ts, force: true});
  out.sensitivity.push({noise, tau_scale: ts, tau_c_s: Math.round(h.tauMed), home_km: h.km, flat_km: h.flat.map(Math.round), U_home: r3(h.U), U_3000: r3(h.U3000)})}
// cross-check: the same optimum from the MEASURED λ of every training-year sheath (no drag model)
const LE = P.lamEvents || [];
const emp = SSP.homeFromLam(LE.filter(e => e.train).map(e => e.lam), 0);
out.cross_checks.measured_lambda_train_years = {km: emp.km, U: r3(emp.U), flat_km: emp.flat.map(Math.round), n: emp.n};
// 5. rolling home, monthly
const ser = SSP.homeSeries(P), dvs = ser.slice(1).map(m => m.dv), dLs = ser.slice(1).map(m => Math.abs(m.dL));
const byYear = {}; ser.forEach(m => {const y = new Date(m.t).getUTCFullYear(); (byYear[y] = byYear[y] || []).push(m)});
out.rolling_home = {window_events: A.homeWindow, months: ser.length, from: new Date(ser[0].t).toISOString().slice(0, 7), to: new Date(ser[ser.length - 1].t).toISOString().slice(0, 7),
  home_km: st(ser.map(m => m.km)), min_km: Math.min(...ser.map(m => m.km)), max_km: Math.max(...ser.map(m => m.km)),
  monthly_change_km: st(dLs), months_without_change_pct: r3(100 * dLs.filter(d => d === 0).length / dLs.length),
  dv_per_month_ms: {...st(dvs), mean: r3(dvs.reduce((a, b) => a + b, 0) / dvs.length), max: r3(Math.max(...dvs)), below_0_1_pct: r3(100 * dvs.filter(d => d < .1).length / dvs.length)},
  dv_per_year_ms: r3(dvs.reduce((a, b) => a + b, 0) / (dvs.length / 12)),
  window_span_days: st(ser.map(m => m.span_d)),
  by_year: Object.fromEntries(Object.entries(byYear).map(([y, a]) => [y, {home_km: Math.round(q(a.map(m => m.km), .5)), window_span_d: Math.round(q(a.map(m => m.span_d), .5)),
    v_shock_kms: Math.round(q(a.map(m => m.vMed), .5)), tau_c_s: Math.round(q(a.map(m => m.tauMed), .5)), dv_ms: r3(a.reduce((s, m) => s + m.dv, 0))}])),
  dv_rule: 'two impulses within the month: 2·|ΔL| / month'};
fs.writeFileSync(path.join(ROOT, 'results', 'home_rolling.json'), JSON.stringify(ser.map(m => ({month: new Date(m.t).toISOString().slice(0, 7), home_km: m.km,
  flat_km: [Math.round(m.lo), Math.round(m.hi)], U: r3(m.U), window_span_d: Math.round(m.span_d), v_shock_kms: Math.round(m.vMed), tau_c_s: m.tauMed, dL_km: m.dL, dv_ms: r3(m.dv)})), null, 1));
// 6. the event-time re-planner, from home
const why = {}; let nd = 0;
for (const e of E) for (const d of e.dec) {nd++;
  const ok = d.cands.filter(c => c.ok), best = ok.reduce((a, b) => !a || b.J < a.J ? b : a, null);
  const k = d.action === 'retarget' ? 'retarget' : d.Uh <= A.uThr ? 'U at or below the limit' : !best ? 'no candidate reachable in budget and time' : d.Uh - best.U <= A.uHyst ? 'best affordable gain below the hysteresis' : 'J of the best move not below holding';
  why[k] = (why[k] || 0) + 1}
out.event_time_replanner = {decision_times: nd, events: n, outcome: why};
// station-keeping (assumed): Δv per craft and year
out.station_keeping_assumed = {every_days: A.skEvery / DAY, jitter_days: A.skJit / DAY, dv_ms: A.skDv, dv_per_year_ms: r3(365 * DAY / A.skEvery * (A.skDv[0] + A.skDv[1]) / 2)};
fs.writeFileSync(path.join(ROOT, 'results', 's5_home.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
