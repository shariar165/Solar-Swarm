// Fix 2: calibrate the forecaster's errors to the real drag-model test error.
//   node tests/js/calibrate_errors.js [n=1000]   ->  results/calibration_errors.json
// Target: the MEAN absolute ETA error of the real M1 drag model on the test years
// (results/summary.json, dbm.models.M1.test.mae_h). The Monte Carlo statistic is the same one:
// the mean |ETA error| of the first forecast (T+0.3 h) over seeds 1 … n.
// 1. scale the v0 observation error (log-normal sigma) up to a cap of 0.35;
// 2. if that is not enough, add a model-error term — extra w_eff spread around each data triple,
//    in the hidden truth AND the forecast members (so the forecaster knows its size and stays
//    calibrated) — and bisect it until the mean error matches.
// Then split the mean squared error into its sources (Shapley average over both orders).
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');
global.window = {}; require(path.join(ROOT, 'ui', 'event_priors.js')); const P = global.window.SS_PRIORS;
const SSP = require(path.join(ROOT, 'ui', 'ss-physics-v11.js')), A = SSP.A, H = 3600;
const n = +process.argv[2] || 1000, CAP = .35;
const real = JSON.parse(fs.readFileSync(path.join(ROOT, 'results', 'summary.json'), 'utf8')).dbm.models.M1.test;

const q = (a, p) => {a = a.filter(Number.isFinite).sort((x, y) => x - y); const i = (a.length - 1) * p, lo = Math.floor(i); return a[lo] + (a[Math.min(lo + 1, a.length - 1)] - a[lo]) * (i - lo)};
const mean = a => {a = a.filter(Number.isFinite); return a.reduce((x, y) => x + y, 0) / a.length};
const cov = a => {a = a.filter(Number.isFinite); return a.filter(z => Math.abs(z) <= 2).length / a.length};
const r2 = x => Math.round(x * 100) / 100, r3 = x => Math.round(x * 1000) / 1000;
function run(obsV0, modelW, modelG = 0) {
  const keep = {obsV0: A.obsV0, modelW: A.modelW, modelG: A.modelG};
  Object.assign(A, {obsV0, modelW, modelG});
  const B = []; for (let i = 1; i <= n; i++) B.push(SSP.makeEvent({seed: i}, P).summary);
  Object.assign(A, keep);
  const e = k => {const a = B.map(s => s[k] / H); return {mean: r2(mean(a)), median: r2(q(a, .5)), rmse: r2(Math.sqrt(mean(a.map(x => x * x))))}};
  const row = {obsV0, modelW, modelG, eta_err_h: {T0: e('err0'), T12: e('err12'), T24: e('err24'), last_before_contact: e('errLast')},
    coverage_2sigma: {initial: r3(cov(B.map(s => s.z0))), all_updates: r3(cov(B.flatMap(s => s.zs))), last_before_contact: r3(cov(B.map(s => s.zLast)))},
    mse0: mean(B.map(s => (s.err0 / H) ** 2))};
  console.error(`obsV0 ${obsV0.toFixed(3)}  w-model ${modelW.toFixed(1)} km/s  ->  T+0.3 h mean ${row.eta_err_h.T0.mean} h, median ${row.eta_err_h.T0.median} h, coverage ${row.coverage_2sigma.all_updates}`);
  return row}

const target = real.mae_h, scan = [];
for (const v of [.10, .15, .20, .25, .30, CAP]) scan.push(run(v, 0));
let obsV0 = CAP, modelW = 0;
if (scan[scan.length - 1].eta_err_h.T0.mean >= target) {           // the observation error alone is enough
  let lo = .10, hi = CAP; for (let i = 0; i < 8; i++) {const m = (lo + hi) / 2; (run(m, 0).eta_err_h.T0.mean < target ? (lo = m) : (hi = m))}
  obsV0 = Math.round((lo + hi) / 2 * 100) / 100;
} else {
  let lo = 0, hi = 300; for (let i = 0; i < 8; i++) {const m = (lo + hi) / 2; (run(CAP, m).eta_err_h.T0.mean < target ? (lo = m) : (hi = m))}
  modelW = Math.round((lo + hi) / 2 / 5) * 5;
}
const chosen = run(obsV0, modelW);
// sources of the T+0.3 h mean squared error: unknown (γ, w_eff) within the data triples (baseline, v0
// nearly exact), the v0 observation error, and the model-error term — Shapley average over both orders
const base = run(.02, 0), onlyObs = run(obsV0, 0), onlyMod = run(.02, modelW);
const dObs = ((onlyObs.mse0 - base.mse0) + (chosen.mse0 - onlyMod.mse0)) / 2, dMod = ((onlyMod.mse0 - base.mse0) + (chosen.mse0 - onlyObs.mse0)) / 2;
const tot = base.mse0 + dObs + dMod, pct = x => Math.round(1000 * x / tot) / 10;
const out = {
  label: 'calibrated to the real test error',
  target: {statistic: 'mean |ETA error| of the first forecast (T+0.3 h)', real_M1_test_mean_h: r2(real.mae_h), real_M1_test_median_h: r2(real.median_abs_h), n_real: real.n},
  n_events: n, obs_v0_cap: CAP,
  chosen: {obsV0, modelW_kms: modelW, modelG: 0, note: obsV0 >= CAP ? `v0 error capped at ${CAP}; the rest is a model-error term on w_eff` : 'v0 error alone'},
  result: chosen,
  scan_obs_v0: scan,
  error_sources_mse_pct: {unknown_gamma_w_within_triples: pct(base.mse0), v0_observation_error: pct(dObs), model_error_term: pct(dMod),
    method: 'Shapley average over both orders, T+0.3 h squared error; baseline = v0 error 2 %, no model term'},
  attribution_runs: {baseline: base, obs_only: onlyObs, model_only: onlyMod}};
fs.writeFileSync(path.join(ROOT, 'results', 'calibration_errors.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify({chosen: out.chosen, T0: chosen.eta_err_h.T0, coverage: chosen.coverage_2sigma, sources: out.error_sources_mse_pct}, null, 1));
