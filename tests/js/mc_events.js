// Per-event Monte Carlo summaries from the JS physics: node tests/js/mc_events.js <n> [seed0] [priors file]
// Used by tests/test_montecarlo.py to check the Python port (ssw/montecarlo.py) against the browser model.
const {SSP, loadPriors} = require('./load.js');
const n = +(process.argv[2] || 200), s0 = +(process.argv[3] || 1), P = loadPriors(process.argv[4] || 'event_priors.js');
const keys = ['seed', 'gain', 'err0', 'err12', 'err24', 'errLast', 'errPost', 'z0', 'zLast', 'dv', 'burns', 'Lfinal', 'nImg', 'movingAtContact'];
const out = [];
for (let i = 0; i < n; i++) {const s = SSP.makeEvent({seed: s0 + i}, P).summary; out.push(Object.fromEntries(keys.map(k => [k, Number.isFinite(s[k]) || typeof s[k] === 'boolean' ? s[k] : null])))}
console.log(JSON.stringify(out));
