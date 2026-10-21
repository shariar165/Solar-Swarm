// Node helper: load the v11 physics module and the event priors exactly as the browser does.
const fs = require('fs'), path = require('path');
const UI = path.join(__dirname, '..', '..', 'ui');
function loadPriors(file = 'event_priors.js') {
  const f = path.join(UI, file);
  if (!fs.existsSync(f)) return null;
  const s = fs.readFileSync(f, 'utf8');
  return JSON.parse(s.slice(s.indexOf('=') + 1, s.trimEnd().lastIndexOf(';')));
}
module.exports = {SSP: require(path.join(UI, 'ss-physics-v11.js')), loadPriors};
