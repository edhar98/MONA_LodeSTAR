// Bounded behavioral checks of the shipped inline JS, without a browser/DOM.
// Run: node test/integration/check_web_ui.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../../web/templates/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Parse the complete frontend, not just selected helpers.
function section(start, end) {
  assert(script.includes(start) && script.includes(end));
  return script.slice(script.indexOf(start), script.indexOf(end, script.indexOf(start)));
}
const prefixCode = section("let API_BASE = '';", 'async function api(');
for (const [pathname, expected] of [
  ['/', '/health'],
  ['/user/alice/proxy/8123/', '/user/alice/proxy/8123/health'],
  ['/user/alice/mona-track/', '/user/alice/mona-track/health'],
  ['/user/alice/server-one/mona-track/', '/user/alice/server-one/mona-track/health'],
  ['/user/alice/mona-track-lab/mona-track/', '/user/alice/mona-track-lab/mona-track/health'],
  ['/prefix/user/alice/server-one/mona-track-2/', '/prefix/user/alice/server-one/mona-track-2/health'],
]) {
  const context = vm.createContext({window: {location: {pathname}}});
  vm.runInContext(prefixCode, context);
  assert.equal(vm.runInContext("apiUrl('/health')", context), expected);
}
function node() {
  const classes = new Set(['hidden']);
  return {style: {}, disabled: true, src: 'old-image', onclick: () => {},
    removeAttribute(name) { delete this[name]; },
    classList: {add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x)}};
}
const nodes = new Map();
let intervalCallback;
let cleared = false;
const statuses = [];
const context = vm.createContext({
  document: {getElementById(id) { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); }},
  setStatus: (...args) => statuses.push(args), esc: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), refreshModels() {}, drawLossChart() {},
  setInterval(fn) { intervalCallback = fn; return 42; },
  clearInterval(id) { assert.equal(id, 42); cleared = true; },
  api: async () => ({status: 'interrupted'}),
});
vm.runInContext('let trainJobId = "resumed"; let lastAbpPlotB64 = "old";', context);
vm.runInContext(section('function pollTrainJob(', 'async function cancelTraining('), context);
vm.runInContext(section('function clearAbpPlot(', 'function optionalFloat('), context);
(async () => {
  vm.runInContext('pollTrainJob("resumed")', context);
  await intervalCallback();
  assert(cleared);
  assert.equal(nodes.get('train-start-btn').disabled, false);
  assert.equal(nodes.get('train-start-btn').classList.contains('hidden'), false);
  assert.equal(vm.runInContext('trainJobId', context), null);
  assert.match(statuses.at(-1)[1], /interrupted/);
  vm.runInContext('showAbpResults({n_tracks: 2, n_real_rows: 6})', context);
  assert.equal(statuses.at(-1)[2], 'warn');
  assert.equal(nodes.get('abp-plot').src, undefined);
  assert.equal(nodes.get('abp-plot-download').onclick, null);
  vm.runInContext('showAbpResults({orientation_note: "No <phi>; angular MSD omitted"})', context);
  assert.match(statuses.at(-1)[1], /No &lt;phi&gt;; angular MSD omitted/);
  vm.runInContext('showAbpResults({D_t: 1, v0: 2, D_r_msd: 3, plot_b64: "new-image"})', context);
  assert.equal(statuses.at(-1)[2], 'ok');
  assert.equal(nodes.get('abp-plot').src, 'new-image');
  vm.runInContext('showAbpResults({plot_error: "failed"})', context);
  assert.equal(statuses.at(-1)[2], 'err');
  assert.equal(nodes.get('abp-plot').src, undefined);
  console.log('PASS: complete JS syntax, 6 proxy prefixes, interrupted training recovery, ABP no-fit/plot-error/stale-image states');
})().catch(error => { console.error(error); process.exitCode = 1; });
