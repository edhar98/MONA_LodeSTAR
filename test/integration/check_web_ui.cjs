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
  return {style: {}, disabled: true, src: 'old-image', onclick: () => {}, options: [], selectedOptions: [], value: '',
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
  vm.runInContext('showAbpResults({D_t:1,v0:2,D_r_msd:3,analysis_note:"Pooled <A> and <B>; filter one class"})', context);
  assert.match(statuses.at(-1)[1], /Pooled &lt;A&gt; and &lt;B&gt;/);
  assert.equal(statuses.at(-1)[2], 'warn');
  vm.runInContext('showAbpResults({plot_error: "failed"})', context);
  assert.equal(statuses.at(-1)[2], 'err');
  assert.equal(nodes.get('abp-plot').src, undefined);
  vm.runInContext('let models = [{id:"a",particle_name:"A"},{id:"b",particle_name:"B"},{id:"c",particle_name:"A"}];', context);
  vm.runInContext(section('function compositeSelection(', 'function updateDetectionModeUI('), context);
  const get = id => context.document.getElementById(id);
  get('det-composite').checked = true;
  get('det-mode').value = 'standard';
  get('det-composite-distance').value = '20';
  get('det-composite-models').selectedOptions = [{value:'a'}];
  assert.throws(() => vm.runInContext('compositeSelection()', context), /2–8/);
  get('det-composite-models').selectedOptions = [{value:'a'},{value:'c'}];
  assert.throws(() => vm.runInContext('compositeSelection()', context), /distinct/);
  get('det-composite-models').selectedOptions = [{value:'a'},{value:'b'}];
  assert.equal(vm.runInContext('compositeSelection().model_ids.length', context), 2);
  get('det-mode').value = 'template';
  assert.throws(() => vm.runInContext('compositeSelection()', context), /Standard/);
  get('det-mode').value = 'standard';
  get('det-composite-distance').value = 'NaN';
  assert.throws(() => vm.runInContext('compositeSelection()', context), /distance/);

  const stored = new Map();
  context.localStorage = {getItem:k=>stored.get(k), setItem:(k,v)=>stored.set(k,v)};
  context.apiUrl = p => '/proxy' + p;
  context.refreshResultFiles = async () => {};
  context.clearInterval = () => {};
  context.Option = function(text, value) { return {text, value}; };
  vm.runInContext('let username = "tester"; let lastTrackCsv = "original_tracks.csv";', context);
  vm.runInContext(section('let trajectoryModels = [];', 'function refreshVizFileList('), context);
  await vm.runInContext('showTrajectoryResult({method:"causal_prediction",output_kind:"predictions",output_csv:"sample_predictions.csv",counts:{prediction_rows:0},warnings:["<unsafe>"]})', context);
  assert.equal(get('trajectory-use').classList.contains('hidden'), true);
  assert.match(get('trajectory-result').innerHTML, /&lt;unsafe&gt;/);
  assert.equal(statuses.at(-1)[2], 'warn');
  assert.equal(vm.runInContext('lastTrackCsv', context), 'original_tracks.csv');
  await vm.runInContext('showTrajectoryResult({method:"bilstm_gap",output_kind:"tracks",output_tracks_csv:"refined_tracks.csv",counts:{refined_rows:3,mean_shift_px:1.2,median_shift_px:1,p95_shift_px:2.5}})', context);
  assert.match(get('trajectory-result').innerHTML, /Mean correction shift \(px\)<\/td><td>1.2/);
  assert.match(get('trajectory-result').innerHTML, /Median correction shift \(px\)<\/td><td>1/);
  assert.match(get('trajectory-result').innerHTML, /95th percentile correction shift \(px\)<\/td><td>2.5/);
  assert.equal(get('trajectory-use').classList.contains('hidden'), false);
  assert.equal(vm.runInContext('lastTrackCsv', context), 'original_tracks.csv'); // explicit opt-in only
  context.api = async () => ({status:'interrupted'});
  vm.runInContext('trajectoryJobId = "job";', context);
  await vm.runInContext('pollTrajectoryJob()', context);
  assert.equal(get('trajectory-run').disabled, false);
  assert.equal(get('trajectory-use').onclick, null);
  assert.match(statuses.at(-1)[1], /Interrupted/);
  context.api = async () => ({status:'failed',error:'bad <checkpoint>'});
  vm.runInContext('trajectoryJobId = "job";', context);
  await vm.runInContext('pollTrajectoryJob()', context);
  assert.match(statuses.at(-1)[1], /failed: bad &lt;checkpoint&gt;/);
  context.api = async () => ({error:'offline'});
  vm.runInContext('trajectoryJobId = "job";', context);
  await vm.runInContext('pollTrajectoryJob()', context);
  assert.match(statuses.at(-1)[1], /reconnect/);
  assert.equal(get('trajectory-run').disabled, false);
  get('trajectory-tracks').value = 'raw_tracks.csv';
  get('trajectory-model').value = 'model';
  await vm.runInContext('runTrajectoryModel()', context);
  assert.equal(get('trajectory-run').disabled, false);
  assert.equal(get('trajectory-use').classList.contains('hidden'), true);
  console.log('PASS: JS syntax, proxy prefixes, training/ABP recovery, composite validation, trajectory no-op/escaping/opt-in/failure/recovery');
})().catch(error => { console.error(error); process.exitCode = 1; });
