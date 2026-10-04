// Bounded shipped-JS contracts; simulated DOM is not a browser layout engine.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../../web/templates/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const section = (start, end) => {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert(a >= 0 && b > a, `Missing script section ${start}`);
  return script.slice(a, b);
};
assert.equal((html.match(/id="normalize-check"/g) || []).length, 1);
assert(!html.includes('server-normalize-check'));
assert.match(script, /document\.getElementById\('normalize-check'\)\.onchange = \(\) => \{\s*if \(currentFileId\) showFrame\(currentFileId, currentFrame\);/);
assert.match(script, /if \(d\.files\.some\(f => f\.id === currentFileId\)\) await previewFile\(currentFileId\);/);
assert.match(script, /d\.added \|\| 0.*new,.*d\.updated \|\| 0.*refreshed/);
assert.match(html, /\.panel\s*\{[^}]*min-height:\s*0;[^}]*overflow-y:\s*auto;/);
assert.match(html, /\.panel > \.row, \.panel > \.card\s*\{\s*flex-shrink:\s*0;/);
assert.match(html, /#merged-video\s*\{[^}]*flex:\s*1 0 180px;/);

function classes(initial = []) {
  const values = new Set(initial);
  return {add: v => values.add(v), remove: v => values.delete(v), contains: v => values.has(v)};
}
const nodes = new Map();
const get = id => {
  if (!nodes.has(id)) nodes.set(id, {value: '', style: {}, classList: classes(), querySelector: () => null});
  return nodes.get(id);
};
const events = new Map(), stored = new Map();
const panel = {id: 'panel-files', scrollTop: 0};
let grip;
const card = {dataset: {}, style: {}, classList: classes(),
  closest: () => panel, querySelector: () => ({textContent: 'Merged videos'}),
  parentElement: {getBoundingClientRect: () => ({width: 700})},
  getBoundingClientRect() { return {width: parseFloat(this.style.width) || 400, height: parseFloat(this.style.height) || 240}; },
  appendChild(value) { grip = value; }};
const document = {getElementById: get,
  querySelectorAll: () => [card],
  body: {classList: classes()},
  createElement() { const handlers = {}; return {handlers,
    addEventListener: (name, fn) => { handlers[name] = fn; },
    setPointerCapture() {}, releasePointerCapture() {}}; },
  addEventListener: (name, fn) => events.set(name, fn),
  removeEventListener: name => events.delete(name)};
const calls = [];
const context = vm.createContext({document, window: {innerHeight: 300}, username: 'tester',
  localStorage: {getItem: key => stored.get(key), setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key)},
  uploadedFiles: {tdms: {type: 'tdms'}}, currentFileId: 'tdms', currentFrame: 0,
  setStatus() {}, refreshFileList() {}, esc: String, apiUrl: p => p, alert: message => assert.fail(message),
  api: async (url, data) => { calls.push({url, data});
    if (url === '/upload/start') return {upload_id: 'id', upload_token: 'token'};
    if (url === '/upload/complete') return {id: 'uploaded', type: 'image'};
    if (url === '/files/load-path') return {files: [], count: 0};
    if (url === '/tdms/export') return {path: 'saved'};
    return {error: 'stop after request'}; },
  fetch: async () => ({ok: true})});
vm.runInContext(section('function layoutStorageKey()', '// ====================================================================\n// LOGIN'), context);
vm.runInContext('initResizableCards()', context);
const pointer = {clientX: 100, clientY: 100, pointerId: 1, preventDefault() {}, stopPropagation() {}};
grip.handlers.pointerdown(pointer);
panel.scrollTop = 400;
events.get('pointermove')({...pointer, clientX: 2000, clientY: 500});
assert.equal(card.style.height, '1040px', 'height extends beyond viewport and includes panel scrolling');
assert.equal(card.style.width, '692px', 'width remains parent-bounded');
events.get('pointerup')(pointer);
assert.equal(events.size, 0);
assert.equal(document.body.classList.contains('resizing-card'), false);
const saved = JSON.parse(stored.get('mona_card_layout:tester'));
assert.equal(saved[card.dataset.layoutKey].height, 1040);
card.style.height = '';
context.card = card;
vm.runInContext('applyCardSize(card, loadCardLayout()[card.dataset.layoutKey])', context);
assert.equal(card.style.height, '1040px');
grip.handlers.dblclick(pointer);
assert.equal(card.style.height, '');
assert.deepEqual(JSON.parse(stored.get('mona_card_layout:tester')), {});
grip.handlers.pointerdown(pointer);
events.get('pointermove')({...pointer, clientX: -1000, clientY: -1000});
assert.equal(card.style.height, '120px');
assert.equal(card.style.width, '220px');
events.get('pointerup')(pointer);
get('layout-reset-btn').onclick();
assert.equal(card.style.width, '');
assert.equal(card.style.height, '');
assert.equal(stored.has('mona_card_layout:tester'), false);

vm.runInContext(section('async function showFrame(', "document.getElementById('frame-prev')"), context);
vm.runInContext(section('async function loadFromServerPath()', '// Chunked upload'), context);
vm.runInContext(section('async function doUpload(', '// TDMS Export'), context);
vm.runInContext(section('async function exportTdms(', '// Video merge'), context);
get('server-path-input').value = '/server/example.tdms';
get('export-format').value = 'png';
get('export-fps').value = '30';
context.file = {name: 'sample.png', size: 1, slice: () => ({arrayBuffer: async () => new Uint8Array([1]).buffer})};
(async () => {
  for (const normalize of [false, true]) {
    calls.length = 0;
    get('normalize-check').checked = normalize;
    await vm.runInContext('loadFromServerPath()', context);
    await vm.runInContext('doUpload(file)', context);
    await vm.runInContext('showFrame("tdms", 0)', context);
    await vm.runInContext('exportTdms(true)', context);
    for (const endpoint of ['/files/load-path', '/upload/start', '/upload/complete', '/tdms/export']) {
      assert.equal(calls.find(call => call.url === endpoint).data.normalize, normalize, endpoint);
    }
    assert(calls.some(call => call.url.includes(`/tdms/frame/tester/tdms/0?cmap=gray&normalize=${normalize}`)));
  }
  console.log('PASS: single TDMS normalization upload/path/preview/export; resize beyond viewport, scroll delta, save/restore/reset and bounds');
})().catch(error => { console.error(error); process.exitCode = 1; });
