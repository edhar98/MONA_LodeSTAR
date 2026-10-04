// Native-popup safety contracts; DOM mocks cannot reproduce a browser popup.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../../web/templates/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const document = {activeElement: null};
const timers = [];
const context = vm.createContext({document, setTimeout: fn => timers.push(fn),
  Option: function(text, value) { return {text, value, selected: false}; }});
vm.runInContext(script.slice(script.indexOf('function updateSelectOptions('), script.indexOf('async function refreshModels(')), context);
function select() {
  const events = new Map();
  return {options: [{text: 'A', value: 'a', selected: true}], writes: 0,
    replaceChildren(...options) { this.options = options; this.writes++; },
    addEventListener(name, fn) { events.set(name, fn); },
    removeEventListener(name) { events.delete(name); },
    fire(name) { events.get(name)?.(); }};
}
const control = select();
context.control = control;
const update = options => { context.options = options; vm.runInContext('updateSelectOptions(control, options)', context); };
const a = {text: 'A', value: 'a'}, b = {text: 'B', value: 'b'}, c = {text: 'C', value: 'c'};
update([a]);
assert.equal(control.writes, 0, 'unchanged responses must not replace options');
document.activeElement = control;
update([a, b]);
update([a, b, c]);
assert.equal(control.writes, 0, 'focused/open selector stays intact');
control.fire('change');
timers.shift()();
assert.equal(control.options.length, 3, 'latest response applied after choice');
assert.equal(control.options[0].selected, true, 'selection retained');
update([a]);
document.activeElement = null;
control.fire('blur');
timers.shift()();
assert.equal(control.options.length, 1);
assert.equal(control.options[0].selected, true);
document.activeElement = control;
update([a, b]);
update([a]);
const writes = control.writes;
control.fire('change');
timers.shift()();
assert.equal(control.writes, writes, 'stale queued options cannot replace latest data');
document.activeElement = null;
update([a, b]);
control.options.forEach(o => { o.selected = true; });
document.activeElement = control;
update([a, b, c]);
control.fire('change');
timers.shift()();
assert.deepEqual(control.options.filter(o => o.selected).map(o => o.value), ['a', 'b']);
update([a, b]);
control.options[0].selected = false; // User's new choice wins over refresh-time state.
control.fire('change');
timers.shift()();
assert.deepEqual(control.options.filter(o => o.selected).map(o => o.value), ['b']);

// Focus and pointer activation hide hints; hover cannot reopen while focused.
const events = {};
const tip = {style: {}, offsetWidth: 50, offsetHeight: 20};
const field = {dataset: {}, closest: () => null, addEventListener: (name, fn) => { events[name] = fn; }};
document.getElementById = id => id === 'field-tip' ? tip : field;
context.FIELD_HINTS = {example: 'Hint'};
context.window = {innerWidth: 800, innerHeight: 600};
vm.runInContext(script.slice(script.indexOf('function applyFieldHints('), script.indexOf('async function initApp(')), context);
vm.runInContext('applyFieldHints()', context);
document.activeElement = null;
events.mouseenter({clientX: 1, clientY: 1, buttons: 0});
assert.equal(tip.style.display, 'block');
events.pointerdown();
assert.equal(tip.style.display, 'none');
document.activeElement = field;
events.mousemove({clientX: 2, clientY: 2, buttons: 0});
assert.equal(tip.style.display, 'none');
console.log('PASS: selector refresh preservation/defer/latest-response and interactive tooltip suppression');
