const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

async function app() {
  const elements = new Map();
  const events = new Map();
  const drawCalls = [];
  const inputs = [];
  function element() {
    const classes = new Set();
    const children = new Map();
    return {
      textContent: '', hidden: false, value: '', checked: true, paused: true,
      width: 100, height: 50, currentTime: 0, style: { setProperty() {} },
      classList: {
        add: name => classes.add(name), remove: name => classes.delete(name),
        toggle(name, enabled) { if (enabled ?? !classes.has(name)) classes.add(name); else classes.delete(name); },
      },
      querySelector(name) { if (!children.has(name)) children.set(name, element()); return children.get(name); },
      replaceChildren() {}, append() {}, add() {}, addEventListener() {}, setAttribute() {}, removeAttribute() {},
      pause() { this.paused = true; }, load() {}, requestVideoFrameCallback() {},
      getBoundingClientRect: () => ({ width: 100, height: 50, left: 0, top: 0 }),
      getContext: () => new Proxy({}, { get: (_, name) => name === 'measureText' ? () => ({ width: 20 }) : (...args) => drawCalls.push([name,...args]) }),
    };
  }
  const byId = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
  const context = vm.createContext({
    document: {
      getElementById: byId, querySelectorAll: () => inputs, querySelector: byId,
      createElement: element, createElementNS: element,
      addEventListener: (name, handler) => events.set(name, handler),
    },
    window: { addEventListener() {} }, location: { search: '' }, history: { replaceState() {} },
    ResizeObserver: class { observe() {} }, URLSearchParams,
    fetch: async () => ({ ok: true, json: async () => ({ runs: [] }) }),
    setTimeout: () => 0, clearTimeout() {},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8'), context);
  await new Promise(setImmediate);
  return { context, byId, events, drawCalls, inputs, run: source => vm.runInContext(source, context),
    json: source => JSON.parse(vm.runInContext(`JSON.stringify(${source})`, context)) };
}

test('player coverage is weighted by tracked duration across confirmed records', async () => {
  const ui = await app();
  ui.run(`state.meta = {roster: {}, statistics: {players: [
    {identity_status:'confirmed',team_id:'home',jersey_number:'00',observed_distance_m:2,tracked_duration_s:4,measured_duration_s:3},
    {identity_status:'confirmed',team_id:'home',jersey_number:'00',observed_distance_m:.5,tracked_duration_s:2,measured_duration_s:1},
    {identity_status:'anonymous',team_id:'home',jersey_number:'00',observed_distance_m:99},
    {identity_status:'confirmed',team_id:'away',jersey_number:'00',observed_distance_m:99}
  ]}}`);
  assert.deepEqual(ui.json("finalPlayerStats('home','00')"), {
    observed_distance_m: 2.5, measured_duration_s: 4, coverage_ratio: 2 / 3,
  });
});

test('a reference roster never attributes statistics to an unconfirmed player', async () => {
  const ui = await app();
  ui.run(`state.meta = {reference_roster: {}, statistics: {players: [
    {identity_status:'confirmed',team_id:'home',jersey_number:'0',observed_distance_m:10,tracked_duration_s:2,measured_duration_s:2}
  ]}}`);
  assert.deepEqual(ui.json("finalPlayerStats('home','0')"), {
    observed_distance_m: null, measured_duration_s: null, coverage_ratio: null,
  });
});

test('zero distance is preserved and missing durations stay unavailable', async () => {
  const ui = await app();
  ui.run(`state.meta = {roster: {}, statistics: {players: [
    {identity_status:'confirmed',team_id:'home',jersey_number:'0',observed_distance_m:0}
  ]}}`);
  assert.deepEqual(ui.json("finalPlayerStats('home','0')"), {
    observed_distance_m: 0, measured_duration_s: null, coverage_ratio: null,
  });
});

test('seeking to a cached frame immediately restores its overlays while paused', async () => {
  const ui = await app();
  ui.run(`state.meta = {timestamps:[0,.1], run:{source:{width:100,height:50}}};
    state.cache.set(1,{observation:{frame_index:1,segment_id:0,width:100,height:50,persons:[]}});
    seekToFrame(1)`);
  assert.equal(ui.run('state.current'), 1);
  assert.equal(ui.run('state.frame.frame_index'), 1);
  assert.ok(ui.run('video.currentTime > .1 && video.currentTime < .2'));
});

test('arrow navigation advances one source second regardless of frame rate', async () => {
  const ui = await app();
  ui.run(`state.meta={timestamps:Array.from({length:20},(_,i)=>i/10),run:{source:{width:100,height:50}}};
    for(let i=0;i<20;i++)state.cache.set(i,{observation:{frame_index:i,segment_id:0,width:100,height:50,persons:[]}})`);
  ui.events.get('keydown')({ code: 'ArrowRight', shiftKey: false, preventDefault() {} });
  assert.equal(ui.run('state.current'), 10);
  assert.ok(ui.run('video.currentTime >= 1 && video.currentTime < 1.1'));
});

test('a failed old run cannot replace the current loading state', async () => {
  const ui = await app();
  let reject;
  ui.context.fetch = () => new Promise((_, fail) => { reject = fail; });
  const pending = ui.run("loadRun('old')");
  ui.run("state.generation++; state.name='current'");
  ui.byId('videoEmpty').querySelector('strong').textContent = 'Current run';
  reject(new Error('Old request failed'));
  await pending;
  assert.equal(ui.byId('videoEmpty').querySelector('strong').textContent, 'Current run');
  assert.equal(ui.run('state.name'), 'current');
});

test('a failed old batch cannot evict the current run batch', async () => {
  const ui = await app();
  let reject;
  ui.context.fetch = () => new Promise((_, fail) => { reject = fail; });
  ui.run("state.meta={timestamps:[0]}; state.name='old'");
  const pending = ui.run('fetchBatch(0)');
  ui.run("state.generation++; state.batches.set(0,'current batch')");
  reject(new Error('Old batch failed'));
  await pending;
  assert.equal(ui.run('state.batches.get(0)'), 'current batch');
});


test('ROI RLE foreground spans rows in source order and malformed masks are ignored', async () => {
  const ui=await app(); ui.drawCalls.length=0;
  assert.equal(ui.run("maskSurface({encoding:'roi_rle_c',roi:[2,3,6,6],counts:[2,5,5]},'#ff0000',10,10).width"),4);
  assert.deepEqual(ui.drawCalls.filter(c=>c[0]==='fillRect'),[['fillRect',2,0,2,1],['fillRect',0,1,3,1]]);
  assert.equal(ui.run("maskSurface({encoding:'roi_rle_c',roi:[0,0,4,3],counts:[1,99]},'#ff0000',10,10)"),null);
});

test('segmentation toggle controls only masks and supports old observations', async () => {
  const ui=await app();
  ui.inputs.push({dataset:{filter:'players'},checked:true},{dataset:{filter:'segmentation'},checked:false});
  ui.run("state.meta={run:{source:{width:10,height:10}}};state.frame={width:10,height:10,segment_id:0,persons:[{role:'player',bbox:[1,1,5,5],track_id:1,segmentation:{status:'ok',mask:{encoding:'roi_rle_c',roi:[1,1,5,5],counts:[0,16]}}}]}");
  ui.drawCalls.length=0;ui.run('drawOverlay(state.frame)');
  assert.equal(ui.drawCalls.filter(c=>c[0]==='drawImage').length,0);
  ui.inputs[1].checked=true;ui.run('drawOverlay(state.frame)');
  assert.equal(ui.drawCalls.filter(c=>c[0]==='drawImage').length,1);
  ui.drawCalls.length=0;ui.run('delete state.frame.persons[0].segmentation;drawOverlay(state.frame)');
  assert.equal(ui.drawCalls.filter(c=>c[0]==='drawImage').length,0);
});
