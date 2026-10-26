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
    const handlers = new Map();
    const attributes = new Map();
    return {
      textContent: '', hidden: false, value: '', checked: true, paused: true,
      width: 100, height: 50, currentTime: 0, style: { setProperty() {} },
      classList: {
        add: name => classes.add(name), remove: name => classes.delete(name), contains: name => classes.has(name),
        toggle(name, enabled) { if (enabled ?? !classes.has(name)) classes.add(name); else classes.delete(name); },
      },
      querySelector(name) { if (!children.has(name)) children.set(name, element()); return children.get(name); },
      replaceChildren() {}, append() {}, add() {},
      addEventListener(name, handler) { handlers.set(name, handler); },
      dispatch(name) { handlers.get(name)?.(); },
      setAttribute(name, value) { attributes.set(name, value); }, getAttribute(name) { return attributes.get(name); }, removeAttribute(name) { attributes.delete(name); },
      pause() { this.paused = true; }, load() {}, requestVideoFrameCallback() {},
      getBoundingClientRect: () => ({ width: 100, height: 50, left: 0, top: 0 }),
      getContext: () => new Proxy({}, { get: (_, name) => name === 'measureText' ? () => ({ width: 20 }) : (...args) => drawCalls.push([name,...args]) }),
    };
  }
  const byId = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
  const context = vm.createContext({
    document: {
      getElementById: byId, querySelectorAll: selector => selector.includes('data-filter') ? inputs : [], querySelector: byId,
      documentElement: {}, createTextNode: text => ({textContent: text}),
      createElement: element, createElementNS: element,
      addEventListener: (name, handler) => events.set(name, handler),
    },
    window: { addEventListener() {} }, location: { search: '' }, history: { replaceState() {} },
    ResizeObserver: class { observe() {} }, URLSearchParams,
    fetch: async () => ({ ok: true, json: async () => ({ runs: [] }) }),
    Option: function(text, value) { return {textContent: text, value}; },
    setTimeout: () => 0, clearTimeout() {},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'i18n.js'), 'utf8'), context);
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

test('cached masks wait for the newly presented image after seeking', async () => {
  const ui = await app();
  ui.run(`state.meta = {timestamps:[0,.1], run:{source:{width:100,height:50}}};
    state.cache.set(1,{observation:{frame_index:1,segment_id:0,width:100,height:50,persons:[]}});
    seekToFrame(1)`);
  ui.byId('video').dispatch('seeked');
  assert.equal(ui.run('state.current'), -1);
  assert.equal(ui.run('state.frame'), null);
  assert.ok(ui.run('video.currentTime > .1 && video.currentTime < .2'));
  ui.run('state.loaded=true; videoFrameLoop(0,{mediaTime:.1})');
  assert.equal(ui.run('state.current'), 1);
  assert.equal(ui.run('state.frame.frame_index'), 1);
});

test('arrow navigation advances one source second regardless of frame rate', async () => {
  const ui = await app();
  ui.run(`state.meta={timestamps:Array.from({length:20},(_,i)=>i/10),run:{source:{width:100,height:50}}};
    for(let i=0;i<20;i++)state.cache.set(i,{observation:{frame_index:i,segment_id:0,width:100,height:50,persons:[]}})`);
  ui.events.get('keydown')({ code: 'ArrowRight', shiftKey: false, preventDefault() {} });
  assert.equal(ui.run('state.current'), -1);
  assert.ok(ui.run('video.currentTime >= 1 && video.currentTime < 1.1'));
});

test('seeked keeps the presented PTS even when currentTime rounds toward the next frame', async () => {
  const ui=await app();
  ui.run(`state.loaded=true;state.meta={timestamps:[51.6,51.6333333333],run:{source:{width:100,height:50}}};
    state.cache.set(0,{observation:{frame_index:0,width:100,height:50,persons:[]}});
    video.currentTime=51.619;videoFrameLoop(0,{mediaTime:51.6})`);
  ui.byId('video').dispatch('seeked');
  assert.equal(ui.run('state.current'),0);
  assert.equal(ui.run('frameAtTime(51.619)'),0);
  assert.equal(ui.run('frameAtTime(51.633333)'),1);
});

test('a late annotation response during a seek cannot restore the previous mask', async () => {
  const ui=await app();let resolve;
  ui.context.fetch=()=>new Promise(done=>{resolve=done;});
  ui.run('state.meta={timestamps:[0,.1]};state.target=0');
  const pending=ui.run('fetchOne(0)');
  ui.byId('video').dispatch('seeking');
  resolve({ok:true,json:async()=>({observation:{frame_index:0}})});
  await pending;
  assert.equal(ui.run('state.current'),-1);
  assert.equal(ui.run('state.frame'),null);
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

function layerFrame(ui) {
  ui.run("state.meta={run:{source:{width:10,height:10}}};state.frame={width:10,height:10,segment_id:0,persons:[{role:'player',bbox:[1,1,5,5],track_id:1,segmentation:{status:'ok',mask:{encoding:'roi_rle_c',roi:[1,1,5,5],counts:[0,16]}}},{role:'referee',bbox:[5,1,9,5],track_id:2,segmentation:{status:'ok',mask:{encoding:'roi_rle_c',roi:[5,1,9,5],counts:[0,16]}}}]}");
  ui.byId('overlay').width=10;ui.byId('overlay').height=10;
  ui.drawCalls.length=0;
}

test('players can show segmentation alone while referees keep their boxes', async () => {
  const ui=await app();layerFrame(ui);
  ui.byId('playerDisplay').value='segmentation';
  ui.byId('refereeDisplay').value='bbox';
  ui.run('drawOverlay(state.frame)');
  assert.equal(ui.drawCalls.filter(c=>c[0]==='drawImage').length,1);
  assert.deepEqual(ui.drawCalls.filter(c=>c[0]==='strokeRect'),[['strokeRect',5,1,4,4]]);
  assert.equal(ui.drawCalls.filter(c=>c[0]==='fillText').length,0);
});

test('both representations and hidden categories are independent', async () => {
  const ui=await app();layerFrame(ui);
  ui.byId('playerDisplay').value='both';
  ui.byId('refereeDisplay').value='hidden';
  ui.run('drawOverlay(state.frame)');
  assert.equal(ui.drawCalls.filter(c=>c[0]==='drawImage').length,1);
  assert.deepEqual(ui.drawCalls.filter(c=>c[0]==='strokeRect'),[['strokeRect',1,1,4,4]]);
  assert.equal(ui.run('visiblePerson(state.frame.persons[1],0,filters())'),false);
});

test('labels can accompany masks without boxes and remain category specific', async () => {
  const ui=await app();layerFrame(ui);
  ui.byId('playerDisplay').value='segmentation';
  ui.byId('refereeDisplay').value='hidden';
  ui.inputs.push({dataset:{filter:'playerLabels'},checked:true},{dataset:{filter:'refereeLabels'},checked:true});
  ui.run('drawOverlay(state.frame)');
  assert.equal(ui.drawCalls.filter(c=>c[0]==='strokeRect').length,0);
  assert.equal(ui.drawCalls.filter(c=>c[0]==='fillText').length,1);
});

test('mask-only mode never falls back to boxes for an old run without masks', async () => {
  const ui=await app();layerFrame(ui);
  ui.byId('playerDisplay').value='segmentation';ui.byId('refereeDisplay').value='hidden';
  ui.run('delete state.frame.persons[0].segmentation;drawOverlay(state.frame)');
  assert.equal(ui.drawCalls.filter(c=>c[0]==='drawImage').length,0);
  assert.equal(ui.drawCalls.filter(c=>c[0]==='strokeRect').length,0);
});


test('a destination presented before the seeking event keeps its synchronized overlay', async () => {
  const ui = await app();
  ui.run(`state.loaded=true; state.meta={timestamps:[0,.1],run:{source:{width:100,height:50}}};
    state.cache.set(1,{observation:{frame_index:1,width:100,height:50,persons:[]}});
    video.currentTime=.101; videoFrameLoop(0,{mediaTime:.1})`);
  ui.byId('video').dispatch('seeking');
  ui.byId('video').dispatch('seeked');
  assert.equal(ui.run('state.current'), 1);
  assert.equal(ui.run('state.presentedTime'), .1);
});

test('changing language preserves the playback position and category choices', async () => {
  const ui = await app();
  ui.run("video.currentTime=12.5; $('playerDisplay').value='segmentation'; $('identityMode').value='identified'; setLanguage('en')");
  assert.equal(ui.run('video.currentTime'), 12.5);
  assert.equal(ui.byId('playerDisplay').value, 'segmentation');
  assert.equal(ui.byId('identityMode').value, 'identified');
  assert.equal(ui.run('t("coverage")'), 'Tracking coverage');
  assert.equal(ui.run('fmtDistance(12.5)'), '12.5');
  ui.run("setLanguage('fr')");
  assert.equal(ui.run('fmtDistance(12.5)'), '12,5');
  assert.equal(ui.run('t("positions", {count: 1})'), '1 position sur le terrain');
});

test('both languages cover the static interface and dynamic message catalog', async () => {
  const ui = await app();
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  const keys = [...html.matchAll(/data-i18n(?:-aria|-title)?="([^"]+)"/g)].map(match => match[1]);
  for (const lang of ['fr', 'en']) {
    const catalog = ui.json(`messages.${lang}`);
    for (const key of keys) assert.ok(catalog[key], `${lang}: ${key}`);
  }
  assert.deepEqual(Object.keys(ui.json('messages.en')).sort(), Object.keys(ui.json('messages.fr')).sort());
});

test('court projection matches source pixels and refuses invalid transforms', async () => {
  const ui=await app();
  assert.deepEqual(ui.json('courtPixel(inverseHomography([[.1,0,-10],[0,.1,-20],[0,0,1]]),[5,7])'),[150,270]);
  for(const matrix of ['null','[[1,0,0],[1,0,0],[0,0,1]]','[[1,0,0],[0,NaN,0],[0,0,1]]'])assert.equal(ui.run(`inverseHomography(${matrix})`),null);
  assert.equal(ui.run('courtPixel([1,0,0,0,1,0,1,0,-1],[1,4])'),null);
});

test('court lines persist on propagated frames without stale detector points', async () => {
  const ui=await app();
  ui.run(`drawCourtOverlay({width:1280,height:720,court:{status:'propagated',image_to_court:[[.03,0,0],[0,.03,0],[0,0,1]],keypoints:null}}, {court:true,courtLandmarks:true})`);
  assert.ok(ui.drawCalls.some(([name,...args])=>name==='setLineDash' && args[0].length===0));
  assert.ok(ui.drawCalls.some(([name])=>name==='lineTo'));
  assert.equal(ui.drawCalls.filter(([name])=>name==='arc').length,0);
});

test('unavailable, singular and horizon-crossing calibrations never draw court lines', async () => {
  for(const [status,matrix] of [['unavailable','[[1,0,0],[0,1,0],[0,0,1]]'],['fit','[[0,0,0],[0,0,0],[0,0,0]]'],['fit','[[1,0,0],[0,1,0],[1,0,-1]]']]) {
    const ui=await app();
    ui.run(`drawCourtOverlay({width:1280,height:720,court:{status:'${status}',image_to_court:${matrix}}}, {court:true})`);
    assert.equal(ui.drawCalls.filter(([name])=>name==='lineTo').length,0);
  }
});

test('court landmark diagnostics are independent of lines and ignore malformed points', async () => {
  const ui=await app();
  ui.run(`drawCourtOverlay({width:1280,height:720,court:{status:'unavailable',keypoints:[[200,300,.9],[NaN,300,.9],[-4,5,.99],[10,10,.1]]}}, {court:false,courtLandmarks:true})`);
  assert.equal(ui.drawCalls.filter(([name])=>name==='arc').length,1);
  assert.equal(ui.drawCalls.filter(([name])=>name==='fillText').length,1);
  assert.equal(ui.drawCalls.filter(([name])=>name==='lineTo').length,0);
});

test('score timeline uses the last observed sample without showing a future update', async () => {
  const ui = await app();
  ui.run(`state.meta = {scoreboard: {available:true,max_age_s:.75, teams:[{team_id:'away',label:'AAA'},{team_id:'home',label:'HHH'}],samples:[
    {time_s:0,period:1,clock_s:684,scores:{home:0,away:2}},
    {time_s:.5,period:1,clock_s:683,scores:{home:2,away:2}},
    {time_s:1,period:null,clock_s:null,scores:null}
  ]}}`);
  assert.equal(ui.json('scoreReading(.49).scores.home'), 0);
  assert.equal(ui.json('scoreReading(.5).scores.home'), 2);
  assert.equal(ui.json('scoreReading(3)'), null);
  ui.run('renderScoreboard(.5)');
  assert.equal(ui.byId('scoreboardMatch').textContent, 'AAA 2  ·  HHH 2');
  assert.equal(ui.byId('scoreboardClock').textContent, '11:23');
  assert.equal(ui.byId('scoreboardPeriod').textContent, 'QT1');
  ui.run("language='en'; renderScoreboard(1)");
  assert.equal(ui.byId('scoreboardMatch').textContent, 'AAA —  ·  HHH —');
  assert.equal(ui.byId('scoreboardPeriod').textContent, 'Quarter unread');
  ui.run('state.meta=null; renderScoreboard(0)');
  assert.equal(ui.byId('scoreboardContext').hidden, true);
});

test('game clock labels keep quarter and overtime labels separate from source elapsed time', async () => {
  const ui = await app();
  assert.equal(ui.run('gameClockLabel(8.7)'), '0:08.7');
  assert.equal(ui.run('gameClockLabel(null)'), '—');
  assert.equal(ui.run('periodLabel(5)'), 'PR1');
  ui.run("language='en'");
  assert.equal(ui.run('periodLabel(6)'), 'OT2');
});

test('phase label distinguishes court context and contacts without carrying across gaps', async () => {
  const ui=await app();
  ui.run(`state.meta={scoreboard:{teams:[{team_id:'a',label:'AAA'},{team_id:'b',label:'BBB'}]},phase_timeline:[
    {start_s:0,end_s:1,attacking_team_id:'a',source:'court_context'},
    {start_s:1.1,end_s:2,attacking_team_id:'b',source:'contacts'}
  ]}; renderGamePhase(.5)`);
  assert.equal(ui.byId('gamePhase').textContent,'AAA attaque · BBB défense');
  ui.run('renderGamePhase(1)');
  assert.equal(ui.byId('gamePhase').hidden,true);
  assert.equal(ui.byId('gamePhase').textContent,'');
  ui.run("language='en';renderGamePhase(1.1)");
  assert.equal(ui.byId('gamePhase').textContent,'BBB offense · AAA defense');
  ui.run('state.meta=null;renderGamePhase(0)');
  assert.equal(ui.byId('gamePhase').hidden,true);
});
