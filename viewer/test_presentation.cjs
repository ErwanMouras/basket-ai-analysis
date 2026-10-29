const test = require('node:test');
const assert = require('node:assert/strict');
const presentation = require('./presentation.js');

const person = (x = 0, extra = {}) => ({role: 'player', track_id: 7, bbox: [x, 10, x + 20, 110], ...extra});
const row = (index, p = person()) => ({frame_index: index, segment_id: 0, width: 500, height: 300, persons: [p]});
const identity = {team_id: 'home', number: '0', player_name: 'Player'};

test('four-frame anonymous appearances are hidden; five-frame appearances are kept from their first image', () => {
  const spans = {'0:player:7': [[0, 3], [10, 14]]};
  assert.equal(presentation.visible(row(0), person(), spans), false);
  assert.equal(presentation.visible(row(10), person(), spans), true);
  assert.equal(presentation.visible(row(14), person(), spans), true);
  assert.equal(presentation.visible(row(5), person(), spans), false);
});

test('real identities survive short spans, while numbers and suppressed identities do not bypass the filter', () => {
  const spans = {'0:player:7': [[0, 0]]};
  assert.equal(presentation.visible(row(0), person(0, {resolved_identity: identity}), spans), true);
  assert.equal(presentation.visible(row(0), person(0, {majority_jersey_number: '0'}), spans), false);
  assert.equal(presentation.visible(row(0), person(0, {resolved_identity: identity, jersey_number_suppressed: true}), spans), false);
  assert.equal(presentation.visible(row(0), person(0, {track_id: null}), spans), false);
});

test('role and scene namespaces prevent an old track lifespan from qualifying another body', () => {
  const spans = {'0:player:7': [[0, 10]], '0:referee:7': [[0, 0]], '1:player:7': [[0, 0]]};
  assert.equal(presentation.visible(row(0), person(0, {role: 'referee'}), spans), false);
  assert.equal(presentation.visible({...row(0), segment_id: 1}, person(), spans), false);
});

test('runs without lifespan metadata remain usable and the filter can be disabled', () => {
  assert.equal(presentation.visible(row(0), person(), undefined), true);
  assert.equal(presentation.visible(row(0), person(), {'0:player:7': [[0, 0]]}, 0), true);
});

test('symmetric smoothing preserves constant-speed position without temporal lag', () => {
  const frames = Array.from({length: 5}, (_, i) => row(i, person(i * 5)));
  assert.deepEqual(presentation.smoothBox(frames[2], frames[2].persons[0], i => frames[i]), frames[2].persons[0].bbox);
});

test('a stationary box with a one-frame jitter is stabilized without modifying source geometry or masks', () => {
  const mask = {roi: [0, 0, 10, 10], counts: [0, 100]};
  const frames = Array.from({length: 5}, (_, i) => row(i, person(i === 2 ? 12 : 0, {segmentation: {mask}})));
  const original = JSON.stringify(frames);
  const display = presentation.prepare(frames[2], null, i => frames[i], {smoothBoxes: true});
  assert.deepEqual(display.persons[0].bbox, [4, 10, 24, 110]);
  assert.equal(display.persons[0].segmentation.mask, mask);
  assert.equal(JSON.stringify(frames), original);
  assert.deepEqual(presentation.prepare(frames[2], null, i => frames[i]).persons[0].bbox, [12, 10, 32, 110]);
});

test('cuts, missing observations, identity changes and teleports stop smoothing', () => {
  for (const reason of ['cut', 'gap', 'identity', 'teleport', 'segment']) {
    const frames = Array.from({length: 5}, (_, i) => row(i, person(i === 2 ? 12 : 0, {resolved_identity: identity})));
    if (reason === 'cut') frames[3].scene_cut = true;
    if (reason === 'gap') frames[1].persons = [];
    if (reason === 'identity') frames[1].persons[0].resolved_identity = {...identity, number: '1'};
    if (reason === 'teleport') frames[1].persons[0].bbox = [400, 10, 420, 110];
    if (reason === 'segment') frames[1].segment_id = 1;
    assert.deepEqual(presentation.smoothBox(frames[2], frames[2].persons[0], i => frames[i]), frames[2].persons[0].bbox, reason);
  }
});

test('random access is deterministic and no absent body is fabricated', () => {
  const frames = Array.from({length: 5}, (_, i) => row(i, person(i === 2 ? 12 : 0)));
  const at = i => frames[i];
  const first = presentation.prepare(frames[2], null, at, {smoothBoxes: true});
  presentation.prepare(frames[4], null, at, {smoothBoxes: true});
  assert.deepEqual(presentation.prepare(frames[2], null, at, {smoothBoxes: true}), first);
  assert.deepEqual(presentation.prepare({...frames[2], persons: []}, null, at, {smoothBoxes: true}).persons, []);
});
