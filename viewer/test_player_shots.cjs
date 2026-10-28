// Player review uses a local synthetic video; no model or NBA footage is needed.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require('playwright');
const base = process.env.VIEWER_URL || 'http://127.0.0.1:8766';
const subject = 'roster:blue:7';
const event = (id, time, start, shooter = subject, status = 'probable') => ({
  id, time_s: time, start_s: start, end_s: time + .6, frame_index: Math.round(time * 30),
  release_window_s: [time - .1, time + .1], status,
  shooter: {subject_id: shooter, team_id: shooter?.includes('pink') ? 'pink' : 'blue'},
});
const fixture = {available: true, probable_count: 4, candidate_count: 1,
  player_counts: {[subject]: 2, 'roster:blue:8': 0, 'roster:pink:12': 1},
  events: [event('first', .7, .2), event('second', 3.7, 3.1),
    event('candidate', 5.1, 4.6, subject, 'candidate'),
    event('other', 2, 1.4, 'roster:pink:12'), event('unassigned', 2.5, 1.9, null)]};

(async () => {
  const browser = await chromium.launch({...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
    ? {executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 1600, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let videoAvailable = true, rosterAvailable = true, shots = fixture, delayFrames = false;
    await page.route('**/api/frames?**', async route => {
      if (delayFrames && new URL(route.request().url()).searchParams.get('start') === '90')
        await new Promise(resolve => setTimeout(resolve, 350));
      await route.continue();
    });
    await page.route('**/api/run?**', async route => {
      const response = await route.fetch();
      const run = await response.json();
      run.shots = shots;
      run.video_available = videoAvailable;
      if (!rosterAvailable) {run.reference_roster = run.roster; run.roster = null;}
      await route.fulfill({response, json: run});
    });
    const wait = () => page.waitForFunction(() => state.loaded && state.frame && video.readyState >= 2);
    const open = async () => {
      await page.locator('#playerShotStat').click();
      await page.waitForFunction(() => playerShotReview.isOpen() && !video.paused && !video.seeking && state.current >= 0);
      await page.locator('#playButton').click();
      await page.waitForFunction(() => video.paused && !video.seeking);
    };
    await page.goto(`${base}/?run=synthetic`);
    await wait();
    await page.selectOption('#playerDisplay', 'both');
    await page.selectOption('#speed', '0.5');
    await page.evaluate(() => {window.originalVideo = video; window.originalCanvas = canvas;});
    await page.locator('#statisticsTab').click();
    await page.locator('.roster-player').filter({hasText: /^7Example A/}).click();
    assert.equal(await page.locator('#playerShotCount').innerText(), '2');
    const original = await page.evaluate(() => ({subject: state.selected.subject, team: state.teamId,
      source: video.src, filters: filters(), rate: video.playbackRate, stats: finalPlayerStats('blue', '7')}));
    await open();
    assert.equal(await page.evaluate(() => state.mode), 'statistics');
    assert.equal(await page.locator('#statisticsTab').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#playerShotTitle').textContent(), 'Example A · #7');
    assert.equal(await page.locator('.player-shot-row').count(), 2);
    assert.equal(await page.locator('#playerShotCandidates').isChecked(), false);
    assert.equal(await page.locator('#playerShotDialog #video').count(), 1);
    assert.equal(await page.locator('#video').count(), 1);
    assert.equal(await page.evaluate(() => originalVideo === video && originalCanvas === canvas), true);
    assert.equal(await page.locator('#playerShotList [aria-current="true"]').getAttribute('data-player-shot'), 'first');
    assert.equal(await page.locator('#previousPlayerShot').isDisabled(), true);

    delayFrames = true;
    await page.locator('[data-player-shot="second"]').click();
    await page.locator('[data-player-shot="first"]').click();
    await page.waitForFunction(() => !video.seeking && state.frame && video.currentTime < 1);
    await page.waitForTimeout(500);
    assert.equal(await page.locator('#playerShotList [aria-current="true"]').getAttribute('data-player-shot'), 'first');
    assert.equal(await page.evaluate(() => state.frame.frame_index === nearestFrame(state.presentedTime)), true, 'a late batch cannot paint another shot over the current image');
    await page.locator('#playButton').click();
    delayFrames = false;

    await page.locator('[data-player-shot="second"]').click();
    await page.waitForFunction(() => !video.seeking && video.currentTime >= 3.1 && state.current >= nearestFrame(3.1));
    await page.locator('#playButton').click();
    assert.equal(await page.locator('#nextPlayerShot').isDisabled(), true);
    await page.locator('#playerShotSeek').fill('3.1');
    await page.waitForFunction(() => !video.seeking && state.current === nearestFrame(3.1));
    await page.locator('#prevSecond').click();
    assert.equal(await page.evaluate(() => video.currentTime >= 3.1 && video.currentTime < 3.14), true, 'navigation stays inside this shot');
    await page.locator('#nextFrame').click();
    await page.waitForFunction(() => state.current === nearestFrame(3.1) + 1);
    await page.locator('#previousPlayerShot').click();
    await page.waitForFunction(() => !video.seeking && video.currentTime < 1);
    await page.locator('#playButton').click();
    await page.locator('#playerShotCandidates').check();
    assert.equal(await page.locator('.player-shot-row').count(), 3);
    await page.locator('[data-player-shot="candidate"]').click();
    await page.waitForFunction(() => !video.seeking && video.currentTime >= 4.6);
    await page.locator('#playerShotCandidates').uncheck();
    await page.waitForFunction(() => !video.seeking && video.currentTime < 1);
    await page.locator('#playButton').click();
    assert.equal(await page.locator('#playerShotCount').innerText(), '2', 'candidates never inflate the player count');

    const beforeLanguage = await page.evaluate(() => ({time: video.currentTime, rate: video.playbackRate}));
    await page.locator('#playerShotLanguage').click();
    assert.equal(await page.locator('#replayPlayerShot').innerText(), 'Replay shot');
    assert.match(await page.locator('#playerShotSummary').innerText(), /2 probable.*1 to review/);
    assert.deepEqual(await page.evaluate(() => ({time: video.currentTime, rate: video.playbackRate})), beforeLanguage);
    await page.locator('#playerShotSeek').fill('2.6');
    await page.waitForFunction(() => !video.seeking);
    await page.locator('#playButton').click();
    await page.waitForFunction(() => video.paused && video.currentTime >= 2.7);
    assert.ok(await page.evaluate(() => video.currentTime < 2.8), 'review stops at the end of the short clip');
    await page.locator('#replayPlayerShot').click();
    await page.waitForFunction(() => !video.seeking && !video.paused && video.currentTime < 1);
    await page.locator('#playButton').click();

    await page.locator('#playerShotStage').focus();
    const beforeStep = await page.evaluate(() => state.current);
    await page.keyboard.press('Shift+ArrowRight');
    await page.waitForFunction(index => state.current === index + 1, beforeStep);
    await page.keyboard.press('Space');
    await page.waitForFunction(() => !video.paused);
    await page.keyboard.press('Space');
    assert.equal(await page.evaluate(() => video.paused), true);

    for (const [width, height] of [[1600, 1000], [1366, 768], [1024, 640], [390, 844]]) {
      await page.setViewportSize({width, height});
      await page.waitForFunction(() => {
        const stage = document.querySelector('#videoStage').getBoundingClientRect();
        const overlay = document.querySelector('#overlay').getBoundingClientRect();
        return overlay.x >= stage.x - 1 && overlay.right <= stage.right + 1;
      }, null, {timeout: 3000});
      const layout = await page.evaluate(() => {
        const dialog = document.querySelector('#playerShotDialog');
        const rect = selector => {const r = document.querySelector(selector).getBoundingClientRect();
          return {x: r.x, y: r.y, width: r.width, height: r.height, right: r.right, bottom: r.bottom};};
        return {width: dialog.clientWidth, scrollWidth: dialog.scrollWidth,
          height: dialog.clientHeight, scrollHeight: dialog.scrollHeight,
          video: rect('#videoStage'), canvas: rect('#overlay'), list: rect('#playerShotList'),
          nav: rect('.player-shot-navigation')};
      });
      assert.ok(layout.scrollWidth <= layout.width, `no horizontal overflow at ${width}`);
      assert.ok(layout.scrollHeight <= layout.height + 1, `review fits height at ${width}`);
      assert.ok(layout.video.height >= (width > 760 ? 200 : 120), `video remains useful at ${width}`);
      assert.ok(layout.canvas.width > 0 && layout.canvas.height > 0);
      assert.ok(layout.canvas.x >= layout.video.x - 1 && layout.canvas.right <= layout.video.right + 1);
      assert.ok(layout.nav.bottom <= height, 'playback navigation stays visible');
    }
    await page.setViewportSize({width: 1600, height: 1000});
    await page.addScriptTag({path: require.resolve('axe-core/axe.min.js')});
    const violations = await page.evaluate(async () => (await axe.run(document, {
      runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa']},
    })).violations.map(v => ({id: v.id, nodes: v.nodes.map(n => n.target)})));
    assert.deepEqual(violations, []);
    if (process.env.VIEWER_SCREENSHOTS) {
      await fs.mkdir(process.env.VIEWER_SCREENSHOTS, {recursive: true});
      await page.screenshot({path: path.join(process.env.VIEWER_SCREENSHOTS, 'player-shot-review.png')});
    }
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !playerShotReview.isOpen() && document.querySelector('.video-panel > #videoStage'));
    assert.equal(await page.locator('#playerShotStat').evaluate(node => node === document.activeElement), true);
    assert.equal(await page.evaluate(() => video.paused), true);
    assert.deepEqual(await page.evaluate(() => ({subject: state.selected.subject, team: state.teamId,
      source: video.src, filters: filters(), rate: video.playbackRate, stats: finalPlayerStats('blue', '7')})),
      original, 'selection, statistics, source, layers and speed survive review');
    assert.equal(await page.locator('.video-panel > .panel-toolbar > #layerControls').count(), 1);
    await page.locator('#visualizationTab').click();
    assert.ok((await page.locator('#videoStage').boundingBox()).height > 200);
    await page.locator('#statisticsTab').click();
    await open();
    await page.evaluate(() => loadRun('synthetic'));
    await wait();
    assert.equal(await page.evaluate(() => playerShotReview.isOpen()), false);
    assert.equal(await page.locator('.video-panel > #videoStage').count(), 1);
    assert.equal(await page.evaluate(() => video.paused), true, 'switching runs stops review');
    await page.locator('.roster-player').filter({hasText: 'Example A2'}).click();
    assert.equal(await page.locator('#playerShotCount').innerText(), '0');
    assert.equal(await page.locator('#playerShotStat').isDisabled(), true);
    shots = {...fixture, probable_count: 0, candidate_count: 1,
      player_counts: {[subject]: 0}, events: [fixture.events[2]]};
    await page.reload();
    await wait();
    await page.locator('#statisticsTab').click();
    assert.equal(await page.locator('#playerShotCount').innerText(), '0');
    await open();
    assert.equal(await page.locator('.player-shot-row.candidate').count(), 1);
    assert.equal(await page.locator('#playerShotCandidatesField').isVisible(), false, 'candidate-only review cannot hide its only event');
    await page.locator('#closePlayerShots').click();
    await page.waitForFunction(() => !playerShotReview.isOpen() && document.querySelector('.video-panel > #videoStage'));
    shots = fixture;
    videoAvailable = false;
    await page.reload();
    await page.waitForFunction(() => state.loaded);
    await page.locator('#statisticsTab').click();
    assert.equal(await page.locator('#playerShotStat').isDisabled(), true, 'missing video cannot open review');
    videoAvailable = true; rosterAvailable = false;
    await page.reload();
    await wait();
    await page.locator('#statisticsTab').click();
    assert.equal(await page.locator('#playerShotCount').innerText(), '—');
    assert.equal(await page.locator('#playerShotStat').isDisabled(), true, 'a reference roster cannot attribute events');
    assert.deepEqual(errors, []);
    console.log('Player shot review: scoped events, playback, clip bounds, frame stepping, overlays, candidates, FR/EN, keyboard close, context restoration, run changes, missing data, responsive layout and accessibility passed');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
