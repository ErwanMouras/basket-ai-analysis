// Run against the public scripted sample; see README.md for the isolated Playwright setup.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright");
const axePath = require.resolve("axe-core/axe.min.js");

const baseURL = process.env.VIEWER_URL || "http://127.0.0.1:8766";
const output = process.env.VIEWER_SCREENSHOTS;
const failures = [];

async function capture(page, name) {
  if (!output) return;
  await fs.mkdir(output, { recursive: true });
  await page.screenshot({
    path: path.join(output, `${name}.png`),
    fullPage: true,
  });
}
async function waitForFrame(page) {
  await page.waitForFunction(
    () => state.loaded && state.frame && video.readyState >= 2,
  );
}
async function openSample(context) {
  const page = await context.newPage();
  await page.goto(`${baseURL}/?run=synthetic`);
  await waitForFrame(page);
  await page.evaluate(() => document.fonts.ready);
  return page;
}
async function assertLayout(page, width, height) {
  await page.setViewportSize({ width, height });
  const layout = await page.evaluate(() => {
    const rect = (selector) => {
      const { x, y, width, height, right, bottom } = document
        .querySelector(selector)
        .getBoundingClientRect();
      return { x, y, width, height, right, bottom };
    };
    return {
      pageWidth: document.documentElement.scrollWidth,
      pageHeight: document.documentElement.scrollHeight,
      video: rect(".video-panel"),
      court: rect(".court-panel"),
      roster: rect(".roster-panel"),
      footer: rect(".workspace-footer"),
      rosterFont: parseFloat(
        getComputedStyle(document.querySelector(".roster-player-name"))
          .fontSize,
      ),
    };
  });
  assert.ok(
    layout.pageWidth <= width,
    `horizontal overflow at ${width}×${height}`,
  );
  assert.ok(layout.rosterFont >= 11, "roster text stays readable");
  if (width > 900) {
    assert.ok(
      layout.pageHeight <= height,
      `desktop must fit viewport at ${width}×${height}`,
    );
    assert.ok(
      layout.court.x >= layout.video.right,
      "court stays to the right of the video",
    );
    assert.ok(layout.video.height > 300, "video retains useful viewing space");
  }
  await page.locator("#statisticsTab").click();
  await settleModes(page);
  assert.equal(await page.locator("#workspace").getAttribute("inert"), "");
  assert.equal(await page.locator("#statisticsWorkspace").getAttribute("inert"), null);
  assert.equal(await page.locator("#statisticsWorkspace").isVisible(), true);
  const stats = await page.locator(".roster-panel").boundingBox();
  assert.ok(stats.x >= 0 && stats.x + stats.width <= width, "statistics fit viewport width");
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  if (width > 900) assert.ok(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight));
  await page.locator("#visualizationTab").click();
  await settleModes(page);
  const transport = await page.locator('.transport-controls').boundingBox();
  const controls = await page.locator('.player-controls').boundingBox();
  assert.ok(Math.abs(transport.x + transport.width / 2 - controls.x - controls.width / 2) < 1, `video transport centered at ${width}`);
  assert.equal(await page.locator('#layerControls').isVisible(), true);
  assert.equal(await page.locator('.video-panel > .panel-toolbar > #layerControls').count(), 1, 'layer selectors belong to the match toolbar');
  assert.equal(await page.locator('#playerDisplay').isVisible(), true);
  assert.equal(await page.locator('#refereeDisplay').isVisible(), true);
  assert.equal(await page.locator('[data-filter="ball"]').isVisible(), true);
  const layerBar = await page.locator('#layerControls').boundingBox();
  const heading = await page.locator('.panel-toolbar .panel-heading').boundingBox();
  const expandButton = await page.locator('#focusVideo').boundingBox();
  if (layerBar.y >= heading.y + heading.height) {
    assert.ok(layerBar.y >= expandButton.y + expandButton.height, 'wrapped controls stay below the title and expand button');
  } else {
    assert.ok(layerBar.x >= heading.x + heading.width, 'layers are after the match title');
    assert.ok(layerBar.x + layerBar.width <= expandButton.x, 'layers are before expand');
  }
  const reset = await page.locator('#resetLayers').boundingBox();
  assert.ok(Math.abs(reset.y + reset.height / 2 - layerBar.y - layerBar.height / 2) < 1, 'reset is vertically centered beside the layer controls');
  assert.ok(await page.locator('#layerControls').evaluate(el => el.scrollWidth <= el.clientWidth), 'layer controls wrap without horizontal scrolling');
  const selectorWidth = await page.locator('#playerDisplay').evaluate(el => el.getBoundingClientRect().width);
  assert.ok(selectorWidth <= 110, 'player selection stays compact');

  assert.ok(layerBar.x >= 0 && layerBar.x + layerBar.width <= width, 'inline layers fit viewport');
  const legend = await page.locator('.timeline-legends').boundingBox();
  const seek = await page.locator('.seek-wrap').boundingBox();
  assert.ok(legend.y >= seek.y + seek.height, 'legend is below timeline');
  assert.ok(Math.abs(legend.x - seek.x) < 1, 'legend aligns left');
  assert.ok(controls.y >= legend.y + legend.height, 'transport is below legend');
  const before = await page.evaluate(() => ({time: video.currentTime, src: video.src, speed: video.playbackRate, filters: filters()}));
  const normalHeight = (await page.locator('#videoStage').boundingBox()).height;
  await page.locator('#focusVideo').click();
  const expanded = await page.locator('#videoStage').boundingBox();
  assert.ok(expanded.height > normalHeight, `expanded video is taller at ${width}`);
  assert.equal(await page.locator('.run-picker').isVisible(), false);
  assert.equal(await page.locator('.mode-bar').isVisible(), false);
  assert.equal(await page.locator('#layerControls').isVisible(), true);
  assert.equal(await page.locator('.panel-toolbar .panel-heading').isVisible(), true);
  assert.ok(await page.locator('.panel-toolbar').evaluate(el => el.getBoundingClientRect().height <= 240), 'expanded toolbar leaves room for the video when controls wrap');
  assert.equal(await page.locator('.insights').isVisible(), false);
  const expandedToolbar = await page.locator('.panel-toolbar').boundingBox();
  const expandedLayers = await page.locator('#layerControls').boundingBox();
  const expandedReset = await page.locator('#resetLayers').boundingBox();
  assert.ok(Math.abs(expandedReset.y + expandedReset.height / 2 - expandedLayers.y - expandedLayers.height / 2) < 1, 'expanded reset stays vertically centered');
  assert.ok(expandedLayers.y >= expandedToolbar.y && expandedLayers.y + expandedLayers.height <= expanded.y, 'expanded layer selectors stay above the footage');
  const originalDisplay = await page.locator('#playerDisplay').inputValue();
  await page.locator('#playerDisplay').selectOption('segmentation');
  assert.equal(await page.evaluate(() => filters().playerMode), 'segmentation', 'layers remain editable while expanded');
  await page.locator('#playerDisplay').selectOption(originalDisplay);
  const expandedControls = await page.locator('.player-controls').boundingBox();
  assert.ok(expandedControls.y + expandedControls.height <= height, 'expanded controls stay visible');
  assert.ok(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight), 'expanded playback has no vertical scroll');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'expanded playback has no horizontal scroll');
  assert.deepEqual(await page.evaluate(() => ({time: video.currentTime, src: video.src, speed: video.playbackRate, filters: filters()})), before, 'expanding preserves playback and layers');
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#focusVideo').getAttribute('aria-pressed'), 'false');
  assert.equal(await page.locator('#layerControls').isVisible(), true);
  assert.equal(await page.locator('#focusVideo').evaluate(el => el === document.activeElement), true, 'Escape restores expand button focus');

}
async function settleModes(page) {
  await page.evaluate(() => Promise.all(document.getAnimations().map(animation => animation.finished.catch(() => {}))));
}
async function checkAccessibility(page) {
  // Audit settled colors, rather than an intermediate frame of a theme transition.
  await page.evaluate(() =>
    Promise.all(
      document
        .getAnimations()
        .map((animation) => animation.finished.catch(() => {})),
    ),
  );
  await page.addScriptTag({ path: axePath });
  const violations = await page.evaluate(async () =>
    (
      await axe.run(document, {
        runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] },
      })
    ).violations.map((item) => ({
      id: item.id,
      nodes: item.nodes.map((node) => node.target),
    })),
  );
  assert.deepEqual(violations, [], "automated accessibility checks");
}
async function main() {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {}),
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1600, height: 1000 },
    });
    context.on("page", (page) =>
      page.on("pageerror", (error) => failures.push(error.message)),
    );
    const page = await openSample(context);
    for (const [width, height] of [
      [1920, 1080],
      [1600, 1000],
      [1440, 900],
      [1366, 768],
      [1280, 720],
      [1024, 768],
      [1024, 640],
      [820, 900],
      [390, 844],
    ]) {
      await assertLayout(page, width, height);
      console.log(`Layout ${width}×${height}: passed`);
    }
    await page.setViewportSize({ width: 1600, height: 1000 });
    assert.equal(await page.locator('#layersMenu, #fullscreen, #runMeta, #modeDescription, #videoFilename, #layerCount, #scoreboardLegend, #timelineMarkers .score, .video-footer').count(), 0);
    assert.equal(await page.locator('#layerControls').isVisible(), true);
    assert.equal(await page.locator('#layerControls svg').count(), 0);
    assert.deepEqual(await page.locator('.layer-toggle span').allTextContents(), ['Pose', 'Noms / numéros', 'Pose', 'Noms / numéros', 'Traînée', 'Repères détectés']);
    assert.equal(await page.locator('#resetLayers').innerText(), 'Réinitialiser');
    await page.locator("body").click({ position: { x: 1, y: 1 } });
    assert.equal(await page.locator(".topbar").count(), 0);
    assert.equal(
      await page.locator(".run-picker-heading .language-switch").count(),
      1,
    );
    assert.doesNotMatch(await page.locator("body").innerText(), /vyneros/i);
    assert.equal(await page.title(), 'Match Analyse');
    assert.equal(await page.locator('[data-i18n="session"], #openCourtHeatmaps, #timelineMarkers .cut').count(), 0);
    assert.equal(await page.locator('#interceptionsPanel').getAttribute('open'), null);
    assert.equal(await page.locator('#interceptionsPanel details, #interceptionsPanel summary, #interceptionsPanel button').count(), 0);
    assert.match(await page.locator('#interceptionsPanel').innerText(), /Arrive bientôt/);
    const placeholderHeight = (await page.locator('#interceptionsPanel').boundingBox()).height;
    await page.locator('#interceptionsPanel').click();
    assert.equal((await page.locator('#interceptionsPanel').boundingBox()).height, placeholderHeight);
    const displayTerms = ["Bbox", "Segmentation", "Bbox + Segmentation"];
    const terms = () =>
      page
        .locator('#playerDisplay option:not([value="hidden"])')
        .allTextContents();
    assert.deepEqual(
      (await terms()).map((value) => value.trim()),
      displayTerms,
    );
    await capture(page, "studio-fr");
    await checkAccessibility(page);
    await page.locator("#statisticsTab").click();
    assert.equal(await page.locator('#detailName').innerText(), 'Example A', 'first player is preselected on entry');
    assert.equal(await page.locator('.roster-player[aria-pressed=true]').count(), 1);
    await page.locator('.roster-player').nth(1).click();
    await page.locator('#visualizationTab').click();
    await page.locator('#statisticsTab').click();
    assert.equal(await page.locator('#detailName').innerText(), 'Example A2', 'a chosen player survives mode changes');
    await page.locator(".roster-player").first().click();
    assert.equal(await page.locator("#detailName").textContent(), "Example A");
    const original = await page.locator(".final-stat-value").allTextContents();
    const selection = await page.evaluate(() => state.selected.subject);
    const beforeTabs = await page.evaluate(() => video.currentTime);
    await page.getByRole("tab", { name: "Équipe", exact: true }).click();
    assert.equal(await page.locator("#teamToggle [aria-pressed=true] strong").textContent(), "Example Blue");
    assert.equal(await page.evaluate(() => state.selected.subject), selection);
    assert.equal(await page.evaluate(() => video.currentTime), beforeTabs);
    assert.equal(await page.locator("#playersPanel").isVisible(), false);
    assert.equal(
      await page.locator("#teamPanel .final-stat-value").count(),
      0,
      "team metrics are not fabricated from player statistics",
    );
    await capture(page, "team-fr");
    await checkAccessibility(page);
    await page.getByRole("tab", { name: "Équipe", exact: true }).focus();
    await page.keyboard.press("Home");
    assert.equal(
      await page.locator("#playersTab").getAttribute("aria-selected"),
      "true",
    );
    assert.equal(await page.locator("#detailName").textContent(), "Example A");
    await page.keyboard.press("ArrowRight");
    assert.equal(
      await page.locator("#teamTab").getAttribute("aria-selected"),
      "true",
    );
    await page.locator('[data-language="en"]').click();
    assert.equal(
      await page.locator("#teamTab").getAttribute("aria-selected"),
      "true",
    );
    await page.waitForFunction(() => document.querySelector('#team-heatmapCharts canvas'));
    assert.equal(await page.locator('#team-heatmapSubject').inputValue(), 'team:blue');
    assert.deepEqual(
      (await terms()).map((value) => value.trim()),
      displayTerms,
    );
    await page.locator("#teamToggle button").last().click();
    assert.equal(await page.locator("#teamToggle [aria-pressed=true] strong").textContent(), "Example Pink");
    assert.equal(
      await page
        .locator('#teamToggle button[aria-pressed="true"]')
        .evaluate((el) => el === document.activeElement),
      true,
      "team changes preserve keyboard focus",
    );
    assert.equal(await page.evaluate(() => state.selected), null);
    await page.getByRole("tab", { name: "Players", exact: true }).click();
    assert.equal(
      await page.locator("#rosterTeamHeading h3").textContent(),
      "Example Pink",
    );
    assert.equal(await page.locator('#detailName').innerText(), 'Example B', 'entering a new team selects its first player');
    await page.getByRole("tab", { name: "Team", exact: true }).click();
    await page.evaluate(() => selectSubject("roster:blue:7"));
    assert.equal(
      await page.locator("#playersTab").getAttribute("aria-selected"),
      "true",
      "selecting a player from the video opens the player tab",
    );
    assert.equal(await page.locator("#detailName").textContent(), "Example A");
    assert.equal(await page.evaluate(() => video.currentTime), beforeTabs);
    await page.locator('[data-language="fr"]').click();
    console.log(
      "Team/player navigation, shared team context and invariant detection terms: passed",
    );
    await page.locator("#visualizationTab").click();
    await page.locator("#playerDisplay").selectOption("segmentation");
    await page.locator('[data-filter="playerPose"]').check();
    await page.keyboard.press("Escape");
    await page.evaluate(() => seekToFrame(40));
    await page.waitForFunction(() => state.current === 40);
    const time = await page.evaluate(() => video.currentTime);
    await page.locator('[data-language="en"]').click();
    assert.equal(await page.locator("html").getAttribute("lang"), "en");
    assert.equal(await page.locator("#detailName").textContent(), "Example A");
    assert.equal(await page.evaluate(() => state.selected.subject), selection);
    assert.equal(await page.evaluate(() => video.currentTime), time);
    assert.equal(
      await page.locator("#playerDisplay").inputValue(),
      "segmentation",
    );
    assert.equal(
      await page.locator('[data-filter="playerPose"]').isChecked(),
      true,
    );
    assert.equal(
      await page.locator(".final-stat-label").first().textContent(),
      "Observed distance",
    );
    assert.equal(
      (await page.locator(".final-stat-value").allTextContents())[0],
      original[0].replace(",", "."),
    );
    await page.locator("#statisticsTab").click();
    await settleModes(page);
    await capture(page, "player-en");
    await checkAccessibility(page);
    await page.locator("#visualizationTab").click();
    await page.locator("#playButton").click();
    await page.waitForFunction(() => !video.paused && video.currentTime > 1.5);
    const playingAt = await page.evaluate(() => video.currentTime);
    await page.locator('#focusVideo').click();
    await page.waitForFunction(time => !video.paused && video.currentTime > time, playingAt);
    await page.locator('#focusVideo').click();
    await page.locator('[data-language="fr"]').click();
    assert.equal(
      await page.evaluate(() => video.paused),
      false,
      "language change preserves playback",
    );
    await page.locator("#playButton").click();
    await page.locator("#statisticsTab").click();
    await page.locator("#rosterBack").click();
    await page.locator("#teamToggle button").last().click();
    assert.equal(
      await page.locator("#rosterTeamHeading h3").textContent(),
      "Example Pink",
    );
    await page.reload();
    await waitForFrame(page);
    assert.equal(
      await page.locator("html").getAttribute("lang"),
      "fr",
      "language persists after reload",
    );
    await page.locator('[data-language="en"]').click();
    await page.reload();
    await waitForFrame(page);
    assert.equal(await page.locator("html").getAttribute("lang"), "en");
    assert.equal(await page.locator('#identityMode, #refereeHint').count(), 0, 'people visibility selection removed');
    await page.locator('#playerDisplay').selectOption('segmentation');
    await page.locator('#refereeDisplay').selectOption('hidden');
    assert.equal(await page.evaluate(() => filters().referees), false);
    await page.locator('#resetLayers').click();
    assert.equal(await page.locator('#playerDisplay').inputValue(), 'bbox');
    assert.equal(await page.locator('#refereeDisplay').inputValue(), 'bbox');
    assert.equal(await page.evaluate(() => filters().referees), true);
    await page.locator('[data-filter="courtLandmarks"]').check();
    await page.locator('[data-filter="court"]').uncheck();
    await page.locator('[data-filter="courtLandmarks"]').uncheck();
    await page.locator("#resetLayers").click();
    assert.equal(await page.locator('[data-filter="court"]').isChecked(), true);
    assert.equal(await page.locator('[data-filter="courtLandmarks"]').isChecked(), false);
    const courtCanvas = await page.evaluate(() => {
      const row={...state.frame,court:{...state.frame.court,status:'fit'}};
      const pixels = () => Array.from(ctx.getImageData(0,0,canvas.width,canvas.height).data);
      ctx.clearRect(0,0,canvas.width,canvas.height);
      drawCourtOverlay(row,{court:true});const direct=pixels();
      ctx.clearRect(0,0,canvas.width,canvas.height);
      drawCourtOverlay({...row,court:{...row.court,status:'propagated',keypoints:null}},{court:true});const tracked=pixels();
      ctx.clearRect(0,0,canvas.width,canvas.height);
      drawCourtOverlay({...row,court:{...row.court,status:'unavailable'}},{court:true});const unavailable=pixels();
      drawOverlay(state.frame);
      return {drawn:direct.some(value=>value!==0),stable:direct.every((value,i)=>value===tracked[i]),cleared:unavailable.every(value=>value===0)};
    });
    assert.deepEqual(courtCanvas,{drawn:true,stable:true,cleared:true},"projected court stays visible without flashing and clears when unavailable");
    await capture(page, "layers-en");
    await checkAccessibility(page);
    await page.locator(".page-heading").click();
    await page.locator("#focusVideo").click();
    assert.equal(await page.locator(".insights").isVisible(), false);
    await page.evaluate(() => setLanguage('fr'));
    assert.equal(await page.locator('#focusLabel').textContent(), 'Réduire');
    await checkAccessibility(page);
    await page.locator("#focusVideo").click();
    assert.equal(await page.locator(".insights").isVisible(), true);
    await page.locator("body").click({ position: { x: 1, y: 1 } });
    await page.keyboard.press("Shift+ArrowRight");
    await waitForFrame(page);
    assert.equal(await page.evaluate(() => video.paused), true);
    console.log(
      "Language, player selection, playback, layers, focus and frame stepping: passed",
    );

    await page.evaluate(() => seekToFrame(20));
    await waitForFrame(page);
    await page.locator("#playButton").click();
    await page.waitForFunction(() => !video.paused);
    await page.locator("#statisticsTab").click();
    const pausedAt = await page.evaluate(() => video.currentTime);
    assert.equal(await page.evaluate(() => video.paused), true, "statistics pause playback");
    await settleModes(page);
    const remembered = await page.evaluate(() => ({selected: state.selected, team: state.teamId, view: state.statsView, src: video.src}));
    await page.locator("#statisticsTab").focus();
    await page.keyboard.press("Home");
    await settleModes(page);
    assert.equal(await page.locator("#visualizationTab").getAttribute("aria-selected"), "true");
    assert.equal(await page.evaluate(() => video.currentTime), pausedAt);
    assert.equal(await page.evaluate(() => video.paused), true, "returning never starts playback automatically");
    await page.keyboard.press("End");
    await settleModes(page);
    assert.equal(await page.locator("#statisticsTab").getAttribute("aria-selected"), "true");
    assert.deepEqual(await page.evaluate(() => ({selected: state.selected, team: state.teamId, view: state.statsView, src: video.src})), remembered);
    await page.locator('[data-language="en"]').click();
    assert.equal(await page.locator('#modeDescription').count(), 0);
    await checkAccessibility(page);
    await page.locator("#visualizationTab").click();
    await settleModes(page);
    // Both screens move laterally during the transition; outgoing controls are inert immediately.
    const transition = await page.evaluate(async () => {
      setWorkspaceMode('statistics');
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
      const panes = ['workspace', 'statisticsWorkspace'].map(id => {
        const pane = document.getElementById(id);
        return {x: pane.getBoundingClientRect().x, inert: pane.inert};
      });
      return panes;
    });
    assert.ok(transition[0].x < 0 && transition[1].x > 0, "both modes slide horizontally");
    assert.equal(transition[0].inert, true);
    await settleModes(page);
    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.locator("#visualizationTab").click();
    assert.equal(await page.locator("#workspace").evaluate(el => getComputedStyle(el).transitionDuration), "0s");
    assert.equal(await page.locator("#statisticsWorkspace").isVisible(), false);
    await page.locator('#focusVideo').click();
    await page.evaluate(() => selectSubject('roster:blue:7'));
    assert.equal(await page.locator('#appShell').evaluate(el => el.classList.contains('focus-video')), false, 'opening a player exits expanded playback');
    assert.equal(await page.locator('.mode-bar').isVisible(), true);
    assert.equal(await page.evaluate(() => state.mode), 'statistics', "selecting from video opens statistics");
    assert.equal(await page.locator("#detailName").isVisible(), true);
    await page.emulateMedia({reducedMotion: 'no-preference'});
    console.log("Workspace modes: lateral motion, keyboard, pause, state, language and reduced motion passed");

    // Long real-world rosters must scroll at laptop sizes instead of shrinking text.
    const longRosterPage = await context.newPage();
    await longRosterPage.route("**/api/run?*", async (route) => {
      const response = await route.fetch();
      const meta = await response.json();
      meta.roster.teams[0].players = Array.from({ length: 35 }, (_, i) => ({
        number: String(i),
        name: `Player with a long name ${i}`,
      }));
      await route.fulfill({ response, json: meta });
    });
    await longRosterPage.goto(`${baseURL}/?run=synthetic`);
    await waitForFrame(longRosterPage);
    await longRosterPage.setViewportSize({ width: 1280, height: 720 });
    await longRosterPage.locator("#statisticsTab").click();
    assert.equal(
      await longRosterPage
        .locator("#rosterList")
        .evaluate((el) => el.scrollHeight > el.clientHeight),
      true,
    );
    await longRosterPage.locator(".roster-player").last().click();
    assert.equal(
      await longRosterPage.locator("#detailName").textContent(),
      "Player with a long name 34",
    );
    assert.ok(
      (
        await longRosterPage.locator(".final-stat-value").allTextContents()
      ).every((value) => value === "—"),
    );
    console.log("Long rosters and missing player statistics: passed");

    const empty = await context.newPage();
    await empty.route("**/api/runs", (route) =>
      route.fulfill({ json: { runs: [] } }),
    );
    await empty.goto(baseURL);
    await empty.waitForFunction(
      () => document.querySelector("#runSelect").disabled,
    );
    await empty.locator('[data-language="en"]').click();
    await empty.locator("#statisticsTab").click();
    await empty.getByRole("tab", { name: "Team", exact: true }).click();
    assert.match(await empty.locator("#team-heatmapStatus").textContent(), /analysis/);
    assert.equal(await empty.locator("#teamToggle button").count(), 0);
    await checkAccessibility(empty);
    assert.equal(
      await empty.locator("#videoEmpty strong").textContent(),
      "No analyses available",
    );
    await empty.locator('[data-language="fr"]').click();
    assert.equal(
      await empty.locator("#videoEmpty strong").textContent(),
      "Aucune analyse disponible",
    );

    const unavailable = await context.newPage();
    await unavailable.route("**/api/run?*", async (route) => {
      const response = await route.fetch();
      const meta = await response.json();
      meta.video_available = false;
      meta.reference_roster = meta.roster;
      meta.roster = null;
      await route.fulfill({ response, json: meta });
    });
    await unavailable.goto(`${baseURL}/?run=synthetic`);
    await unavailable.waitForFunction(() => state.loaded);
    await unavailable.locator('[data-language="en"]').click();
    assert.equal(
      await unavailable.locator("#videoEmpty strong").textContent(),
      "Source video unavailable",
    );
    await unavailable.locator("#statisticsTab").click();
    await unavailable.locator(".roster-player").first().click();
    assert.ok(
      (await unavailable.locator(".final-stat-value").allTextContents()).every(
        (value) => value === "—",
      ),
    );
    assert.match(
      await unavailable.locator("#finalStatsNote").textContent(),
      /verified identification/,
    );
    console.log("Empty state, unavailable video and reference roster: passed");

    const fallback = await browser.newContext();
    fallback.on("page", (page) =>
      page.on("pageerror", (error) => failures.push(error.message)),
    );
    await fallback.addInitScript(() => {
      delete HTMLVideoElement.prototype.requestVideoFrameCallback;
    });
    const fallbackPage = await openSample(fallback);
    await fallbackPage.evaluate(() => seekToFrame(50));
    await fallbackPage.waitForFunction(() => state.current === 50);
    await fallbackPage.locator('[data-language="en"]').click();
    assert.equal(await fallbackPage.evaluate(() => state.current), 50);
    console.log("Seeking without video presentation callbacks: passed");
    assert.deepEqual(failures, [], "no browser errors");
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
