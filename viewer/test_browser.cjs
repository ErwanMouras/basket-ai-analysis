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
    assert.ok(
      layout.roster.y >= layout.court.bottom,
      "roster stays below court",
    );
    assert.ok(layout.video.height > 300, "video retains useful viewing space");
  }
  const summary = page.locator("#layersMenu summary");
  await summary.click();
  const popover = await page.locator(".layers-popover").boundingBox();
  assert.ok(
    popover.x >= 0 && popover.x + popover.width <= width,
    "layer controls stay within viewport",
  );
  await page.keyboard.press("Escape");
  assert.equal(await page.locator("#layersMenu").getAttribute("open"), null);
  assert.equal(
    await summary.evaluate((element) => element === document.activeElement),
    true,
    "Escape restores focus",
  );
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
      [820, 900],
      [390, 844],
    ]) {
      await assertLayout(page, width, height);
      console.log(`Layout ${width}×${height}: passed`);
    }
    await page.setViewportSize({ width: 1600, height: 1000 });
    const summary = page.locator("#layersMenu summary");
    await summary.focus();
    await page.keyboard.press("Space");
    assert.equal(
      await page.locator("#layersMenu").evaluate((el) => el.open),
      true,
      "Space opens the native disclosure",
    );
    assert.equal(
      await page.evaluate(() => video.paused),
      true,
      "disclosure keyboard activation does not play the video",
    );
    await page.keyboard.press("Escape");
    await page.locator("body").click({ position: { x: 1, y: 1 } });
    assert.equal(await page.locator(".topbar").count(), 0);
    assert.equal(
      await page.locator(".run-picker-heading .language-switch").count(),
      1,
    );
    assert.doesNotMatch(await page.locator("body").innerText(), /vyneros/i);
    assert.doesNotMatch(await page.title(), /vyneros/i);
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
    await page.locator(".roster-player").first().click();
    assert.equal(await page.locator("#detailName").textContent(), "Example A");
    const original = await page.locator(".final-stat-value").allTextContents();
    const selection = await page.evaluate(() => state.selected.subject);
    const beforeTabs = await page.evaluate(() => video.currentTime);
    await page.getByRole("tab", { name: "Équipe", exact: true }).click();
    assert.equal(await page.locator("#teamName").textContent(), "Example Blue");
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
    assert.match(
      await page.locator("#teamStatsNote").textContent(),
      /presence/,
    );
    assert.deepEqual(
      (await terms()).map((value) => value.trim()),
      displayTerms,
    );
    await page.locator("#teamToggle button").last().click();
    assert.equal(await page.locator("#teamName").textContent(), "Example Pink");
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
    await page.locator("#layersMenu summary").click();
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
    await capture(page, "player-en");
    await checkAccessibility(page);
    await page.locator("#playButton").click();
    await page.waitForFunction(() => !video.paused && video.currentTime > 1.5);
    await page.locator('[data-language="fr"]').click();
    assert.equal(
      await page.evaluate(() => video.paused),
      false,
      "language change preserves playback",
    );
    await page.locator("#playButton").click();
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
    await page.locator("#layersMenu summary").click();
    await page.locator("#identityMode").selectOption("identified");
    assert.equal(await page.locator("#refereeDisplay").isDisabled(), true);
    assert.equal(await page.locator("#refereeHint").isVisible(), true);
    assert.equal(await page.locator("#layerCount").textContent(), "3");
    await page.locator("#resetLayers").click();
    assert.equal(await page.locator("#identityMode").inputValue(), "all");
    assert.equal(await page.locator("#refereeDisplay").isDisabled(), false);
    assert.equal(await page.locator("#playerDisplay").inputValue(), "bbox");
    assert.equal(await page.locator("#layerCount").textContent(), "4");
    await page.locator('[data-filter="courtLandmarks"]').check();
    await page.locator('[data-filter="court"]').uncheck();
    assert.equal(await page.locator("#layerCount").textContent(), "4", "landmarks alone count as one court category");
    await page.locator('[data-filter="courtLandmarks"]').uncheck();
    assert.equal(await page.locator("#layerCount").textContent(), "3");
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
    assert.equal(
      await page.locator("#layersMenu").getAttribute("open"),
      null,
      "outside click dismisses layers",
    );
    await page.locator("#focusVideo").click();
    assert.equal(await page.locator(".insights").isVisible(), false);
    await page.locator('[data-language="fr"]').click();
    assert.equal(
      await page.locator("#focusLabel").textContent(),
      "Afficher les panneaux",
    );
    await page.locator("#focusVideo").click();
    assert.equal(await page.locator(".insights").isVisible(), true);
    await page.locator("body").click({ position: { x: 1, y: 1 } });
    await page.keyboard.press("Shift+ArrowRight");
    await waitForFrame(page);
    assert.equal(await page.evaluate(() => video.paused), true);
    console.log(
      "Language, player selection, playback, layers, focus and frame stepping: passed",
    );

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
    await empty.getByRole("tab", { name: "Team", exact: true }).click();
    assert.match(await empty.locator("#teamStatsNote").textContent(), /roster/);
    assert.equal(await empty.locator("#teamHero").isVisible(), false);
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
