// Browser checks use only the public, scripted sample, never local match footage.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright");
const axePath = require.resolve("axe-core/axe.min.js");
const base = process.env.VIEWER_URL || "http://127.0.0.1:8766";
const output = process.env.VIEWER_SCREENSHOTS;
const errors = [];
const modes = {
  presence: 1,
  zones: 1,
  comparison: 3,
};

async function settled(page, count) {
  await page.waitForFunction(
    (count) =>
      document.querySelector("#heatmapStatus").hidden &&
      document.querySelectorAll("#heatmapDialog .hm-map-card").length === count,
    count,
  );
}
async function calculate(page, action, mode) {
  const response = page.waitForResponse(async (response) => {
    const url = new URL(response.url());
    if (
      url.pathname !== "/api/heatmaps" ||
      url.searchParams.get("mode") !== mode
    )
      return false;
    const currentPhase = await page.locator('#heatmapPhaseSummary [aria-pressed="true"]').getAttribute('data-heatmap-phase');
    const expectedSubject = await page.evaluate(
      (mode) =>
        mode === "comparison"
          ? [...document.querySelectorAll("#heatmapSubject option")].find(
              (option) => option.value.startsWith("team:"),
            )?.value
          : document.querySelector("#heatmapSubject").value,
      mode,
    );
    if (url.searchParams.get("subject") !== expectedSubject) return false;
    return url.searchParams.get('phase') === currentPhase && !url.searchParams.has('episode');
  });
  await action();
  const data = await (await response).json();
  assert.equal(data.mode, mode);
  await settled(page, data.scope.phase === "unknown" ? 0 : modes[mode]);
  for (const map of data.maps) {
    assert.ok(Math.abs(map.grid.reduce((a, b) => a + b, 0) - map.total) < 1e-8);
    assert.ok(
      Math.abs(map.zones.reduce((a, b) => a + b, 0) - map.total) < 1e-8,
    );
  }
  return data;
}
async function accessibility(page) {
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
  assert.deepEqual(violations, [], "heatmap accessibility");
}
async function openSpatial(page) {
  await page.locator('#statisticsTab').click();
  await page.locator('#playersTab').click();
  await page.locator('#openPlayerHeatmaps').click();
}
async function open(context) {
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${base}/?run=synthetic`);
  await page.waitForFunction(
    () => state.loaded && state.frame && video.readyState >= 2,
  );
  await page.evaluate(() => document.fonts.ready);
  return page;
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
    const page = await open(context);
    assert.deepEqual(
      await page
        .locator("#heatmapDialog [data-heatmap-mode]")
        .evaluateAll((buttons) =>
          buttons.map((button) => button.dataset.heatmapMode),
        ),
      ["presence", "zones", "comparison"],
    );
    await calculate(
      page,
      () => openSpatial(page),
      "presence",
    );
    assert.equal(await page.evaluate(() => video.paused), true);
    const time = await page.evaluate(() => video.currentTime);
    for (const subject of ["team:blue", "roster:blue:7"]) {
      await calculate(
        page,
        () => page.locator("#heatmapSubject").selectOption(subject),
        "presence",
      );
      for (const phase of ["offense", "defense"]) {
        await calculate(
          page,
          () => page.locator(`#heatmapPhaseSummary [data-heatmap-phase="${phase}"]`).click(),
          "presence",
        );
        for (const [mode, count] of Object.entries(modes)) {
          const data = await calculate(
            page,
            () => page.locator(`#heatmapDialog [data-heatmap-mode="${mode}"]`).click(),
            mode,
          );
          assert.equal(await page.locator("#heatmapDialog .hm-map-card").count(), count);
          assert.ok(
            data.maps.every(
              (map) =>
                map.subject.kind ===
                (mode === "comparison" || subject.startsWith("team:")
                  ? "team"
                  : "player"),
            ),
          );
          assert.equal(data.court.extent, "half");
          assert.equal(data.court.length, data.court.full_length / 2);
          assert.equal(await page.locator("#heatmapCompare").count(), 0);
          assert.equal(
            await page.locator("#heatmapSubject").isVisible(),
            mode !== "comparison",
          );
          if (mode === "comparison") {
            assert.deepEqual(
              data.maps.map((map) => map.subject.id),
              ["team:blue", "team:pink"],
            );
            assert.equal(await page.locator('#heatmapPhaseSummary button').count(), 2, 'comparison keeps phase controls');
            assert.equal(
              await page.locator("#heatmapMatchup").isVisible(),
              true,
            );
          }
          assert.equal(
            await page
              .locator("#heatmapDialog .hm-canvas")
              .first()
              .getAttribute("data-rendering"),
            mode === "zones" ? "cells" : "contours",
          );
          await accessibility(page);
          if (output) {
            await fs.mkdir(output, { recursive: true });
            await page.screenshot({
              path: path.join(
                output,
                `heatmap-${subject.startsWith("team:") ? "team" : "player"}-${phase}-${mode}.png`,
              ),
              fullPage: true,
            });
          }
        }
        await calculate(
          page,
          () => page.locator('#heatmapDialog [data-heatmap-mode="presence"]').click(),
          "presence",
        );
      }
      await calculate(
        page,
        () => page.locator('#heatmapPhaseSummary [data-heatmap-phase="offense"]').click(),
        "presence",
      );
    }
    assert.equal(
      await page.evaluate(() => video.currentTime),
      time,
      "analysis controls preserve playback position",
    );
    const beforeRendering = {
      pixels: await page
        .locator("#heatmapDialog .hm-canvas")
        .first()
        .evaluate((canvas) => canvas.toDataURL()),
    };
    let extraRequests = 0;
    const countRequest = (request) => {
      if (request.url().includes("/api/heatmaps")) extraRequests++;
    };
    page.on("request", countRequest);
    assert.match(
      await page.locator(".hm-map-legend").first().innerText(),
      /m²/,
    );
    await page.locator("#heatmapSmooth").uncheck();
    assert.equal(
      await page.locator("#heatmapDialog .hm-canvas").first().getAttribute("data-rendering"),
      "cells",
    );
    assert.doesNotMatch(
      await page.locator(".hm-map-legend").first().innerText(),
      /m²/,
    );
    assert.notEqual(
      await page
        .locator("#heatmapDialog .hm-canvas")
        .first()
        .evaluate((canvas) => canvas.toDataURL()),
      beforeRendering.pixels,
    );
    await page.locator("#heatmapSmooth").check();
    assert.equal(
      await page.locator("#heatmapDialog .hm-canvas").first().getAttribute("data-rendering"),
      "contours",
    );
    assert.equal(
      extraRequests,
      0,
      "rendering does not refetch or change measurements",
    );
    page.off("request", countRequest);
    assert.equal(await page.locator('#heatmapPhase, #heatmapPassage, #heatmapStart, #heatmapEnd').count(), 0, 'removed temporal controls are absent');
    assert.equal(await page.locator('.hm-map-stats, .hm-inspect, .hm-data, .hm-limit').count(), 0, 'removed data sections are absent');
    await calculate(page, () => page.locator('#heatmapDialog [data-heatmap-mode="comparison"]').click(), 'comparison');
    const quarterComparison = await calculate(
      page,
      () => page.locator("#heatmapPeriod").selectOption("1"),
      "comparison",
    );
    assert.equal(quarterComparison.scope.start_s, 3);
    assert.ok(quarterComparison.maps.every((map) => map.start_s === 3));
    await calculate(
      page,
      () => page.locator('#heatmapDialog [data-heatmap-mode="presence"]').click(),
      "presence",
    );
    const settings = await page.locator("#heatmapPeriod").inputValue();
    await page.locator("#heatmapLanguage").click();
    assert.equal(
      await page.locator("#heatmapModeTitle").innerText(),
      "Presence",
    );
    assert.equal(await page.locator("#heatmapPeriod").inputValue(), settings);
    assert.equal(
      await page.locator("#heatmapSubject").inputValue(),
      "roster:blue:7",
    );
    await page.locator("#heatmapLanguage").click();
    assert.equal(await page.locator('.hm-export').count(), 0);
    const canvasHeight = await page.locator('#heatmapDialog .hm-canvas').first().evaluate(canvas => canvas.height);
    assert.ok(canvasHeight > 1100, 'maps retain NBA half-court proportions');
    for (const [width, height] of [
      [1920, 1080],
      [1600, 1000],
      [1366, 768],
      [1024, 768],
      [820, 900],
      [390, 844],
    ]) {
      await page.setViewportSize({ width, height });
      const bounds = await page.locator("#heatmapDialog").boundingBox();
      assert.ok(
        bounds.x >= 0 &&
          bounds.y >= 0 &&
          bounds.x + bounds.width <= width + 1 &&
          bounds.y + bounds.height <= height + 1,
        `modal fits ${width}×${height}`,
      );
      assert.equal(
        await page
          .locator("#heatmapDialog")
          .evaluate((el) => el.scrollWidth <= el.clientWidth),
        true,
        "no modal horizontal overflow",
      );
      await calculate(
        page,
        () => page.locator('#heatmapDialog [data-heatmap-mode="comparison"]').click(),
        "comparison",
      );
      assert.equal(
        await page
          .locator("#heatmapCharts")
          .evaluate((el) => el.scrollWidth <= el.clientWidth),
        true,
        `comparison fits ${width}×${height}`,
      );
      await calculate(
        page,
        () => page.locator('#heatmapDialog [data-heatmap-mode="presence"]').click(),
        "presence",
      );
    }
    await page.setViewportSize({ width: 1600, height: 1000 });
    await page.keyboard.press("Escape");
    assert.equal(
      await page
        .locator("#openPlayerHeatmaps")
        .evaluate((el) => el === document.activeElement),
      true,
      "Escape restores opener focus",
    );
    await page.locator("#statisticsTab").click();
    await page.locator(".roster-player").first().click();
    await calculate(
      page,
      () => page.locator("#openPlayerHeatmaps").click(),
      "presence",
    );
    assert.equal(
      await page.locator("#heatmapSubject").inputValue(),
      await page.evaluate(() => state.selected.subject),
    );
    await page.locator("#closeHeatmaps").click();
    await page.locator("#teamTab").click();
    await page.waitForFunction(() => document.querySelector('#team-heatmapCharts canvas'));
    assert.equal(await page.locator('#heatmapDialog').evaluate(dialog => dialog.open), false, 'team heatmaps appear inline');
    assert.equal(await page.locator('#team-heatmapSubject').inputValue(), await page.evaluate(() => `team:${state.teamId}`));
    assert.equal(await page.locator('#teamHeatmaps [data-heatmap-phase]').count(), 2);
    await page.close();

    // A missing B must never produce an apparent spatial advantage for A.
    const missing = await open(context);
    await missing.route("**/api/heatmaps?**", async (route) => {
      const response = await route.fetch();
      const data = await response.json();
      if (data.mode === "comparison") {
        const map = data.maps[1];
        map.grid.fill(0);
        map.zones.fill(0);
        map.total =
          map.localized_s =
          map.measured_s =
          map.localized_video_s =
            0;
        map.coverage = null;
      }
      await route.fulfill({ response, json: data });
    });
    await calculate(
      missing,
      () => openSpatial(missing),
      "presence",
    );
    await calculate(
      missing,
      () => missing.locator('#heatmapDialog [data-heatmap-mode="comparison"]').click(),
      "comparison",
    );
    assert.equal(
      await missing.locator("#heatmapDialog .hm-map-card").nth(2).locator(".hm-empty").count(),
      1,
    );
    const pixel = await missing
      .locator("#heatmapDialog .hm-canvas")
      .nth(2)
      .evaluate((canvas) =>
        [...canvas.getContext("2d").getImageData(204, 345, 1, 1).data].slice(
          0,
          3,
        ),
      );
    assert.deepEqual(
      pixel,
      [23, 28, 32],
      "unavailable comparison remains neutral",
    );
    await accessibility(missing);
    await missing.close();

    // A late API response must not replace the user's newer choice.
    const race = await open(context);
    await calculate(
      race,
      () => openSpatial(race),
      "presence",
    );
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    await race.route("**/api/heatmaps?**", async (route) => {
      if (
        new URL(route.request().url()).searchParams.get("mode") === "presence"
      )
        await gate;
      await route.continue();
    });
    await race.locator('#heatmapDialog [data-heatmap-mode="presence"]').click();
    await calculate(
      race,
      () => race.locator('#heatmapDialog [data-heatmap-mode="zones"]').click(),
      "zones",
    );
    const late = race.waitForResponse(
      (response) =>
        new URL(response.url()).searchParams.get("mode") === "presence",
    );
    release();
    await (await late).finished();
    await race.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
    assert.equal(await race.locator("#heatmapModeTitle").innerText(), "Zones");
    await race.close();

    const unavailable = await open(context);
    await unavailable.route("**/api/heatmaps?**", (route) =>
      route.fulfill({ json: { available: false, reason: "geometry" } }),
    );
    await openSpatial(unavailable);
    await unavailable.waitForFunction(
      () => document.querySelector("#heatmapSubject").disabled,
    );
    assert.equal(await unavailable.locator("#heatmapDialog .hm-map-card").count(), 0);
    assert.equal(await unavailable.locator("#heatmapStatus").isVisible(), true);
    await accessibility(unavailable);
    await unavailable.close();
    assert.deepEqual(errors, [], "no browser exceptions");
    console.log(
      "Heatmaps: two player/team views + automatic team comparison, half courts, offense/defense, quarter ranges, phase buttons, hidden neutral intervals, languages, missing data, response races, responsive layout and accessibility passed",
    );
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
