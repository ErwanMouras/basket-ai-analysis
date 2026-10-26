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
  movement: 1,
  zones: 1,
  comparison: 3,
  evolution: 2,
};

async function settled(page, count) {
  await page.waitForFunction(
    (count) =>
      document.querySelector("#heatmapStatus").hidden &&
      document.querySelectorAll(".hm-map-card").length === count,
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
    const current = await page.evaluate(() =>
      Object.fromEntries(
        [
          ["subject", "heatmapSubject"],
          ["plan", "heatmapPlan"],
          ["start", "heatmapStart"],
          ["end", "heatmapEnd"],
        ].map(([key, id]) => [key, document.getElementById(id).value]),
      ),
    );
    return Object.entries(current).every(([key, value]) =>
      key === "start" || key === "end"
        ? Math.abs(Number(value) - Number(url.searchParams.get(key))) < 1e-6
        : value === url.searchParams.get(key),
    );
  });
  await action();
  const data = await (await response).json();
  assert.equal(data.mode, mode);
  await settled(page, modes[mode]);
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
    await calculate(
      page,
      () => page.locator("#openCourtHeatmaps").click(),
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
      for (const [mode, count] of Object.entries(modes)) {
        const data = await calculate(
          page,
          () => page.locator(`[data-heatmap-mode="${mode}"]`).click(),
          mode,
        );
        assert.equal(await page.locator(".hm-map-card").count(), count);
        assert.ok(
          data.maps.every(
            (map) =>
              map.subject.kind ===
              (subject.startsWith("team:") ? "team" : "player"),
          ),
        );
        assert.equal(
          await page.locator("#heatmapTables tbody tr").count(),
          count * 9,
        );
        await accessibility(page);
        if (output) {
          await fs.mkdir(output, { recursive: true });
          await page.screenshot({
            path: path.join(
              output,
              `heatmap-${subject.startsWith("team:") ? "team" : "player"}-${mode}.png`,
            ),
            fullPage: true,
          });
        }
      }
      await calculate(
        page,
        () => page.locator('[data-heatmap-mode="presence"]').click(),
        "presence",
      );
    }
    assert.equal(
      await page.evaluate(() => video.currentTime),
      time,
      "analysis controls preserve playback position",
    );
    await page.locator("#heatmapStart").fill("0.5");
    await calculate(
      page,
      () => page.locator("#heatmapStart").dispatchEvent("change"),
      "presence",
    );
    await page.locator("#heatmapEnd").fill("1.5");
    const clipped = await calculate(
      page,
      () => page.locator("#heatmapEnd").dispatchEvent("change"),
      "presence",
    );
    assert.ok(
      Math.abs(clipped.maps[0].total - 1) < 1e-8,
      "time range clips actual time, not frame counts",
    );
    await page.locator("#heatmapEnd").fill("0.4");
    await page.locator("#heatmapEnd").dispatchEvent("change");
    assert.equal(
      await page.locator(".hm-map-card").count(),
      0,
      "bad range clears old charts",
    );
    assert.equal(await page.locator("#heatmapStatus").isVisible(), true);
    const second = await calculate(
      page,
      () => page.locator("#heatmapPlan").selectOption("1"),
      "presence",
    );
    assert.ok(
      second.maps[0].start_s >= 3,
      "changing plan resets range inside that plan",
    );
    const settings = await page.locator("#heatmapStart").inputValue();
    await page.locator("#heatmapLanguage").click();
    assert.equal(
      await page.locator("#heatmapModeTitle").innerText(),
      "Presence",
    );
    assert.equal(await page.locator("#heatmapStart").inputValue(), settings);
    assert.equal(
      await page.locator("#heatmapSubject").inputValue(),
      "roster:blue:7",
    );
    await page.locator("#heatmapLanguage").click();
    const download = page.waitForEvent("download");
    await page.locator(".hm-export").first().click();
    assert.match(
      (await download).suggestedFilename(),
      /^heatmap-presence-.*\.png$/,
    );
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
    }
    await page.setViewportSize({ width: 1600, height: 1000 });
    await page.keyboard.press("Escape");
    assert.equal(
      await page
        .locator("#openCourtHeatmaps")
        .evaluate((el) => el === document.activeElement),
      true,
      "Escape restores opener focus",
    );
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
    await calculate(
      page,
      () => page.locator("#openTeamHeatmaps").click(),
      "presence",
    );
    assert.equal(
      await page.locator("#heatmapSubject").inputValue(),
      await page.evaluate(() => `team:${state.teamId}`),
    );
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
      () => missing.locator("#openCourtHeatmaps").click(),
      "presence",
    );
    await calculate(
      missing,
      () => missing.locator('[data-heatmap-mode="comparison"]').click(),
      "comparison",
    );
    assert.equal(
      await missing.locator(".hm-map-card").nth(2).locator(".hm-empty").count(),
      1,
    );
    const pixel = await missing
      .locator(".hm-canvas")
      .nth(2)
      .evaluate((canvas) =>
        [...canvas.getContext("2d").getImageData(204, 345, 1, 1).data].slice(
          0,
          3,
        ),
      );
    assert.deepEqual(
      pixel,
      [41, 38, 32],
      "unavailable comparison remains neutral",
    );
    await accessibility(missing);
    await missing.close();

    // A late API response must not replace the user's newer choice.
    const race = await open(context);
    await calculate(
      race,
      () => race.locator("#openCourtHeatmaps").click(),
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
    await race.locator('[data-heatmap-mode="presence"]').click();
    await calculate(
      race,
      () => race.locator('[data-heatmap-mode="movement"]').click(),
      "movement",
    );
    const late = race.waitForResponse(
      (response) =>
        new URL(response.url()).searchParams.get("mode") === "presence",
    );
    release();
    await late;
    assert.equal(
      await race.locator("#heatmapModeTitle").innerText(),
      "Déplacements",
    );
    await race.close();

    const unavailable = await open(context);
    await unavailable.route("**/api/heatmaps?**", (route) =>
      route.fulfill({ json: { available: false, reason: "geometry" } }),
    );
    await unavailable.locator("#openCourtHeatmaps").click();
    await unavailable.waitForFunction(
      () => document.querySelector("#heatmapSubject").disabled,
    );
    assert.equal(await unavailable.locator(".hm-map-card").count(), 0);
    assert.equal(await unavailable.locator("#heatmapStatus").isVisible(), true);
    await accessibility(unavailable);
    await unavailable.close();
    assert.deepEqual(errors, [], "no browser exceptions");
    console.log(
      "Heatmaps: five modes × player/team, time ranges, camera plans, languages, exports, missing data, response races, responsive layout and accessibility passed",
    );
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
