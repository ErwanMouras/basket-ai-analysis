// Shot UI uses real detector output from a deterministic synthetic flight. No NBA images.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { chromium } = require("playwright");
const base = process.env.VIEWER_URL || "http://127.0.0.1:8766";
const data = JSON.parse(
  execFileSync(
    process.env.PYTHON || "python",
    [
      "-c",
      `
import json
from viewer.test_shots import rows,detect
p=detect(rows()).metadata()
c=detect(rows(), {'court_coordinate_system':{}}).metadata()['events'][0]
c['id']='candidate-demo';c['time_s']=3.5;c['start_s']=2.9;c['end_s']=4.4;c['frame_index']=105
c['release_window_s']=[3.4,3.6]
p['events'].append(c);p['candidate_count']=1
print(json.dumps(p))
`,
    ],
    { cwd: path.join(__dirname, ".."), encoding: "utf8" },
  ),
);
(async () => {
  const browser = await chromium.launch({
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {}),
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1600, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.route("**/api/run?**", async (route) => {
      const response = await route.fetch();
      const run = await response.json();
      run.shots = data;
      // Keep the synthetic sample's roster: an unavailable identity stays unattributed.
      await route.fulfill({ response, json: run });
    });
    await page.goto(`${base}/?run=synthetic`);
    await page.waitForFunction(() => state.loaded && video.readyState >= 2);
    assert.equal(
      await page.locator("#timelineMarkers .shot.probable").count(),
      1,
    );
    assert.equal(
      await page.locator("#timelineMarkers .shot.candidate").count(),
      1,
    );
    await page.locator("#shotsPanel summary").click();
    assert.equal(await page.locator(".shot-row").count(), 2);
    assert.match(
      await page.locator("#shotsSummary").innerText(),
      /1 probables.*1 à vérifier/,
    );
    assert.match(
      await page.locator(".shot-row").first().innerText(),
      /Montée puis descente/,
    );
    assert.match(
      await page.locator(".shot-row").first().innerText(),
      /Tireur non attribué/,
    );
    await page.locator("#shotsCandidates").uncheck();
    assert.equal(await page.locator(".shot-row").count(), 1);
    assert.equal(
      await page.locator("#timelineMarkers .shot.candidate").count(),
      0,
    );
    await page.locator('[data-language="en"]').click();
    assert.match(
      await page.locator("#shotsSummary").innerText(),
      /1 probable.*1 to review/,
    );
    assert.equal(await page.locator("#shotsCandidates").isChecked(), false);
    await page.locator("#shotsCandidates").check();
    await page.locator("#timelineMarkers .shot.candidate").click();
    await page.waitForFunction(
      () =>
        video.currentTime >= 2.9 &&
        video.currentTime < 3.05 &&
        state.current >= nearestFrame(2.9),
    );
    assert.equal(await page.evaluate(() => video.paused), true);
    await page.evaluate(() => {
      video.currentTime = 3.7;
      updateClock(3.7);
    });
    assert.equal(await page.locator(".shot-row.active").count(), 1);
    await page.locator(".shot-row").first().click();
    assert.ok(await page.evaluate(() => video.currentTime < 1));
    for (const [width, height] of [
      [1920, 1080],
      [1366, 768],
      [1024, 768],
      [390, 844],
    ]) {
      await page.setViewportSize({ width, height });
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        true,
        `shot list fits ${width}`,
      );
    }
    await page.setViewportSize({ width: 1600, height: 1000 });
    await page.addScriptTag({ path: require.resolve("axe-core/axe.min.js") });
    const violations = await page.evaluate(async () =>
      (
        await axe.run(document, {
          runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] },
        })
      ).violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.target),
      })),
    );
    assert.deepEqual(violations, []);
    if (process.env.VIEWER_SCREENSHOTS) {
      await fs.mkdir(process.env.VIEWER_SCREENSHOTS, { recursive: true });
      await page.screenshot({
        path: path.join(process.env.VIEWER_SCREENSHOTS, "shots-scripted.png"),
        fullPage: true,
      });
    }
    // Resetting to a legacy run removes markers and explains missing evidence.
    await page.evaluate(() => {
      state.meta.shots = { available: false, events: [] };
      shotStudio.render();
    });
    assert.equal(await page.locator("#timelineMarkers .shot").count(), 0);
    assert.match(
      await page.locator("#shotsSummary").innerText(),
      /insufficient data/,
    );
    assert.deepEqual(errors, []);
    console.log(
      "Shots: detector-backed probable/candidate events, markers, filtering, seeking, FR/EN, missing data, responsive layout and accessibility passed",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
