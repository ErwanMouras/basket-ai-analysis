// Exercise only the public scripted scorebug, including its edited Q1 -> Q3 cut.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const base = process.env.VIEWER_URL || "http://127.0.0.1:8766";
(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`${base}/?run=synthetic`);
    await page.waitForFunction(() => state.loaded && state.current >= 0);
    assert.equal(await page.locator("#scoreboardContext").isVisible(), true);
    assert.match(
      await page.locator("#scoreboardMatch").innerText(),
      /BLU 2.*PNK 0/,
    );
    assert.equal(await page.locator("#scoreboardPeriod").innerText(), "QT1");
    assert.equal(await page.locator("#timelineMarkers .score").count(), 4);
    await page.locator("#timelineMarkers .score").nth(2).click();
    await page.waitForFunction(() => state.current >= 90 && video.paused);
    assert.equal(await page.locator("#scoreboardPeriod").innerText(), "QT3");
    assert.match(
      await page.locator("#scoreboardMatch").innerText(),
      /BLU 52.*PNK 48/,
    );
    await page.locator('[data-language="en"]').click();
    assert.equal(await page.locator("#scoreboardPeriod").innerText(), "Q3");
    assert.match(
      await page
        .locator("#timelineMarkers .score")
        .nth(2)
        .getAttribute("aria-label"),
      /Q3.*Observed score/,
    );
    await page.locator("#openCourtHeatmaps").click();
    await page.waitForFunction(
      () => document.querySelector("#heatmapStatus").hidden,
    );
    assert.equal(await page.locator("#heatmapPeriodField").isVisible(), true);
    assert.equal(await page.locator("#heatmapPeriod option").count(), 3);
    const firstQuarter = page.waitForResponse(
      (r) =>
        r.url().includes("/api/heatmaps?") &&
        new URL(r.url()).searchParams.get("end") === "3",
    );
    await page.selectOption("#heatmapPeriod", "0");
    assert.equal((await (await firstQuarter).json()).scope.end_s, 3);
    await page.waitForFunction(
      () => document.querySelector("#heatmapStatus").hidden,
    );
    const response = page.waitForResponse(
      (r) =>
        r.url().includes("/api/heatmaps?") &&
        new URL(r.url()).searchParams.get("start") === "3",
    );
    await page.selectOption("#heatmapPeriod", "1");
    const data = await (await response).json();
    assert.equal(data.scope.start_s, 3);
    assert.ok(data.scope.end_s <= 179 / 30 + 0.000001);
    await page.waitForFunction(
      () => document.querySelector("#heatmapStatus").hidden,
    );
    assert.ok(
      (await page.locator("#heatmapPassage option").allTextContents())
        .slice(1)
        .every((label) => label.startsWith("Q3")),
    );
    assert.match(
      await page.locator("#heatmapOrientation").innerText(),
      /Sides held per quarter/,
    );
    assert.match(await page.locator("#heatmapScope").innerText(), /Q3/);
    const download = page.waitForEvent("download");
    await page.locator(".hm-map-card button").first().click();
    assert.match((await download).suggestedFilename(), /-q3-/);
    assert.match(
      await page.locator("#heatmapEvidence").innerText(),
      /court area: 0 s/,
    );
    await page.locator("#closeHeatmaps").click();
    await page.route("**/api/run?**", async (route) => {
      const response = await route.fetch();
      const body = await response.json();
      body.phase_timeline = [
        {
          start_s: 0,
          end_s: 179 / 30,
          attacking_team_id: "blue",
          source: "court_context",
        },
      ];
      await route.fulfill({ response, json: body });
    });
    await page.reload();
    await page.waitForFunction(() => state.loaded && state.current >= 0);
    assert.equal(
      await page.locator("#gamePhase").innerText(),
      "BLU attacking · court area",
    );
    await page.locator('[data-language="fr"]').click();
    assert.equal(
      await page.locator("#gamePhase").innerText(),
      "BLU en attaque · zone du terrain",
    );
    await page.unroute("**/api/run?**");
    await page.route("**/api/run?**", async (route) => {
      const response = await route.fetch();
      const body = await response.json();
      delete body.scoreboard;
      await route.fulfill({ response, json: body });
    });
    await page.reload();
    await page.waitForFunction(() => state.loaded);
    assert.equal(await page.locator("#scoreboardContext").isVisible(), false);
    assert.equal(await page.locator("#timelineMarkers .score").count(), 0);
    assert.deepEqual(errors, []);
    console.log(
      "Scoreboard browser checks passed: seeking, period ranges, EN, legacy runs.",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
