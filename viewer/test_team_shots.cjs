// Counts and maps use synthetic events; no source footage or accuracy claim.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { chromium } = require("playwright");
const base = process.env.VIEWER_URL || "http://127.0.0.1:8766";
const fixture = JSON.parse(
  execFileSync(
    process.env.PYTHON || "python",
    [
      "-c",
      `
import json,copy
from pathlib import Path
from viewer.shot_analytics import summarize_shots
p=Path('examples/demo/runs/synthetic')
r=json.loads((p/'roster.json').read_text());g=json.loads((p/'run.json').read_text())['court_coordinate_system']
def event(i,team,subject,xy,direction,status='probable'):
 return {'id':f'shot-{i}','status':status,'type':'shot_attempt','time_s':i*.7,'start_s':i*.7-.6,'end_s':i*.7+.4,'release_window_s':[i*.7-.1,i*.7+.1],
         'shooter':{'team_id':team,'subject_id':subject,'track_id':i},'origin':{'position_m':xy},'court_direction':direction,
         'frame_index':i*21,'evidence':{'release_observations':3,'arc':True,'rim_approach':True,'camera_compensated':True}}
events=[event(1,'blue','roster:blue:7',[24,4],1),event(2,'pink','roster:pink:12',[g['length']-24,g['width']-4],-1),
 event(3,'blue',None,None,1),event(4,'blue','roster:blue:7',[23,5],1,'candidate'),event(5,None,None,None,None),
 event(6,'pink','roster:pink:12',[g['length']-23,g['width']-5],-1,'candidate')]
meta={'available':True,'events':events,'probable_count':4,'candidate_count':2,**summarize_shots(events,g,r,True)}
print(json.dumps(meta))
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
      run.shots = fixture;
      await route.fulfill({ response, json: run });
    });
    await page.goto(`${base}/?run=synthetic`);
    await page.waitForFunction(() => state.loaded && video.readyState >= 2);
    await page.locator("#statisticsTab").click();
    await page
      .locator(".roster-player")
      .filter({ hasText: "Example A" })
      .first()
      .click();
    assert.equal(await page.locator("#playerShotCount").innerText(), "1");
    await page.locator("#rosterBack").click();
    await page
      .locator(".roster-player")
      .filter({ hasText: "Example A2" })
      .click();
    assert.equal(await page.locator("#playerShotCount").innerText(), "0");
    await page.locator("#teamTab").click();
    assert.match(
      await page.locator("#teamShotSummary").innerText(),
      /2 probables.*1 à vérifier.*2\/3 localisés/,
    );
    assert.match(
      await page.locator("#teamShotMissing").innerText(),
      /Positions manquantes : 1/,
    );
    assert.equal(await page.locator("#teamShotCanvas").isVisible(), true);
    assert.equal(await page.locator('#teamShotLegend').count(), 0, 'shot density scale removed');
    assert.doesNotMatch(await page.locator('#teamShotMap').innerText(), /sans équipe identifiée|tirs\/m²/);
    let pixels = await page
      .locator("#teamShotCanvas")
      .evaluate((c) => c.toDataURL());
    const canvas = page.locator("#teamShotCanvas");
    await page.waitForFunction(() => document.querySelector('#team-heatmapStatus').hidden);
    await canvas.scrollIntoViewIfNeeded();
    const bounds = await canvas.boundingBox();
    const position = fixture.shot_maps.maps[0].points[0],
      geometry = fixture.shot_maps.court;
    const sizes = await canvas.evaluate((c) => ({
      width: c.width,
      height: c.height,
    }));
    await page.mouse.move(
      bounds.x +
        ((24 + (position.x / geometry.length) * (sizes.width - 48)) /
          sizes.width) *
          bounds.width,
      bounds.y +
        ((24 + (position.y / geometry.width) * (sizes.height - 48)) /
          sizes.height) *
          bounds.height,
    );
    assert.match(
      await page.locator("#teamShotInspect").innerText(),
      /Tirs à proximité : 1/,
    );
    const candidate = fixture.shot_maps.maps[0].points.find(point => point.status === 'candidate');
    await page.mouse.move(
      bounds.x + ((24 + candidate.x / geometry.length * (sizes.width - 48)) / sizes.width) * bounds.width,
      bounds.y + ((24 + candidate.y / geometry.width * (sizes.height - 48)) / sizes.height) * bounds.height,
    );
    assert.match(await page.locator('#teamShotInspect').innerText(), /À vérifier/, 'candidate origin is rendered and inspectable');
    assert.equal(await page.locator('#exportTeamShots').count(), 0);
    await page.locator('[data-language="en"]').click();
    assert.match(
      await page.locator("#teamShotSummary").innerText(),
      /2 probable.*1 to review.*2\/3 located/,
    );
    pixels = await canvas.evaluate((c) => c.toDataURL());
    await page.locator("#teamToggle button").nth(1).click();
    assert.match(
      await page.locator("#teamShotSummary").innerText(),
      /1 probable.*1 to review.*2\/2 located/,
    );
    assert.equal(
      (await canvas.evaluate((c) => c.toDataURL())) === pixels,
      true,
      "both cameras produce the same normalized shot position and common scale",
    );
    await page.locator("#playersTab").click();
    await page.locator(".roster-player").first().click();
    assert.equal(await page.locator("#playerShotCount").innerText(), "1");
    await page.locator("#teamTab").click();
    for (const [width, height] of [
      [1920, 1080],
      [1840, 950],
      [1366, 768],
      [1024, 640],
      [1024, 768],
      [390, 844],
    ]) {
      await page.setViewportSize({ width, height });
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        true,
      );
      const presence = await page.locator('.team-presence-map').boundingBox();
      const shots = await page.locator('#teamShotMap').boundingBox();
      if (width > 700) {
        assert.ok(Math.abs(presence.y - shots.y) < 1, 'presence and shots share a row');
        assert.ok(shots.x >= presence.x + presence.width, 'shot map sits beside presence');
      } else assert.ok(shots.y >= presence.y + presence.height, 'maps stack on phones');
      if (width > 900) {
        const layout = await page.evaluate(() => {
          const panel = document.getElementById('teamPanel');
          const presence = document.querySelector('.team-presence-map .hm-canvas').getBoundingClientRect();
          const shots = document.getElementById('teamShotCanvas').getBoundingClientRect();
          const phases = document.getElementById('team-heatmapPhaseSummary').getBoundingClientRect();
          return {overflow: panel.scrollHeight - panel.clientHeight, presence, shots, phases};
        });
        assert.ok(layout.overflow <= 1, `both maps fit without scrolling at ${width}×${height}`);
        assert.ok(Math.abs(layout.presence.y - layout.shots.y) < 1, 'courts align vertically');
        assert.ok(Math.abs(layout.presence.height - layout.shots.height) < 1, 'courts have the same height');
        assert.ok(layout.phases.y >= layout.presence.bottom, 'phase controls sit below presence');
      }
      assert.equal(await canvas.evaluate(c => c.width >= c.clientWidth * devicePixelRatio), true, 'shot map has enough pixels for the display');
      if (width === 1920) {
        assert.ok((await canvas.boundingBox()).width >= 500, 'shot court uses the available width');
        assert.ok((await page.locator('.team-presence-map .hm-canvas').boundingBox()).width >= 500, 'presence court uses the available width');
      }
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
        path: path.join(
          process.env.VIEWER_SCREENSHOTS,
          "team-shot-map-scripted.png",
        ),
        fullPage: true,
      });
    }
    await page.evaluate(() => {
      state.meta.shots.available = false;
      state.meta.shots.player_counts = null;
      state.meta.shots.shot_maps.available = false;
      renderStatsPanel();
    });
    assert.equal(await canvas.isVisible(), false);
    assert.match(
      await page.locator("#teamShotSummary").innerText(),
      /unavailable/,
    );
    await page.locator("#playersTab").click();
    assert.equal(await page.locator("#playerShotCount").innerText(), "—");
    assert.deepEqual(errors, []);
    console.log(
      "Team shot maps: probable and candidate maps, probable player counts, missing positions, camera normalization, shared scale, side-by-side layout, FR/EN, hover, responsive and accessibility passed",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
