// Team heatmaps use the same data and rendering as the spatial dialog.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const base = process.env.VIEWER_URL || 'http://127.0.0.1:8766';
const settle = page => page.evaluate(() => Promise.all(document.getAnimations().map(a => a.finished.catch(() => {}))));
async function ready(page, count = 1, label = null) {
  await page.waitForFunction(({count, label}) => {
    const root = document.querySelector('#teamHeatmaps');
    return document.querySelector('#team-heatmapStatus').hidden &&
      root.querySelectorAll('.hm-canvas').length === count &&
      (!label || root.querySelector('canvas')?.getAttribute('aria-label').includes(` · ${label} · `));
  }, {count, label});
}
(async () => {
  const browser = await chromium.launch({executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage({viewport: {width: 1600, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`${base}/?run=synthetic`);
    await page.waitForFunction(() => state.loaded && state.frame);
    await page.locator('#statisticsTab').click();
    await page.locator('#teamTab').click();
    await ready(page, 1, 'Example Blue');
    assert.equal(await page.locator('.team-presence-map [data-heatmap-phase]').count(), 2, 'phase controls belong to presence');
    assert.equal(await page.locator('#teamShotMap [data-heatmap-phase]').count(), 0, 'shot map has no phase controls');
    assert.equal(await page.locator('#heatmapDialog').evaluate(el => el.open), false);
    assert.equal(await page.locator('#teamHero').count(), 0, 'no duplicate team heading');
    assert.equal(await page.locator('.hm-export, #exportTeamShots').count(), 0, 'PNG exports removed');
    assert.equal(await page.locator('#team-heatmapPhase, #team-heatmapPassage, #team-heatmapStart, #team-heatmapEnd').count(), 0);
    assert.equal(await page.locator('#teamHeatmaps .hm-map-stats, #teamHeatmaps details, #teamHeatmaps .hm-inspect, #teamHeatmaps .hm-limit, #teamHeatmaps .hm-map-period').count(), 0, 'data sections removed from DOM');

    await settle(page);
    const widths = () => page.locator('#teamToggle button').evaluateAll(buttons => buttons.map(b => b.getBoundingClientRect().width));
    let [a,b] = await widths();
    assert.ok(Math.abs(a / b - 3) < 0.01, 'selected team takes 75%');
    const shotPixels = await page.locator('#teamShotCanvas').evaluate(canvas => canvas.toDataURL());
    await page.locator('#teamHeatmaps [data-heatmap-phase=defense]').click();
    await ready(page);
    assert.equal(await page.locator('#teamShotCanvas').evaluate(canvas => canvas.toDataURL()), shotPixels, 'changing phase preserves the shot map');
    await page.selectOption('#team-heatmapPeriod', '1');
    await ready(page);
    const requests = [];
    page.on('request', req => {
      const url = new URL(req.url());
      if (url.pathname === '/api/heatmaps') requests.push(Object.fromEntries(url.searchParams));
    });
    await page.locator('#teamToggle button').last().click();
    await ready(page, 1, 'Example Pink');
    await settle(page);
    [a,b] = await widths();
    assert.ok(Math.abs(b / a - 3) < 0.01, '75/25 allocation reverses');
    assert.equal(await page.locator('#teamHeatmaps [aria-pressed=true][data-heatmap-phase]').getAttribute('data-heatmap-phase'), 'defense');
    assert.equal(await page.locator('#team-heatmapPeriod').inputValue(), '1');
    assert.equal(requests.at(-1).subject, 'team:pink');
    assert.equal(requests.at(-1).phase, 'defense');
    await page.selectOption('#team-heatmapPeriod', '');
    await ready(page);
    await page.locator('#teamHeatmaps [data-heatmap-mode="zones"]').click();
    await ready(page);
    await page.locator('#compareTeamHeatmaps').click();
    await ready(page, 3);
    assert.equal(requests.at(-1).mode, 'comparison');
    assert.equal(await page.locator('#compareTeamHeatmaps').getAttribute('aria-pressed'), 'true');
    await page.locator('#teamHeatmaps [data-heatmap-phase=offense]').click();
    await ready(page, 3);
    assert.equal(requests.at(-1).phase, 'offense');
    assert.equal(await page.locator('#teamHeatmaps [data-heatmap-phase=offense]').evaluate(el => el === document.activeElement), true, 'phase button keeps keyboard focus after rendering');
    await page.locator('#teamHeatmaps [data-heatmap-phase=defense]').click();
    await ready(page, 3);
    assert.equal(requests.at(-1).phase, 'defense');

    await page.locator('[data-language="en"]').click();
    assert.equal(await page.locator('#compareTeamHeatmaps').innerText(), 'Compare');
    assert.equal(await page.locator('#compareTeamHeatmaps').getAttribute('aria-pressed'), 'true');
    await page.locator('#compareTeamHeatmaps').click();
    await ready(page, 1, 'Example Pink');
    assert.equal(requests.at(-1).mode, 'zones', 'comparison returns to previous map');
    await page.locator('#visualizationTab').click();
    await page.locator('#statisticsTab').click();
    assert.equal(await page.locator('#teamHeatmaps [aria-pressed=true][data-heatmap-phase]').getAttribute('data-heatmap-phase'), 'defense');
    // The dialog remains independent of the team filters.
    await page.locator('#playersTab').click();
    await page.locator('#openPlayerHeatmaps').click();
    await page.waitForFunction(() => document.querySelector('#heatmapStatus').hidden);
    await page.locator('#heatmapPhaseSummary [data-heatmap-phase=offense]').click();
    await page.locator('#closeHeatmaps').click();
    await page.locator('#teamTab').click();
    assert.equal(await page.locator('#teamHeatmaps [aria-pressed=true][data-heatmap-phase]').getAttribute('data-heatmap-phase'), 'defense');
    await settle(page);
    await page.addScriptTag({path: require.resolve('axe-core/axe.min.js')});
    const violations = await page.evaluate(async () => (await axe.run(document, {
      runOnly: {type:'tag',values:['wcag2a','wcag2aa','wcag21aa']},
    })).violations.map(v => ({id: v.id, nodes: v.nodes.map(n => n.target)})));
    assert.deepEqual(violations, [], 'inline heatmaps accessibility');
    for (const [width,height] of [[1280,720],[1024,640],[820,900],[390,844]]) {
      await page.setViewportSize({width,height});
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `no overflow at ${width}`);
      if (width > 900) assert.ok(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight));
      [a,b] = await widths();
      assert.ok(Math.abs(b / a - 3) < 0.01, `75/25 ratio at ${width}`);
    }
    // A slow response from an old team must not replace the latest selection.
    await page.setViewportSize({width:1600,height:1000});
    let slowResponse;
    const arrived = new Promise(resolve => slowResponse = resolve);
    let release;
    const gate = new Promise(resolve => release = resolve);
    await page.route('**/api/heatmaps?**', async route => {
      const url = new URL(route.request().url());
      if (url.searchParams.get('subject') === 'team:blue') {
        const response = await route.fetch();
        slowResponse();
        await gate;
        await route.fulfill({response});
      } else await route.continue();
    });
    await page.locator('#teamToggle button').first().click();
    await arrived;
    await page.locator('#teamToggle button').last().click();
    await ready(page, 1, 'Example Pink');
    release();
    await page.unrouteAll({behavior:'wait'});
    await ready(page, 1, 'Example Pink');
    assert.deepEqual(errors, []);
    console.log('Inline team heatmaps: 75/25 selector, filters, teams, comparison toggle, FR/EN, independent dialog, responsive, accessibility and stale responses passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
