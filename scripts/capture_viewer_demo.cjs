// Capture existing viewer outputs. No inference, fabricated detections or API fixtures.
const fs = require('node:fs/promises');
const {createReadStream} = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {chromium} = require('../viewer/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
function option(name, fallback) {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
}
const output = path.resolve(option('--output', path.join(root, 'release/demo-2026-10-27')));
const mode = option('--mode', 'all');
const realBase = option('--real-url', 'http://127.0.0.1:8765');
const sampleBase = option('--sample-url', 'http://127.0.0.1:8766');
const run = option('--run', 'hou-lal-sam-yolo26l-ball');
const manifest = {schema_version: 1, viewer_commit: execFileSync('git', ['rev-parse', 'HEAD'], {cwd:root, encoding:'utf8'}).trim(), captures: []};
async function hash(file) {
  const digest = createHash('sha256');
  for await (const part of createReadStream(file)) digest.update(part);
  return digest.digest('hex');
}
async function settle(page) {
  await page.evaluate(() => Promise.all(document.getAnimations().map(a => a.finished.catch(() => {}))));
}
async function seek(page, time) {
  await page.evaluate(time => {
    video.pause();
    const index = nearestFrame(time);
    if (state.frame?.frame_index !== index || Math.abs(video.currentTime - state.meta.timestamps[index]) > 0.001)
      seekToFrame(index);
  }, time);
  await page.waitForFunction(time => !video.seeking && state.frame?.frame_index === nearestFrame(time) &&
    state.current === nearestFrame(time) && Math.abs(video.currentTime - time) < 0.04, time);
}
async function frame(page, file, overlay = true, jpeg = false) {
  const data = await page.evaluate(({overlay, jpeg}) => {
    const result = document.createElement('canvas');
    result.width = jpeg ? 1280 : video.videoWidth;
    result.height = jpeg ? 720 : video.videoHeight;
    const ctx = result.getContext('2d');
    ctx.drawImage(video, 0, 0, result.width, result.height);
    if (overlay) ctx.drawImage(document.getElementById('overlay'), 0, 0, result.width, result.height);
    return result.toDataURL(jpeg ? 'image/jpeg' : 'image/png', 0.94).split(',')[1];
  }, {overlay, jpeg});
  await fs.writeFile(file, Buffer.from(data, 'base64'));
}
async function open(browser, base, name, options = {}) {
  const context = await browser.newContext({viewport:{width:1840,height:950},deviceScaleFactor:2, reducedMotion:'reduce',...options});
  const recordingStartedAt = Date.now();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`${base}/?${new URLSearchParams({run:name})}`);
  await page.waitForFunction(() => state.loaded && state.frame && video.readyState >= 2);
  await page.locator('[data-language="fr"]').click();
  await page.evaluate(() => document.fonts.ready);
  return {page, context, errors, recordingStartedAt};
}
async function screen(page, dir, name, selector) {
  await settle(page);
  await (selector ? page.locator(selector) : page).screenshot({path:path.join(dir, `${name}.png`)});
}
async function teamMaps(page, dir) {
  await page.locator('#statisticsTab').click();
  await page.locator('#teamTab').click();
  await page.locator('#teamToggle button').last().click();
  await page.waitForFunction(() => document.getElementById('team-heatmapStatus').hidden &&
    document.querySelector('.team-presence-map .hm-canvas')?.getAttribute('aria-label').includes(state.meta.roster.teams.at(-1).name));
  await screen(page, dir, 'team');
  await screen(page, dir, 'team-cards', '.team-map-grid');
  await screen(page, dir, 'presence', '.team-presence-map .hm-canvas');
  await screen(page, dir, 'shots', await page.locator('#teamShotCanvas').isVisible() ? '#teamShotCanvas' : '#teamShotMap');
}
async function real(browser) {
  const dir = path.join(output, 'real');
  await fs.mkdir(dir, {recursive:true});
  const response = await fetch(`${realBase}/api/run?${new URLSearchParams({name:run})}`);
  if (!response.ok) throw Error(`Real run unavailable: HTTP ${response.status}`);
  const data = await response.json();
  const localRun = path.join(root, 'runs/analysis', run);
  const fullRun = JSON.parse(await fs.readFile(path.join(localRun,'run.json'),'utf8'));
  const source = data.run.source.path;
  const observations = path.join(localRun, 'observations.jsonl');
  const sourceHash = await hash(source);
  if (sourceHash !== data.run.source.sha256) throw Error('Source hash does not match run.json');
  const {page, context, errors} = await open(browser, realBase, run);
  const shots = data.shots.events;
  const probable = shots.find(e => e.frame_index === 504 && e.status === 'probable');
  const candidate = shots.find(e => e.frame_index === 1427 && e.status === 'candidate');
  if (!probable || !candidate) throw Error('The reviewed 16.80 s / 47.57 s examples changed; review the storyboard before capture');
  await seek(page, probable.time_s);
  await frame(page, path.join(dir,'raw.png'), false);
  await frame(page, path.join(dir,'bbox.png'));
  await screen(page,dir,'playback');
  await screen(page,dir,'probable-row',`.shot-row[data-shot-id="${probable.id}"]`);
  await page.selectOption('#playerDisplay','segmentation');
  await page.selectOption('#refereeDisplay','segmentation');
  await page.locator('[data-filter="playerLabels"]').uncheck();
  await page.locator('[data-filter="refereeLabels"]').uncheck();
  await frame(page,path.join(dir,'annotated.png'));
  await page.locator('[data-filter="playerPose"]').check();
  await seek(page, probable.time_s);
  await frame(page,path.join(dir,'segmentation.png'));
  await page.locator('#resetLayers').click();
  await page.locator(`.shot-row[data-shot-id="${probable.id}"] .shot-review`).click();
  await page.waitForFunction(time => !video.seeking && Math.abs(video.currentTime - time) < 0.04, probable.start_s);
  await screen(page,dir,'review-start');
  await seek(page,candidate.time_s);
  await screen(page,dir,'candidate-row',`.shot-row[data-shot-id="${candidate.id}"]`);
  await frame(page,path.join(dir,'candidate.png'));
  await page.evaluate(subject=>selectSubject(subject),probable.shooter.subject_id);
  await screen(page,dir,'player');
  const playerSummary = await page.locator('#rosterDetail').innerText();
  await teamMaps(page,dir);
  const teamSummary = await page.locator('#teamShotSummary').innerText();
  await page.locator('#visualizationTab').click();
  await page.locator('#resetLayers').click();
  await page.selectOption('#playerDisplay','segmentation');
  await page.selectOption('#refereeDisplay','segmentation');
  await page.locator('[data-filter="playerLabels"]').uncheck();
  await page.locator('[data-filter="refereeLabels"]').uncheck();
  const frames = path.join(dir,'frames');
  await fs.mkdir(frames,{recursive:true});
  const start=16.1, seconds=3.6, fps=20, count=Math.round(seconds*fps);
  for(let index=0;index<count;index++) {
    await seek(page,start+index/fps);
    await frame(page,path.join(frames,`frame-${String(index).padStart(4,'0')}.jpg`),true,true);
    if(index%20===0) console.log(`Real overlay clip: ${index}/${count} frames`);
  }
  execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-framerate',String(fps),'-i',path.join(frames,'frame-%04d.jpg'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',path.join(dir,'annotated.mp4')]);
  execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-ss',String(start),'-i',source,'-t',String(seconds),'-an','-vf',`fps=${fps},scale=1280:720`,'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',path.join(dir,'raw.mp4')]);
  if(errors.length) throw Error(errors.join('\n'));
  manifest.real = {run,source_sha256:sourceHash,observations_sha256:await hash(observations),frames:data.run.frames,
    source_duration_s:data.timestamps.at(-1),timings:fullRun.timings,capabilities:data.run.capabilities,
    models:Object.fromEntries(Object.entries(fullRun.models).filter(([,v])=>v.checkpoint_sha256).map(([k,v])=>[k,v.checkpoint_sha256])),
    probable:{id:probable.id,time_s:probable.time_s,frame_index:probable.frame_index,status:probable.status,subject:probable.shooter.subject_id,review_start_s:probable.start_s},
    candidate:{id:candidate.id,time_s:candidate.time_s,frame_index:candidate.frame_index,status:candidate.status,subject:candidate.shooter.subject_id},
    team_maps:data.shots.shot_maps.maps.map(({team_id,probable_count,candidate_count,located_count,detected_count})=>({team_id,probable_count,candidate_count,located_count,detected_count})),
    player_summary:playerSummary,team_summary:teamSummary,overlay_clip:{start_s:start,duration_s:seconds,fps,source_fps:data.run.source.nominal_fps},
    presentation_layers:{players:'segmentation',referees:'segmentation',names:false,pose:false,ball:true,court:true},
    note:'Existing inference on development footage. Playback is not real-time inference. Raw footage and these derived captures remain in the ignored release folder.'};
  await context.close();
}
async function sample(browser) {
  const dir=path.join(output,'public');await fs.mkdir(dir,{recursive:true});
  const {page,context,errors}=await open(browser,sampleBase,'synthetic');
  await seek(page,1);
  await screen(page,dir,'playback');
  await page.locator('#statisticsTab').click();
  await screen(page,dir,'player');
  await teamMaps(page,dir);
  await page.locator('#visualizationTab').click();
  await page.selectOption('#playerDisplay','segmentation');
  await screen(page,dir,'layers');
  if(errors.length)throw Error(errors.join('\n'));
  manifest.public={run:'synthetic',synthetic:true,scripted:true,learned_predictions:false,files:['playback.png','player.png','team.png','layers.png']};
  await context.close();
}
async function walkthrough(browser) {
  const recordingDir=path.join(output,'screen-recording');
  await fs.mkdir(recordingDir,{recursive:true});
  const {page,context,errors,recordingStartedAt}=await open(browser,realBase,run,{deviceScaleFactor:1,
    recordVideo:{dir:recordingDir,size:{width:1840,height:950}}});
  await page.selectOption('#playerDisplay','segmentation');
  await page.selectOption('#refereeDisplay','segmentation');
  await page.locator('[data-filter="playerLabels"]').uncheck();
  await page.locator('[data-filter="refereeLabels"]').uncheck();
  await seek(page,16.1);
  const start=Date.now();
  const chapters=[];
  const mark=title=>chapters.push({time_s:(Date.now()-start)/1000,title});
  mark('Lecture des résultats existants · segmentation, ballon et terrain');
  await page.locator('#playButton').click();
  await page.waitForTimeout(4500);
  await page.locator('#playButton').click();
  mark('Voir le tir · retour en pause avant le départ estimé');
  await page.locator('.shot-row[data-shot-id="shot-3"] .shot-review').click();
  await page.waitForFunction(()=>video.paused&&!video.seeking&&Math.abs(video.currentTime-16.2)<0.04);
  await page.waitForTimeout(3500);
  mark('Fiche joueur · distance observée, durée et couverture');
  await page.evaluate(()=>selectSubject('roster:los_angeles_lakers:10'));
  await settle(page);
  await page.waitForTimeout(4000);
  mark('Équipe · présence et origines des tirs');
  await page.locator('#teamTab').click();
  await page.waitForFunction(()=>document.getElementById('team-heatmapStatus').hidden&&document.querySelector('.team-presence-map .hm-canvas'));
  await page.waitForTimeout(5000);
  mark('Défense · seule la présence change');
  await page.locator('#team-heatmapPhaseSummary [data-heatmap-phase="defense"]').click();
  await page.waitForFunction(()=>document.getElementById('team-heatmapStatus').hidden);
  await page.waitForTimeout(4000);
  mark('Candidat à vérifier · relecture, sans résultat marqué / raté');
  await page.locator('#visualizationTab').click();
  await page.locator('.shot-row[data-shot-id="shot-7"] .shot-review').click();
  await page.waitForFunction(()=>video.paused&&!video.seeking&&Math.abs(video.currentTime-46.96666666666666)<0.04);
  await page.waitForTimeout(4000);
  mark('Retour aux cartes · contexte conservé');
  await page.locator('#statisticsTab').click();
  await settle(page);
  await page.waitForTimeout(4000);
  const duration=(Date.now()-start)/1000;
  const recording=page.video();
  await context.close();
  if(errors.length)throw Error(errors.join('\n'));
  const file=await recording.path();
  const target=path.join(output,'viewer-walkthrough.mp4');
  execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-ss',String((start-recordingStartedAt)/1000),'-i',file,
    '-t',String(duration),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',target]);
  await fs.writeFile(path.join(output,'walkthrough-chapters.json'),JSON.stringify({duration_s:duration,chapters,
    note:'Continuous recording of the actual viewer. Analyses were computed before playback. Chapter times describe recording time, not source time.'},null,2)+'\n');
  manifest.walkthrough={file:'viewer-walkthrough.mp4',sha256:await hash(target),duration_s:duration,chapters};
  console.log(`Continuous viewer recording: ${duration.toFixed(1)} s`);
}
(async()=>{
  if(!['all','real','public','walkthrough'].includes(mode))throw Error('--mode must be all, real, public or walkthrough');
  await fs.mkdir(output,{recursive:true});
  if(mode==='walkthrough') {
    const prior=JSON.parse(await fs.readFile(path.join(output,'capture-manifest.json'),'utf8'));
    if(prior.viewer_commit!==manifest.viewer_commit)throw Error('Viewer commit changed since the capture; rerun --mode all');
    Object.assign(manifest,prior);
  }
  const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
  try{
    if(mode==='all'||mode==='real')await real(browser);
    if(mode==='all'||mode==='public')await sample(browser);
    if(mode==='all'||mode==='walkthrough')await walkthrough(browser);
  } finally {await browser.close();}
  manifest.captures = [];
  for(const kind of ['real','public']) {
    if(!manifest[kind])continue;
    const dir=path.join(output,kind);
    for(const name of await fs.readdir(dir))if(/\.(png|mp4)$/.test(name))manifest.captures.push({file:`${kind}/${name}`,sha256:await hash(path.join(dir,name))});
  }
  await fs.writeFile(path.join(output,'capture-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  console.log(`Capture manifest: ${path.join(output,'capture-manifest.json')}`);
})().catch(error=>{console.error(error);process.exitCode=1;});
