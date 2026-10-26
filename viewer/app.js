const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const C = { player: '#c9f579', referee: '#ffbd78', ball: '#ffe579', pose: '#b9a6ff', court: '#65d9f2' };
const state = { runs: [], name: null, meta: null, frame: null, cache: new Map(), batches: new Map(),
  target: -1, current: -1, presentedTime: null, generation: 0, loaded: false, selected: null, teamId: null, statsView: 'players' };
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const maskCache = new Map();
let maskCachePixels = 0;
function filters() {
  const options=Object.fromEntries([...document.querySelectorAll('[data-filter]')].map(input=>[input.dataset.filter,input.checked]));
  options.playerMode=$('playerDisplay').value || 'bbox';
  options.refereeMode=$('refereeDisplay').value || 'bbox';
  options.players=options.playerMode !== 'hidden';
  options.referees=options.refereeMode !== 'hidden';
  return options;
}
function personOptions(person, on) {
  const prefix=person.role === 'referee' ? 'referee' : 'player';
  const mode=on[`${prefix}Mode`];
  return {bbox:mode === 'bbox' || mode === 'both',
    segmentation:mode === 'segmentation' || mode === 'both',
    pose:on[`${prefix}Pose`],labels:on[`${prefix}Labels`]};
}
const params = (values) => new URLSearchParams(values).toString();
let toastTimer;
function toast(message) { const el = $('toast'); el.textContent = message; el.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 4200); }
async function json(url) { const res = await fetch(url); const data = await res.json(); if (!res.ok) throw Error(data.error || t('httpError', {status: res.status})); return data; }
const fmtTime = (n) => { n = Math.max(0, Number(n) || 0); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${(n % 60).toFixed(2).padStart(5, '0')}`; };
const fmtDistance = (n) => (n || 0).toLocaleString(locale(), { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const safeNumber = (n) => typeof n === 'number' && Number.isFinite(n);
const shortName = (path) => path?.split('/').pop() || '—';
function nearestFrame(time) {
  const a = state.meta?.timestamps || [];
  if (!a.length) return -1;
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < time) lo = mid + 1; else hi = mid; }
  if (lo >= a.length) return a.length - 1;
  return lo > 0 && time - a[lo - 1] <= a[lo] - time ? lo - 1 : lo;
}
// currentTime can lie inside a frame; its nearest timestamp may belong to the next image.
function frameAtTime(time) {
  const a = state.meta?.timestamps || [];
  if (!a.length) return -1;
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] <= time + 1e-6) lo = mid + 1; else hi = mid; }
  return Math.max(0, lo - 1);
}
function clearPresentation() {
  state.presentedTime = null; state.target = -1; state.current = -1; state.frame = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height); $('courtPoints').replaceChildren();
}
function duration() { const a = state.meta?.timestamps || []; return a.length ? a[a.length - 1] : 0; }
function renderRunOptions() {
  const select = $('runSelect');
  select.replaceChildren();
  if (!state.runs.length) {
    select.add(new Option(t('noRuns'), ''));
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const run of [...state.runs].sort((a, b) => b.duration_s - a.duration_s || a.name.localeCompare(b.name))) {
    select.add(new Option(`${run.match || run.video || t('matchAnalysis')} · ${fmtTime(run.duration_s)} · ${run.name}`, run.name));
  }
  if (state.name) select.value = state.name;
}
function setEmptyState(title, description, detail = '') {
  state.empty = {title, description, detail};
  $('videoEmpty').classList.remove('hidden');
  renderEmptyState();
}
function renderEmptyState() {
  if (!state.empty) return;
  $('videoEmpty').querySelector('strong').textContent = t(state.empty.title);
  $('videoEmpty').querySelector('span:last-child').textContent = state.empty.description ? t(state.empty.description) : state.empty.detail;
}
function renderMatchHeading() {
  if (!state.meta) return;
  const run = state.runs.find(item => item.name === state.name);
  const teams = matchRoster()?.teams || [];
  const title = $('matchTitle');
  title.removeAttribute('data-i18n');
  title.replaceChildren();
  if (teams.length === 2) {
    title.append(document.createTextNode(teams[0].name));
    const separator = document.createElement('span');
    separator.className = 'match-versus'; separator.textContent = ' / ';
    title.append(separator, document.createTextNode(teams[1].name));
  } else title.textContent = run?.match || t('matchAnalysis');
  const count = (state.meta.statistics?.players || []).filter(player => player.identity_status === 'confirmed').length;
  // Parse date-only metadata without allowing timezone conversion to move the day.
  const date = run?.date && /^\d{4}-\d{2}-\d{2}$/.test(run.date)
    ? new Date(`${run.date}T12:00:00`).toLocaleDateString(locale(), {day: 'numeric', month: 'long', year: 'numeric'}) : run?.date;
  $('runMeta').textContent = [date, t(state.meta.roster ? 'matchRoster' : state.meta.reference_roster ? 'referenceRoster' : 'noRoster'), t('identifiedCount', {count})].filter(Boolean).join(' · ');
}
function renderCourtLegend() {
  const legend = $('courtLegend');
  legend.replaceChildren();
  const entries = (matchRoster()?.teams || []).map(team => ({label: team.name, color: teamColor(team.team_id)}));
  if (!entries.length) entries.push({label: t('players'), color: C.player});
  entries.push({label: t('referees'), color: state.meta?.run?.display_colors?.referee || C.referee});
  entries.push({label: t('calibratedArea'), color: '#b09b80'});
  for (const entry of entries) {
    const item = document.createElement('span');
    const dot = document.createElement('i'); dot.className = 'legend-dot'; dot.setAttribute('aria-hidden', 'true');
    dot.style.setProperty('--team-color', entry.color);
    item.append(dot, document.createTextNode(entry.label)); legend.append(item);
  }
}
function updateLayerSummary() {
  const on = filters();
  const count = Number(on.players) + Number(on.referees && $('identityMode').value !== 'identified') + Number(on.ball) + Number(on.court || on.courtLandmarks);
  $('layerCount').textContent = count;
  $('layerSummary').textContent = count ? t('activeLayers', {count}) : t('noLayers');
  $('refereeOptions').disabled = $('identityMode').value === 'identified';
  $('refereeHint').hidden = !$('refereeOptions').disabled;
}
function updateFocusLabel() {
  const focused = $('workspace').classList.contains('focus-video');
  $('focusLabel').textContent = t(focused ? 'showPanels' : 'expandVideo');
  $('focusVideo').setAttribute('aria-pressed', String(focused));
}
function setLanguage(next) {
  if (!Object.hasOwn(messages, next)) return;
  language = next;
  try { localStorage.setItem('vyneros.language', language); } catch { /* Keep the session usable without storage. */ }
  translateDocument();
  renderRunOptions(); renderMatchHeading(); renderEmptyState();
  renderStatsPanel(); renderCourtLegend(); renderMarkers();
  updateClock(); updateFocusLabel(); updateLayerSummary();
  window.heatmapStudio?.translate();
  if (state.frame) render({observation: state.frame});
  else if (state.name && !state.meta) $('runMeta').textContent = t('loadingMatch');
}
async function init() {
  translateDocument(); updateLayerSummary(); renderCourtLegend(); updateClock();
  try {
    const {runs} = await json('/api/runs'); state.runs = runs;
    renderRunOptions();
    if (!runs.length) { setEmptyState('noRuns', 'noRunsHelp'); renderStatsPanel(); return; }
    const requested = new URLSearchParams(location.search).get('run');
    const initial = runs.find(run => run.name === requested)?.name || [...runs].sort((a,b) => b.duration_s - a.duration_s)[0].name;
    $('runSelect').value = initial;
    await loadRun(initial);
  } catch (err) { setEmptyState('loadError', null, err.message); toast(err.message); }
}
async function loadRun(name) {
  if (!name) return;
  window.heatmapStudio?.reset();
  video.pause(); state.loaded = false; state.name = name; state.meta = null; state.frame = null;
  state.cache.clear(); state.batches.clear(); state.target = -1; state.current = -1; state.selected = null; state.teamId = null; state.statsView = 'players';
  state.presentedTime = null;
  renderScoreboard(0); renderMarkers();
  maskCache.clear(); maskCachePixels = 0;
  const generation = ++state.generation; ctx.clearRect(0, 0, canvas.width, canvas.height); $('courtPoints').replaceChildren();
  $('runMeta').textContent = t('loadingMatch');
  $('currentTime').textContent = fmtTime(0); $('totalTime').textContent = fmtTime(0);
  $('frameDetails').textContent = '—'; $('courtStatus').textContent = t('loadingPositions');
  $('courtStatusDot').classList.add('off'); $('calibrationDot').classList.add('off');
  $('calibrationStatus').textContent = t('courtWaiting');
  renderStatsPanel();
  setEmptyState('loadingRun', null, name);
  try {
    const data = await json(`/api/run?${params({ name })}`);
    if (generation !== state.generation) return;
    state.meta = data;
    state.loaded = true;
    history.replaceState(null, '', `?${params({ run: name })}`);
    renderMatchHeading(); renderCourtLegend();
    renderStatsPanel();
    $('videoFilename').textContent = shortName(data.run.source?.path);
    $('totalTime').textContent = fmtTime(duration());
    $('seek').max = Math.max(duration(), .001);
    $('seek').value = 0;
    $('seekFill').style.width = '0%';
    if (data.video_available) { video.src = `/api/video?${params({ name })}`; video.load(); }
    else { video.removeAttribute('src'); video.load(); setEmptyState('videoUnavailable', 'videoUnavailableHelp'); }
    renderMarkers();
    await fetchBatch(0);
    if (generation !== state.generation) return;
    if (data.video_available) { state.empty = null; $('videoEmpty').classList.add('hidden'); }
    if (!data.video_available || !video.requestVideoFrameCallback) displayAt(0, true);
    else if (state.presentedTime !== null) displayAt(state.presentedTime, true, true);
  } catch (err) {
    if (generation !== state.generation) return;
    setEmptyState('loadError', null, err.message); toast(err.message);
  }
}
async function fetchBatch(index) {
  if (!state.meta || index < 0 || index >= state.meta.timestamps.length) return;
  const start = Math.floor(index / 30) * 30;
  if (state.batches.has(start)) return state.batches.get(start);
  const generation = state.generation;
  const promise = json(`/api/frames?${params({ name: state.name, start, count: 30 })}`).then(({frames}) => {
    if (generation !== state.generation) return;
    for (const data of frames) state.cache.set(data.observation.frame_index, data);
    while (state.cache.size > 180) {
      const oldest=state.cache.keys().next().value;
      state.cache.delete(oldest);state.batches.delete(Math.floor(oldest/30)*30);
    }
    if (state.cache.has(state.target) && state.current !== state.target) displayAt(state.meta.timestamps[state.target], true, true);
    else if (state.current >= start && state.current < start + 35 && state.frame) drawOverlay(state.frame);
  }).catch(err => {
    if (generation !== state.generation) return;
    state.batches.delete(start); toast(err.message);
  });
  state.batches.set(start, promise);
  return promise;
}
async function fetchOne(index) {
  const generation = state.generation;
  try {
    const data = await json(`/api/frame?${params({ name: state.name, index })}`);
    if (generation !== state.generation) return;
    state.cache.set(index, data);
    if (state.target === index) displayAt(state.meta.timestamps[index], true, true);
  } catch (err) { if (generation === state.generation) toast(err.message); }
}
function displayAt(time, force = false, presented = false) {
  const index = presented ? nearestFrame(time) : frameAtTime(time);
  if (index < 0) return;
  state.target = index;
  const data = state.cache.get(index);
  if (data) {
    if (force || state.current !== index) {
      state.current = index; state.frame = data.observation; render(data);
    }
  } else {
    state.current = -1; state.frame = null; ctx.clearRect(0, 0, canvas.width, canvas.height); $('courtPoints').replaceChildren();
    fetchBatch(index);
    if (force) fetchOne(index);
  }
  if (filters().ball && index > 0) fetchBatch(Math.max(0, index - 5));
  if (!video.paused) fetchBatch(Math.min(index + 30, state.meta.timestamps.length - 1));
}
function updateClock(time = video.currentTime || 0) {
  const max = duration(); time = Math.min(time, max);
  $('currentTime').textContent = fmtTime(time);
  $('seek').value = time;
  $('seekFill').style.width = `${max ? (time / max) * 100 : 0}%`;
  renderScoreboard(time);
  window.shotStudio?.update(time);
  $('playIcon').querySelector('path').setAttribute('d', video.paused ? 'm7 4 9 6-9 6Z' : 'M5 4h4v12H5Zm6 0h4v12h-4Z');
  $('playButton').setAttribute('aria-label', t(video.paused ? 'play' : 'pause'));
  if (state.meta && video.currentTime > max + .04 && !video.paused) { video.pause(); video.currentTime = max; }
}
function periodLabel(period) {
  return period <= 4 ? `${t('quarterShort')}${period}` : `${t('overtimeShort')}${period - 4}`;
}
function gameClockLabel(seconds) {
  if (seconds == null) return '—';
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}${seconds % 1 ? `.${Math.round((seconds % 1) * 10)}` : ''}`;
}
function scoreReading(time) {
  const data = state.meta?.scoreboard;
  if (!data?.available) return null;
  const samples = data.samples || [];
  let lo = 0, hi = samples.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (samples[mid].time_s <= time + 1e-8) lo = mid + 1; else hi = mid; }
  const reading = samples[lo - 1];
  return reading && time - reading.time_s <= data.max_age_s ? reading : null;
}
function scoreLabel(reading) {
  return (state.meta?.scoreboard?.teams || []).map(team => `${team.label} ${reading?.scores?.[team.team_id] ?? '—'}`).join('  ·  ');
}
function renderScoreboard(time) {
  const available = state.meta?.scoreboard?.available;
  $('scoreboardContext').hidden = !available;
  if (!available) return;
  const reading = scoreReading(time);
  $('scoreboardMatch').textContent = scoreLabel(reading);
  $('scoreboardPeriod').textContent = reading?.period ? periodLabel(reading.period) : t('gamePeriodUnknown');
  $('scoreboardClock').textContent = gameClockLabel(reading?.clock_s);
  $('scoreboardContext').title = t('scoreObservedHelp');
  renderGamePhase(time);
}
function renderGamePhase(time) {
  const spans = state.meta?.phase_timeline || [];
  const target = $('gamePhase');
  target.hidden = !spans.length;
  if (!spans.length) return;
  let lo = 0, hi = spans.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (spans[mid].start_s <= time) lo = mid + 1;
    else hi = mid;
  }
  const span = spans[lo - 1];
  const owner = span && time < span.end_s ? span.attacking_team_id : null;
  target.hidden = !owner;
  if (!owner) { target.textContent = ''; return; }
  const team = (state.meta?.scoreboard?.teams || []).find(team => team.team_id === owner);
  const opponent = (state.meta?.scoreboard?.teams || []).find(team => team.team_id !== owner);
  target.textContent = t('gamePhaseMatchup', {attack: team?.label || owner, defense: opponent?.label || '—'});
  target.setAttribute('data-source', 'settled');
  target.title = t('gamePhaseHelp');
}
function videoFrameLoop(_now, metadata) {
  if (state.loaded) {
    state.presentedTime = metadata.mediaTime;
    updateClock(metadata.mediaTime); displayAt(metadata.mediaTime, false, true);
  }
  video.requestVideoFrameCallback(videoFrameLoop);
}
if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(videoFrameLoop);
else { const fallback = () => { if (state.loaded && !video.paused && !video.seeking) { updateClock(); displayAt(video.currentTime); } requestAnimationFrame(fallback); }; requestAnimationFrame(fallback); }
function resizeOverlay() {
  const stage = $('videoStage').getBoundingClientRect();
  const w = video.videoWidth || state.meta?.run.source?.width || 1920;
  const h = video.videoHeight || state.meta?.run.source?.height || 1080;
  const scale = Math.min(stage.width / w, stage.height / h);
  const dw = w * scale, dh = h * scale;
  canvas.style.width = `${dw}px`; canvas.style.height = `${dh}px`;
  canvas.style.left = `${(stage.width - dw) / 2}px`; canvas.style.top = `${(stage.height - dh) / 2}px`;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  if (state.frame) drawOverlay(state.frame);
}
const SKELETON = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
function personSubject(person, segment) {
  if(person.jersey_number_suppressed)return `anonymous-s${segment}-t${person.track_id}`;
  const resolved=person.resolved_identity;
  if(resolved?.team_id&&resolved?.number!=null)return `roster:${resolved.team_id}:${resolved.number}`;
  if(Object.hasOwn(person,'majority_jersey_number'))return `anonymous-s${segment}-t${person.track_id}`;
  const jersey = person.jersey || {};
  return jersey.identity_status === 'unique_number' && jersey.status === 'confirmed' && jersey.team_id
    ? `roster:${jersey.team_id}:${jersey.number}` : `anonymous-s${segment}-t${person.track_id}`;
}
function identifiedSubject(person, segment) {
  if (person.role !== 'player') return null;
  const subject = personSubject(person, segment);
  if (!subject.startsWith('roster:')) return null;
  const roster = state.meta?.roster;
  return roster?.teams?.some(team => (team.players || []).some(player =>
    subject === `roster:${team.team_id}:${player.number}`)) ? subject : null;
}
function visiblePerson(person, segment, on) {
  if ($('identityMode').value === 'identified')
    return person.role === 'player' && on.players && Boolean(identifiedSubject(person, segment));
  if (person.role === 'referee') return on.referees;
  return person.role === 'player' && on.players;
}
function observedJersey(track, segment) { return state.meta?.observed_jerseys?.[`${segment}:${track}`] || null; }
function rosterCandidates(number) {
  if(number==null)return [];
  const roster=state.meta?.roster || state.meta?.reference_roster;
  return (roster?.teams||[]).flatMap(team=>(team.players||[])
    .filter(player=>player.number===number&&player.eligible!==false)
    .map(player=>({team_id:team.team_id,team_name:team.name,player_name:player.name})));
}
function personNumber(person, segment) {
  if(person.jersey_number_suppressed)return null;
  if(person.resolved_identity?.number!=null)return person.resolved_identity.number;
  if(Object.hasOwn(person,'majority_jersey_number'))return person.majority_jersey_number;
  return person.jersey?.status==='confirmed'&&person.jersey?.number!=null
    ? person.jersey.number : observedJersey(person.track_id,segment)?.number || null;
}
function personColor(person, segment) {
  const palette = state.meta?.run?.display_colors || {};
  if (person.role === 'referee') return palette.referee || C.referee;
  if(person.jersey_number_suppressed)return person.display_color||C.player;
  const resolved=person.resolved_identity;
  if(resolved?.team_id)return palette.teams?.[resolved.team_id]
    ||state.meta?.reference_display_colors?.teams?.[resolved.team_id]||person.display_color||C.player;
  if(person.resolved_team_id)return palette.teams?.[person.resolved_team_id]||person.display_color||C.player;
  if(Object.hasOwn(person,'majority_jersey_number'))return person.display_color||C.player;
  const jersey = person.jersey || {};
  if(jersey.team_id)return palette.teams?.[jersey.team_id]||person.display_color||C.player;
  if(jersey.team_group)return palette.groups?.[jersey.team_group]||person.display_color||C.player;
  const number=personNumber(person,segment), candidates=rosterCandidates(number);
  if(candidates.length===1)return state.meta?.reference_display_colors?.teams?.[candidates[0].team_id]
    ||palette.teams?.[candidates[0].team_id]||C.player;
  return person.display_color || C.player;
}
function maskSurface(mask, color, width, height) {
  if (mask?.encoding !== 'roi_rle_c') return null;
  const roi = mask.roi, counts = mask.counts;
  if (!Array.isArray(roi) || roi.length !== 4 || !roi.every(Number.isInteger)) return null;
  const [a,b,c,d] = roi, w = c-a, h = d-b, pixels = w*h;
  if (!(a>=0 && b>=0 && w>0 && h>0 && c<=width && d<=height && pixels<=8388608)) return null;
  const cached = maskCache.get(mask);
  if (cached?.color === color) return cached.surface;
  if (!Array.isArray(counts) || !counts.length || !counts.every(n=>Number.isInteger(n) && n>=0)
      || counts.reduce((sum,n)=>sum+n,0) !== pixels) return null;
  if (cached) { maskCache.delete(mask); maskCachePixels -= cached.pixels; }
  while (maskCache.size && (maskCachePixels + pixels > 8388608 || maskCache.size >= 64)) {
    const key = maskCache.keys().next().value;
    maskCachePixels -= maskCache.get(key).pixels; maskCache.delete(key);
  }
  const surface = document.createElement('canvas'); surface.width=w; surface.height=h;
  const paint=surface.getContext('2d'); paint.fillStyle=color;
  let offset=0;
  counts.forEach((count,index)=>{
    const end=offset+count;
    if (index%2) {
      while (offset<end) {
        const x=offset%w, y=Math.floor(offset/w), length=Math.min(w-x,end-offset);
        paint.fillRect(x,y,length,1); offset+=length;
      }
    } else offset=end;
  });
  maskCache.set(mask,{surface,color,pixels}); maskCachePixels+=pixels;
  return surface;
}
function drawSegmentation(row, on) {
  for (const person of row.persons || []) {
    if (!visiblePerson(person,row.segment_id,on) || !personOptions(person,on).segmentation || person.segmentation?.status !== 'ok') continue;
    const mask=person.segmentation.mask;
    const surface=maskSurface(mask,personColor(person,row.segment_id),row.width,row.height);
    if (!surface) continue;
    ctx.save(); ctx.globalAlpha=.38;
    ctx.drawImage(surface,mask.roi[0],mask.roi[1]); ctx.restore();
  }
}
// Same nominal NBA ground-plane references as training/court/geometry.py.
function courtMarkings() {
  const L=28.6512, W=15.24, ft=.3048;
  const lines=[[[0,0],[L,0],[L,W],[0,W],[0,0]], [[L/2,0],[L/2,W]]];
  const arc=(cx,cy,r,a,b,n=120)=>Array.from({length:n+1},(_,i)=>{
    const angle=a+(b-a)*i/n; return [cx+r*Math.cos(angle),cy+r*Math.sin(angle)];
  });
  for(const r of [6*ft,2*ft])lines.push(arc(L/2,W/2,r,0,Math.PI*2));
  const rim=5.25*ft, foul=19*ft, radius=23.75*ft, angle=Math.asin(22*ft/radius);
  const join=rim+Math.sqrt(radius**2-(22*ft)**2);
  for(const mirrored of [false,true]) {
    const half=[[[0,17*ft],[foul,17*ft],[foul,33*ft],[0,33*ft]],
      arc(foul,W/2,6*ft,0,Math.PI*2), [[0,3*ft],[join,3*ft]],
      [[0,47*ft],[join,47*ft]], arc(rim,W/2,radius,-angle,angle),
      [[4*ft,W/2-4*ft],...arc(rim,W/2,4*ft,-Math.PI/2,Math.PI/2,60),[4*ft,W/2+4*ft]]];
    lines.push(...half.map(line=>line.map(([px,py])=>[mirrored?L-px:px,py])));
  }
  return lines;
}
const projectedMarkings=courtMarkings();
function inverseHomography(matrix) {
  if(!Array.isArray(matrix)||matrix.length!==3||matrix.some(row=>!Array.isArray(row)||row.length!==3||!row.every(safeNumber)))return null;
  const magnitude=Math.max(...matrix.flat().map(Math.abs));
  if(!magnitude)return null;
  const [a,b,c,d,e,f,g,h,i]=matrix.flat().map(value=>value/magnitude);
  const cofactors=[e*i-f*h,c*h-b*i,b*f-c*e,f*g-d*i,a*i-c*g,c*d-a*f,d*h-e*g,b*g-a*h,a*e-b*d];
  const det=a*cofactors[0]+b*cofactors[3]+c*cofactors[6];
  return !Number.isFinite(det)||Math.abs(det)<1e-15?null:cofactors.map(value=>value/det);
}
function courtPixel(matrix, point) {
  if(!Array.isArray(point)||point.length!==2||!point.every(safeNumber))return null;
  const [px,py]=point, w=matrix[6]*px+matrix[7]*py+matrix[8];
  if(!Number.isFinite(w)||Math.abs(w)<1e-9)return null;
  const result=[(matrix[0]*px+matrix[1]*py+matrix[2])/w,(matrix[3]*px+matrix[4]*py+matrix[5])/w];
  return result.every(value=>Number.isFinite(value)&&Math.abs(value)<1e6)?result:null;
}
function drawCourtOverlay(row, on) {
  const record=row.court;
  if(!record)return;
  const calibrated=['fit','propagated'].includes(record.status);
  const matrix=calibrated?inverseHomography(record.image_to_court):null;
  const scale=Math.max(1,row.height/540);
  ctx.save();
  if(on.court && matrix) {
    const corners=[[0,0],[28.6512,0],[28.6512,15.24],[0,15.24]];
    const denominators=corners.map(([px,py])=>matrix[6]*px+matrix[7]*py+matrix[8]);
    // Refuse a model crossing the projective horizon, including old artifacts.
    if(Math.min(...denominators)*Math.max(...denominators)>0) {
      const color=C.court;
      const support=Array.isArray(record.support_polygon_m)?record.support_polygon_m.map(point=>courtPixel(matrix,point)):[];
      if(support.length>=3 && support.every(Boolean)) {
        ctx.beginPath();support.forEach(([px,py],i)=>i?ctx.lineTo(px,py):ctx.moveTo(px,py));ctx.closePath();
        ctx.fillStyle=color;ctx.globalAlpha=.09;ctx.fill();ctx.globalAlpha=1;
      }
      ctx.beginPath();
      for(const line of projectedMarkings) {
        let connected=false;
        for(const point of line) {
          const pixel=courtPixel(matrix,point);
          if(!pixel){connected=false;continue;}
          if(connected)ctx.lineTo(...pixel);else ctx.moveTo(...pixel);
          connected=true;
        }
      }
      ctx.lineJoin='round';ctx.lineCap='round';
      // Keep the same appearance between detector frames: changing color or
      // dash every five frames causes visible flashing during playback.
      ctx.setLineDash([]);
      ctx.strokeStyle='#081116';ctx.lineWidth=4*scale;ctx.stroke();
      ctx.strokeStyle=color;ctx.lineWidth=1.8*scale;ctx.stroke();
    }
  }
  if(on.courtLandmarks && Array.isArray(record.keypoints)) {
    const fit=record.fit_attempt;
    const inliers=new Set(fit?.inlier_ids || []);
    record.keypoints.forEach((point,id)=>{
      if(!Array.isArray(point)||point.length!==3||!point.every(safeNumber)||point[2]<.5||point[0]<=1||point[0]>=row.width-1||point[1]<=1||point[1]>=row.height-1)return;
      const accepted=calibrated && !fit?.rejection && inliers.has(id);
      const color=accepted?C.court:'#ff8e9a';
      const [px,py]=point;
      ctx.setLineDash([]);ctx.beginPath();ctx.arc(px,py,4*scale,0,Math.PI*2);
      ctx.strokeStyle='#081116';ctx.lineWidth=3*scale;ctx.stroke();
      ctx.strokeStyle=color;ctx.lineWidth=1.5*scale;ctx.stroke();
      ctx.font=`bold ${10*scale}px Arial, sans-serif`;ctx.lineWidth=2.5*scale;
      ctx.strokeStyle='#081116';ctx.strokeText(String(id+1),px+6*scale,py-5*scale);
      ctx.fillStyle=color;ctx.fillText(String(id+1),px+6*scale,py-5*scale);
    });
  }
  ctx.restore();
}
function drawOverlay(row) {
  resizeCanvasOnly(row);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const on = filters();
  drawCourtOverlay(row, on);
  drawSegmentation(row,on);
  for (const person of row.persons || []) {
    const role = person.role;
    if (!visiblePerson(person, row.segment_id, on)) continue;
    const options=personOptions(person,on);
    const showBox = options.bbox;
    const box = person.bbox;
    if (!Array.isArray(box) || box.length !== 4) continue;
    const color = personColor(person,row.segment_id);
    const selected = state.selected?.subject === personSubject(person, row.segment_id);
    const inferredTeam = role==='player' && !Object.hasOwn(person,'majority_jersey_number')
      && !person.jersey?.team_id && !person.jersey?.team_group
      && rosterCandidates(personNumber(person,row.segment_id)).length===1;
    const [x1,y1,x2,y2] = box;
    const size = Math.max(13, Math.min(28, (x2-x1)*.16));
    if (showBox) {
      ctx.strokeStyle = color; ctx.lineWidth = selected ? 4 : 2.2; ctx.fillStyle = selected ? `${color}29` : `${color}12`;
      ctx.fillRect(x1,y1,x2-x1,y2-y1);
      if(inferredTeam)ctx.setLineDash([8,5]);
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      ctx.setLineDash([]);
      // Draw short corners so the frame remains legible over game footage.
      ctx.beginPath(); ctx.moveTo(x1+size,y1);ctx.lineTo(x1,y1);ctx.lineTo(x1,y1+size);ctx.moveTo(x2-size,y1);ctx.lineTo(x2,y1);ctx.lineTo(x2,y1+size);ctx.moveTo(x1,y2-size);ctx.lineTo(x1,y2);ctx.lineTo(x1+size,y2);ctx.moveTo(x2-size,y2);ctx.lineTo(x2,y2);ctx.lineTo(x2,y2-size);ctx.stroke();
    }
    if (options.labels) {
      const number=personNumber(person,row.segment_id);
      const base = (!person.jersey_number_suppressed && person.resolved_identity?.player_name) || (!Object.hasOwn(person,'majority_jersey_number') && !person.jersey_number_suppressed && person.jersey?.player_name) || (number ? `#${number}` : role === 'referee' ? t('referee') : `Track ${person.track_id ?? '—'}`);
      const title = !person.resolved_identity && !Object.hasOwn(person,'majority_jersey_number')
        && !person.jersey_number_suppressed && person.jersey?.team_group
        ? `${base} · G${person.jersey.team_group.slice(-1)}` : base;
      ctx.font = 'bold 15px Arial, sans-serif';
      const labelW = ctx.measureText(title).width + 18;
      const top = Math.max(0, y1 - 27);
      ctx.fillStyle = '#0c1512eb'; ctx.fillRect(x1, top, labelW, 24);
      ctx.fillStyle = color; ctx.fillText(title, x1+9, top+17);
    }
    if (options.pose && person.pose?.keypoints) {
      const kp = person.pose.keypoints, valid = person.pose.valid || [];
      ctx.strokeStyle = C.pose; ctx.lineWidth = 2; ctx.globalAlpha = .85;
      for (const [a,b] of SKELETON) if (valid[a] && valid[b] && kp[a] && kp[b]) { ctx.beginPath();ctx.moveTo(...kp[a]);ctx.lineTo(...kp[b]);ctx.stroke(); }
      ctx.fillStyle = C.pose;
      kp.forEach((p,i) => { if (valid[i] && p) { ctx.beginPath();ctx.arc(p[0],p[1],3.2,0,Math.PI*2);ctx.fill(); } });
      ctx.globalAlpha = 1;
    }
  }
  if (on.ball && row.ball?.position_px) {
    const current = row.frame_index;
    for (let age = on.ballTrail ? 5 : 0; age >= 1; age--) {
      const previous = state.cache.get(current - age)?.observation;
      if (!previous || previous.segment_id !== row.segment_id) continue;
      // A scene cut ends the visible trail even when segment IDs are absent in older runs.
      if (Array.from({length: age}, (_, offset) => state.cache.get(current - offset)?.observation)
        .some(frame => !frame || frame.scene_cut)) continue;
      const position = previous.ball?.position_px;
      if (!Array.isArray(position) || !position.every(safeNumber)) continue;
      ctx.globalAlpha = .13 + (6 - age) * .12;
      ctx.fillStyle = C.ball;
      ctx.beginPath(); ctx.arc(position[0], position[1], 2 + (6 - age) * 1.15, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    const [x,y] = row.ball.position_px;
    ctx.strokeStyle = C.ball;ctx.fillStyle = C.ball;ctx.lineWidth = 2.5;
    ctx.beginPath();ctx.arc(x,y,11,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.moveTo(x-18,y);ctx.lineTo(x-13,y);ctx.moveTo(x+13,y);ctx.lineTo(x+18,y);ctx.moveTo(x,y-18);ctx.lineTo(x,y-13);ctx.moveTo(x,y+13);ctx.lineTo(x,y+18);ctx.stroke();
  }
}

function resizeCanvasOnly(row) { if (canvas.width !== row.width || canvas.height !== row.height) { canvas.width = row.width; canvas.height = row.height; resizeOverlay(); } }
function svg(tag, attrs, parent) { const el = document.createElementNS(SVG, tag); for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v)); parent.append(el); return el; }
const court = { L: 28.6512, W: 15.24, scale: 20.9, x0: 31, y0: 28 };
const x = m => court.x0 + m * court.scale;
const y = m => court.y0 + m * court.scale;
function drawCourtLines() {
  const g = $('courtLines'); g.replaceChildren();
  const line = { fill:'none',stroke:'#b8a48a', 'stroke-width':1.6, opacity:.72 };
  svg('rect', { x:x(0), y:y(0), width:court.L*court.scale, height:court.W*court.scale, rx:1, ...line }, g);
  svg('line', { x1:x(court.L/2), x2:x(court.L/2), y1:y(0), y2:y(court.W), ...line }, g);
  svg('circle', { cx:x(court.L/2), cy:y(court.W/2), r:1.8288*court.scale, ...line }, g);
  svg('circle', { cx:x(court.L/2), cy:y(court.W/2), r:.6096*court.scale, ...line }, g);
  for (const side of [0,1]) {
    const rim = side ? court.L - 1.6002 : 1.6002;
    const foul = side ? court.L - 5.7912 : 5.7912;
    const laneX = side ? foul : 0;
    svg('rect', { x:x(laneX), y:y((court.W-4.8768)/2), width:5.7912*court.scale,height:4.8768*court.scale,...line },g);
    svg('circle', { cx:x(foul), cy:y(court.W/2),r:1.8288*court.scale,...line },g);
    svg('circle',{cx:x(rim),cy:y(court.W/2),r:7,...line},g);
    svg('line',{x1:x(side?court.L:0),x2:x(side?court.L-.55:.55),y1:y(court.W/2),y2:y(court.W/2),...line},g);
    const radius = 7.239, cy = court.W/2, cornerY = .9144, join = Math.sqrt(radius*radius-(cy-cornerY)**2);
    for (const yy of [cornerY,court.W-cornerY]) svg('line',{x1:x(side?court.L:0),x2:x(rim+(side?-join:join)),y1:y(yy),y2:y(yy),...line},g);
    const points=[];
    for(let i=0;i<=60;i++){const yy=cornerY+(court.W-2*cornerY)*i/60;const dx=Math.sqrt(Math.max(0,radius*radius-(yy-cy)**2));points.push(`${i?'L':'M'}${x(rim+(side?-dx:dx)).toFixed(2)},${y(yy).toFixed(2)}`);}
    svg('path',{d:points.join(' '),...line},g);
    svg('path',{d:`M ${x(rim)},${y(cy-1.2192)} A ${1.2192*court.scale},${1.2192*court.scale} 0 0 ${side?0:1} ${x(rim)},${y(cy+1.2192)}`,...line,opacity:.5},g);
  }
}
function renderCourt(row) {
  const g=$('courtPoints');g.replaceChildren();
  const on=filters();
  const support=row.court?.support_polygon_m;
  if (Array.isArray(support) && support.length>2) svg('polygon',{points:support.filter(p=>safeNumber(p?.[0])&&safeNumber(p?.[1])).map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'#b09b80',opacity:.12,stroke:'#b09b80','stroke-width':2},g);
  let points=0;
  for(const person of row.persons||[]) {
    if (!visiblePerson(person, row.segment_id, on)) continue;
    const p=person.position_m;
    if(!Array.isArray(p)||!safeNumber(p[0])||!safeNumber(p[1])||p[0]<0||p[0]>court.L||p[1]<0||p[1]>court.W)continue;
    points++;
    const px=x(p[0]),py=y(p[1]), color=personColor(person,row.segment_id);
    const subject=personSubject(person,row.segment_id);
    const clickable=Boolean(identifiedSubject(person,row.segment_id));
    const selected=state.selected?.subject===subject;
    svg('circle',{cx:px,cy:py,r:9,fill:color,opacity:.14},g);
    const point=svg('circle',{cx:px,cy:py,r:selected?7:4.2,fill:color,stroke:'#211d19','stroke-width':1.5,class:clickable?'court-point':''},g);
    if(clickable)point.addEventListener('click',()=>selectSubject(subject));
    if(person.track_id!=null) svg('text',{x:px+8,y:py-7,fill:'#f3f0ea','font-size':9,'font-weight':800},g).textContent=personNumber(person,row.segment_id) ? `#${personNumber(person,row.segment_id)}` : $('identityMode').value==='all' ? String(person.track_id) : '';
  }
  $('courtStatus').textContent = points ? t('positions', {count: points}) : t('noPositions');
  $('courtStatusDot').classList.toggle('off',!points);
}
function matchRoster() { return state.meta?.roster || state.meta?.reference_roster || null; }
function rosterPlayer(subject) {
  for (const team of matchRoster()?.teams || []) {
    for (const player of Array.isArray(team.players) ? team.players : []) {
      if (subject === 'roster:'+team.team_id+':'+player.number) return {team,player};
    }
  }
  return null;
}
function teamColor(teamId) {
  const colors=state.meta?.roster ? state.meta?.run?.display_colors?.teams
    : state.meta?.reference_display_colors?.teams;
  return colors?.[teamId] || C.player;
}
function finalPlayerStats(teamId, number) {
  // The player card is sourced exclusively from completed-run statistics.
  const records=state.meta?.roster ? (state.meta?.statistics?.players || []).filter(record =>
    record.identity_status==='confirmed' && record.team_id===teamId
      && String(record.jersey_number)===String(number)) : [];
  const measured=records.filter(record=>safeNumber(record.observed_distance_m));
  const sum = key => records.length && records.every(record=>safeNumber(record[key]))
    ? records.reduce((total,record)=>total+record[key],0) : null;
  const tracked=sum('tracked_duration_s'), duration=sum('measured_duration_s');
  return {observed_distance_m:measured.length
    ? measured.reduce((sum,record)=>sum+record.observed_distance_m,0) : null,
    measured_duration_s:duration,
    coverage_ratio:tracked>0 && duration!=null ? duration/tracked : null};
}
const FINAL_PLAYER_METRICS=[
  {key:'observed_distance_m',label:'distance',unit:'m',format:fmtDistance},
  {key:'measured_duration_s',label:'measuredDuration',unit:'s',format:fmtDistance},
  {key:'coverage_ratio',label:'coverage',unit:'%',format:value=>(100*value).toLocaleString(locale(),{maximumFractionDigits:0})},
];
function setStatsView(view) {
  if (!['players', 'team'].includes(view)) return;
  state.statsView = view;
  renderStatsPanel();
}
function selectTeam(teamId) {
  if (!matchRoster()?.teams?.some(team => team.team_id === teamId)) return;
  const restoreFocus = document.activeElement && $('teamToggle').contains(document.activeElement);
  state.teamId = teamId;
  const selected = state.selected && rosterPlayer(state.selected.subject);
  if (selected && selected.team.team_id !== teamId) state.selected = null;
  renderStatsPanel();
  if (restoreFocus) $('teamToggle').querySelector('[aria-pressed="true"]')?.focus();
  if (state.frame) { drawOverlay(state.frame); renderCourt(state.frame); }
}
function renderTeamSelector(teams) {
  const toggle = $('teamToggle');
  toggle.replaceChildren();
  for (const team of teams) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'team-option';
    button.classList.toggle('active', team.team_id === state.teamId);
    button.setAttribute('aria-pressed', String(team.team_id === state.teamId));
    button.style.setProperty('--team-color', teamColor(team.team_id));
    const dot = document.createElement('i'); dot.setAttribute('aria-hidden', 'true');
    const name = document.createElement('span'); name.textContent = team.name;
    button.append(dot, name);
    button.addEventListener('click', () => selectTeam(team.team_id));
    toggle.append(button);
  }
}
function renderStatsNavigation(teams) {
  if (!teams.some(team => team.team_id === state.teamId)) state.teamId = teams[0]?.team_id || null;
  const teamView = state.statsView === 'team';
  $('playersPanel').hidden = teamView;
  $('teamPanel').hidden = !teamView;
  for (const view of ['players', 'team']) {
    const tab = $(view === 'players' ? 'playersTab' : 'teamTab');
    const active = state.statsView === view;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  renderTeamSelector(teams);
  const team = teams.find(item => item.team_id === state.teamId);
  $('teamHero').hidden = !team;
  $('openTeamHeatmaps').disabled = !team || !state.meta?.roster;
  if (team) {
    $('teamHero').style.setProperty('--team-color', teamColor(team.team_id));
    $('teamName').textContent = team.name;
    $('teamRosterCount').textContent = t('playerCount', {count: Array.isArray(team.players) ? team.players.length : 0});
  }
  $('teamStatsNote').textContent = t(team ? 'teamStatsEmpty' : 'noTeamStats');
}
function renderStatsPanel() {
  const roster=matchRoster();
  renderStatsNavigation(roster?.teams || []);
  const selected=state.selected ? rosterPlayer(state.selected.subject) : null;
  $('rosterBrowse').hidden=Boolean(selected);
  $('rosterDetail').hidden=!selected;
  if (selected) {
    const {team,player}=selected;
    $('openPlayerHeatmaps').disabled = !state.meta?.roster;
    $('playerHero').style.setProperty('--team-color',teamColor(team.team_id));
    $('detailNumber').textContent='#'+player.number;
    $('detailTeam').textContent=team.name;
    $('detailName').textContent=player.name;
    const stats=finalPlayerStats(team.team_id,player.number);
    const tiles=$('finalStats');tiles.replaceChildren();
    for (const metric of FINAL_PLAYER_METRICS) {
      const tile=document.createElement('div');tile.className='final-stat';
      const label=document.createElement('span');label.className='final-stat-label';label.textContent=t(metric.label);
      const value=document.createElement('strong');value.className='final-stat-value';
      const measured=stats[metric.key];value.textContent=measured==null?'—':metric.format(measured);
      tile.append(label,value);
      if(measured!=null){const unit=document.createElement('span');unit.className='final-stat-unit';unit.textContent=metric.unit;tile.append(unit);}
      tiles.append(tile);
    }
    $('finalStatsNote').textContent=!state.meta?.roster
      ? t('referenceNote')
      : stats.observed_distance_m==null ? t('noDistance') : t('statsNote');
    return;
  }
  const heading=$('rosterTeamHeading'),list=$('rosterList');
  heading.replaceChildren();list.replaceChildren();
  if (!roster?.teams?.length) {
    heading.textContent=t('noRosterHeading');
    list.textContent=t('noRosterHelp');
    return;
  }
  const team=roster.teams.find(item=>item.team_id===state.teamId);
  const players=Array.isArray(team.players) ? team.players : [];
  const name=document.createElement('h3');name.textContent=team.name;
  const count=document.createElement('span');count.textContent=t('playerCount', {count: players.length});
  heading.append(name,count);
  for (const player of players) {
    const button=document.createElement('button');button.type='button';button.className='roster-player';
    button.style.setProperty('--team-color',teamColor(team.team_id));
    button.setAttribute('aria-label',t('playerSummary', {name: player.name}));
    const badge=document.createElement('span');badge.className='roster-number';badge.textContent=player.number;
    const playerName=document.createElement('span');playerName.className='roster-player-name';playerName.textContent=player.name;
    const arrow=document.createElement('span');arrow.className='roster-arrow';arrow.setAttribute('aria-hidden','true');arrow.textContent='›';
    button.append(badge,playerName,arrow);
    button.addEventListener('click',()=>selectSubject('roster:'+team.team_id+':'+player.number));
    list.append(button);
  }
}
function render(data){
  const row=data.observation;
  const calibrated = ['fit', 'propagated'].includes(row.court?.status);
  $('calibrationDot').classList.toggle('off', !calibrated);
  $('calibrationStatus').textContent = t(row.court?.status === 'fit' ? 'courtDetected' : row.court?.status === 'propagated' ? 'courtTracked' : 'courtUnavailable');
  $('frameDetails').textContent = t('detections', {count: row.persons?.length || 0}) + (row.included ? '' : ` · ${t('excluded')}`);
  drawOverlay(row);renderCourt(row);
}
function renderMarkers(){
  const target=$('timelineMarkers');target.replaceChildren();
  const used=new Set();
  for(const marker of (state.meta?.markers||[]).filter(marker=>marker.type==='cut')){
    const key=`${marker.index}:${marker.type}`;if(used.has(key))continue;used.add(key);
    const button=document.createElement('button');button.className=marker.type;button.type='button';
    button.style.left=`${100*marker.time/Math.max(duration(),.001)}%`;
    button.title=`${fmtTime(marker.time)} · ${t('sceneCut')}`;button.setAttribute('aria-label',button.title);
    button.addEventListener('click',()=>seekToFrame(marker.index));target.append(button);
  }
  const data=state.meta?.scoreboard;
  $('scoreboardLegend').hidden=!data?.available;
  for(const event of data?.events || []) {
    if(event.time_s > duration()) continue;
    const button=document.createElement('button');button.className='score';button.type='button';
    button.style.left=`${100*event.time_s/Math.max(duration(),.001)}%`;
    button.title=`${fmtTime(event.time_s)} · ${periodLabel(event.period)} · ${gameClockLabel(event.clock_s)} · ${scoreLabel(event)} · ${t('scoreChanges')}`;
    button.setAttribute('aria-label',button.title);
    button.addEventListener('click',()=>seekToFrame(nearestFrame(event.time_s)));target.append(button);
  }
  window.shotStudio?.render();
}

function seekToFrame(index){
  if(!state.meta)return;
  index=Math.max(0,Math.min(index,state.meta.timestamps.length-1));
  const timestamp=state.meta.timestamps[index], next=state.meta.timestamps[index+1];
  // Seek inside the frame: browser timestamp rounding can otherwise select its predecessor.
  const margin=next==null ? .001 : Math.min(.001,(next-timestamp)/2);
  clearPresentation();
  video.currentTime=timestamp+margin;updateClock();
  fetchBatch(index);
}
function stepFrame(delta){video.pause();seekToFrame((state.current>=0?state.current:nearestFrame(video.currentTime))+delta);}
function stepMarker(delta){
  const markers=[...new Set((state.meta?.markers||[]).filter(m=>m.type==='cut').map(m=>m.index))].sort((a,b)=>a-b);
  const current=nearestFrame(video.currentTime);
  const next=delta>0?markers.find(i=>i>current):markers.reverse().find(i=>i<current);
  if(next!=null)seekToFrame(next);
}
function selectSubject(subject){
  const location=rosterPlayer(subject);
  if (!location) return;
  state.selected={subject};
  state.statsView = 'players';
  state.teamId=location.team.team_id;
  renderStatsPanel();
  if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}
}
function backToRoster(){
  state.selected=null;
  renderStatsPanel();
  if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}
}
for (const [id, view] of [['playersTab', 'players'], ['teamTab', 'team']]) {
  const tab = $(id);
  tab.addEventListener('click', () => setStatsView(view));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.code === 'ArrowRight' || event.code === 'ArrowLeft') next = view === 'players' ? 'team' : 'players';
    else if (event.code === 'Home') next = 'players';
    else if (event.code === 'End') next = 'team';
    if (!next) return;
    event.preventDefault();
    setStatsView(next);
    $(next === 'players' ? 'playersTab' : 'teamTab').focus();
  });
}
$('runSelect').addEventListener('change',e=>loadRun(e.target.value));
$('rosterBack').addEventListener('click',backToRoster);
$('playButton').addEventListener('click',()=>{if(!state.meta?.video_available)return toast(t('videoUnavailable'));if(video.paused){if(video.currentTime>=duration())video.currentTime=0;video.play().catch(err=>toast(err.message));}else video.pause();});
video.addEventListener('play',()=>updateClock());
video.addEventListener('pause',()=>updateClock());
video.addEventListener('timeupdate',()=>updateClock());
video.addEventListener('seeking', () => {
  // Chromium may present the destination before dispatching `seeking` after a resize.
  // Keep that already-presented frame; clearing it would leave paused overlays empty.
  if (state.presentedTime !== null && nearestFrame(state.presentedTime) === frameAtTime(video.currentTime)) return;
  clearPresentation();
});
video.addEventListener('seeked',()=>{
  updateClock();
  // A seek may finish before or after the presentation callback. Never overwrite its PTS
  // with the requested currentTime, nor paint a requested mask over the previous image.
  if (!video.requestVideoFrameCallback) displayAt(video.currentTime,true);
  else if (state.presentedTime !== null) displayAt(state.presentedTime,true,true);
});
video.addEventListener('loadedmetadata',resizeOverlay);
video.addEventListener('error',()=>{if(state.loaded)toast(t('videoError'));});
$('seek').addEventListener('input',e=>{clearPresentation();video.currentTime=Number(e.target.value);updateClock();});
$('prevFrame').addEventListener('click',()=>stepFrame(-1));
$('nextFrame').addEventListener('click',()=>stepFrame(1));
$('prevMarker').addEventListener('click',()=>stepMarker(-1));
$('nextMarker').addEventListener('click',()=>stepMarker(1));
$('focusVideo').addEventListener('click',()=>{$('workspace').classList.toggle('focus-video');updateFocusLabel();resizeOverlay();});
canvas.addEventListener('click',e=>{
  if(!state.frame)return;
  const bounds=canvas.getBoundingClientRect(), px=(e.clientX-bounds.left)*canvas.width/bounds.width, py=(e.clientY-bounds.top)*canvas.height/bounds.height;
  const matches=(state.frame.persons||[]).filter(p=>identifiedSubject(p,state.frame.segment_id)&&p.bbox&&px>=p.bbox[0]&&px<=p.bbox[2]&&py>=p.bbox[1]&&py<=p.bbox[3]);
  matches.sort((a,b)=>(a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1])-(b.bbox[2]-b.bbox[0])*(b.bbox[3]-b.bbox[1]));
  if(matches[0])selectSubject(personSubject(matches[0],state.frame.segment_id));
});
$('speed').addEventListener('change',e=>video.playbackRate=Number(e.target.value));
$('fullscreen').addEventListener('click',()=>{const el=$('videoStage');if(document.fullscreenElement)document.exitFullscreen();else el.requestFullscreen?.();});
document.addEventListener('fullscreenchange',resizeOverlay);window.addEventListener('resize',resizeOverlay);
new ResizeObserver(resizeOverlay).observe($('videoStage'));
document.querySelectorAll('[data-filter], [data-display]').forEach(input=>input.addEventListener('change',()=>{updateLayerSummary();if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}}));
$('identityMode').addEventListener('change',()=>{updateLayerSummary();if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}});
document.addEventListener('keydown',e=>{
  if ($('heatmapDialog')?.open) return;
  if (e.code === 'Escape' && $('layersMenu').open) {
    $('layersMenu').open = false; $('layersMenu').querySelector('summary').focus(); return;
  }
  if(['INPUT','SELECT','BUTTON','TEXTAREA','SUMMARY'].includes(document.activeElement?.tagName))return;
  if(e.code==='Space'){e.preventDefault();$('playButton').click();}
  else if(e.code==='ArrowRight'||e.code==='ArrowLeft'){e.preventDefault();if(e.shiftKey)stepFrame(e.code==='ArrowRight'?1:-1);else seekToFrame(nearestFrame(video.currentTime+(e.code==='ArrowRight'?1:-1)));}
  else if(e.code==='KeyN'){stepMarker(1);}else if(e.code==='KeyP'){stepMarker(-1);}
});
document.querySelectorAll('[data-language]').forEach(button => button.addEventListener('click', () => setLanguage(button.dataset.language)));
$('closeLayers').addEventListener('click', () => { $('layersMenu').open = false; $('layersMenu').querySelector('summary').focus(); });
$('resetLayers').addEventListener('click', () => {
  document.querySelectorAll('[data-filter]').forEach(input => { input.checked = input.defaultChecked; });
  $('playerDisplay').value = 'bbox'; $('refereeDisplay').value = 'bbox'; $('identityMode').value = 'all';
  updateLayerSummary();
  if (state.frame) render({observation: state.frame});
});
document.addEventListener('pointerdown', event => {
  if (!$('layersMenu').contains(event.target)) $('layersMenu').open = false;
});
drawCourtLines();init();
