const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const C = { player: '#c9f579', referee: '#ffbd78', ball: '#ffe579', pose: '#b9a6ff', court: '#e3b1ab' };
const state = { runs: [], name: null, meta: null, frame: null, cache: new Map(), batches: new Map(),
  target: -1, current: -1, generation: 0, loaded: false, selected: null, teamId: null };
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const filters = () => Object.fromEntries([...document.querySelectorAll('[data-filter]')].map(input => [input.dataset.filter, input.checked]));
const params = (values) => new URLSearchParams(values).toString();
let toastTimer;
function toast(message) { const el = $('toast'); el.textContent = message; el.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 4200); }
async function json(url) { const res = await fetch(url); const data = await res.json(); if (!res.ok) throw Error(data.error || `Erreur HTTP ${res.status}`); return data; }
const fmtTime = (n) => { n = Math.max(0, Number(n) || 0); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${(n % 60).toFixed(2).padStart(5, '0')}`; };
const fmtDistance = (n) => (n || 0).toLocaleString('fr-FR', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
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
function duration() { const a = state.meta?.timestamps || []; return a.length ? a[a.length - 1] : 0; }
async function init() {
  try {
    const { runs } = await json('/api/runs'); state.runs = runs;
    const select = $('runSelect'); select.replaceChildren();
    if (!runs.length) { const opt = new Option('Aucun run terminé', ''); select.add(opt); $('videoEmpty').querySelector('span:last-child').textContent = 'Aucun run terminé dans runs/analysis.'; renderRosterPanel(); return; }
    for (const run of [...runs].sort((a,b) => b.duration_s - a.duration_s || a.name.localeCompare(b.name)))
      select.add(new Option(`${run.match || run.video || 'Vidéo'} · ${fmtTime(run.duration_s)} · ${run.name}`, run.name));
    const requested = new URLSearchParams(location.search).get('run');
    const initial = runs.find(r => r.name === requested)?.name || [...runs].sort((a,b) => b.duration_s - a.duration_s)[0].name;
    select.value = initial;
    await loadRun(initial);
  } catch (err) { toast(err.message); }
}
async function loadRun(name) {
  if (!name) return;
  video.pause(); state.loaded = false; state.name = name; state.meta = null; state.frame = null;
  state.cache.clear(); state.batches.clear(); state.target = -1; state.current = -1; state.selected = null; state.teamId = null;
  const generation = ++state.generation; ctx.clearRect(0, 0, canvas.width, canvas.height); $('courtPoints').replaceChildren();
  $('runMeta').textContent='Chargement du match…';
  renderRosterPanel();
  $('videoEmpty').classList.remove('hidden'); $('videoEmpty').querySelector('strong').textContent = 'Chargement du run…'; $('videoEmpty').querySelector('span:last-child').textContent = name;
  try {
    const data = await json(`/api/run?${params({ name })}`);
    if (generation !== state.generation) return;
    state.meta = data;
    state.loaded = true;
    history.replaceState(null, '', `?${params({ run: name })}`);
    const run = state.runs.find(r => r.name === name);
    const named=(data.statistics?.players||[]).filter(player=>player.identity_status==='confirmed').length;
    $('runMeta').textContent = [run?.match, run?.date,
      data.roster ? 'Effectif du match' : data.reference_roster ? 'Effectif indicatif' : 'Sans effectif',
      `${named} joueur${named>1?'s':''} identifié${named>1?'s':''}`].filter(Boolean).join(' · ');
    renderRosterPanel();
    $('videoFilename').textContent = shortName(data.run.source?.path);
    $('totalTime').textContent = fmtTime(duration());
    $('seek').max = Math.max(duration(), .001);
    $('seek').value = 0;
    $('seekFill').style.width = '0%';
    if (data.video_available) { video.src = `/api/video?${params({ name })}`; video.load(); }
    else { video.removeAttribute('src'); video.load(); $('videoEmpty').classList.remove('hidden'); $('videoEmpty').querySelector('strong').textContent = 'Vidéo source indisponible'; $('videoEmpty').querySelector('span:last-child').textContent = 'Vérifiez le chemin source et --video-root.'; }
    renderMarkers();
    await fetchBatch(0);
    if (generation !== state.generation) return;
    if (data.video_available) $('videoEmpty').classList.add('hidden');
    displayAt(0, true);
  } catch (err) { $('videoEmpty').querySelector('strong').textContent = 'Impossible de charger le run'; $('videoEmpty').querySelector('span:last-child').textContent = err.message; toast(err.message); }
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
    if (state.cache.has(state.target) && state.current !== state.target) displayAt(state.meta.timestamps[state.target], true);
    else if (state.current >= start && state.current < start + 35 && state.frame) drawOverlay(state.frame);
  }).catch(err => { state.batches.delete(start); if (generation === state.generation) toast(err.message); });
  state.batches.set(start, promise);
  return promise;
}
async function fetchOne(index) {
  const generation = state.generation;
  try {
    const data = await json(`/api/frame?${params({ name: state.name, index })}`);
    if (generation !== state.generation) return;
    state.cache.set(index, data);
    if (state.target === index) displayAt(state.meta.timestamps[index], true);
  } catch (err) { if (generation === state.generation) toast(err.message); }
}
function displayAt(time, force = false) {
  const index = nearestFrame(time);
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
  $('playIcon').textContent = video.paused ? '▶' : '❚❚';
  if (state.meta && video.currentTime > max + .04 && !video.paused) { video.pause(); video.currentTime = max; }
}
function videoFrameLoop(_now, metadata) {
  if (state.loaded) { updateClock(metadata.mediaTime); displayAt(metadata.mediaTime); }
  video.requestVideoFrameCallback(videoFrameLoop);
}
if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(videoFrameLoop);
else { const fallback = () => { if (state.loaded && !video.paused) { updateClock(); displayAt(video.currentTime); } requestAnimationFrame(fallback); }; requestAnimationFrame(fallback); }
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
function drawOverlay(row) {
  resizeCanvasOnly(row);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const on = filters();
  for (const person of row.persons || []) {
    const role = person.role;
    if (!visiblePerson(person, row.segment_id, on)) continue;
    const showBox = true;
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
    if (on.labels && showBox) {
      const number=personNumber(person,row.segment_id);
      const base = person.resolved_identity?.player_name || (!Object.hasOwn(person,'majority_jersey_number') && !person.jersey_number_suppressed && person.jersey?.player_name) || (number ? `#${number}` : role === 'referee' ? 'Arbitre' : `Track ${person.track_id ?? '—'}`);
      const title = !person.resolved_identity && !Object.hasOwn(person,'majority_jersey_number')
        && !person.jersey_number_suppressed && person.jersey?.team_group
        ? `${base} · G${person.jersey.team_group.slice(-1)}` : base;
      ctx.font = 'bold 15px Manrope, sans-serif';
      const labelW = ctx.measureText(title).width + 18;
      const top = Math.max(0, y1 - 27);
      ctx.fillStyle = '#0c1512eb'; ctx.fillRect(x1, top, labelW, 24);
      ctx.fillStyle = color; ctx.fillText(title, x1+9, top+17);
    }
    if (on.pose && person.pose?.keypoints) {
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
    for (let age = 5; age >= 1; age--) {
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
  if (on.court && row.court?.keypoints) {
    ctx.fillStyle = C.court;ctx.strokeStyle = C.court;ctx.lineWidth = 1.6;
    row.court.keypoints.forEach(p => { if (!p || p[2] < .5) return;ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(p[0],p[1],2,0,Math.PI*2);ctx.fill(); });
  }
}
function resizeCanvasOnly(row) { if (canvas.width !== row.width || canvas.height !== row.height) { canvas.width = row.width; canvas.height = row.height; resizeOverlay(); } }
function svg(tag, attrs, parent) { const el = document.createElementNS(SVG, tag); for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v)); parent.append(el); return el; }
const court = { L: 28.6512, W: 15.24, scale: 20.9, x0: 31, y0: 28 };
const x = m => court.x0 + m * court.scale;
const y = m => court.y0 + m * court.scale;
function drawCourtLines() {
  const g = $('courtLines'); g.replaceChildren();
  const line = { fill:'none',stroke:'#afd692', 'stroke-width':1.6, opacity:.72 };
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
  if (Array.isArray(support) && support.length>2) svg('polygon',{points:support.filter(p=>safeNumber(p?.[0])&&safeNumber(p?.[1])).map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'#c8f57e',opacity:.09,stroke:'#c8f57e','stroke-width':2},g);
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
    const point=svg('circle',{cx:px,cy:py,r:selected?7:4.2,fill:color,stroke:'#132216','stroke-width':1.5,class:clickable?'court-point':''},g);
    if(clickable)point.addEventListener('click',()=>selectSubject(subject));
    if(person.track_id!=null) svg('text',{x:px+8,y:py-7,fill:'#e6f5d7','font-size':9,'font-weight':800},g).textContent=personNumber(person,row.segment_id) ? `#${personNumber(person,row.segment_id)}` : $('identityMode').value==='all' ? String(person.track_id) : '';
  }
  $('courtStatus').textContent=points?`${points} position${points>1?'s':''} sur le terrain`:'Aucune position sur le terrain à cet instant';
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
  return {observed_distance_m:measured.length
    ? measured.reduce((sum,record)=>sum+record.observed_distance_m,0) : null};
}
const FINAL_PLAYER_METRICS=[
  {key:'observed_distance_m',label:'DISTANCE OBSERVÉE',unit:'m',format:fmtDistance},
];
function renderRosterPanel() {
  const roster=matchRoster();
  const selected=state.selected ? rosterPlayer(state.selected.subject) : null;
  $('rosterBrowse').hidden=Boolean(selected);
  $('rosterDetail').hidden=!selected;
  if (selected) {
    const {team,player}=selected;
    $('playerHero').style.setProperty('--team-color',teamColor(team.team_id));
    $('detailNumber').textContent='#'+player.number;
    $('detailTeam').textContent=team.name;
    $('detailName').textContent=player.name;
    const stats=finalPlayerStats(team.team_id,player.number);
    const tiles=$('finalStats');tiles.replaceChildren();
    for (const metric of FINAL_PLAYER_METRICS) {
      const tile=document.createElement('div');tile.className='final-stat';
      const label=document.createElement('span');label.className='final-stat-label';label.textContent=metric.label;
      const value=document.createElement('strong');value.className='final-stat-value';
      const measured=stats[metric.key];value.textContent=measured==null?'—':metric.format(measured);
      tile.append(label,value);
      if(measured!=null){const unit=document.createElement('span');unit.className='final-stat-unit';unit.textContent=metric.unit;tile.append(unit);}
      tiles.append(tile);
    }
    $('finalStatsNote').textContent=!state.meta?.roster
      ? 'Effectif indicatif : aucune statistique attribuée sans identification validée.'
      : stats.observed_distance_m==null ? 'Aucune distance mesurée pour ce joueur.'
      : 'Distance finale observée sur les séquences analysées.';
    return;
  }
  const toggle=$('teamToggle'),heading=$('rosterTeamHeading'),list=$('rosterList');
  toggle.replaceChildren();heading.replaceChildren();list.replaceChildren();
  if (!roster?.teams?.length) {
    heading.textContent='Aucun effectif disponible';
    list.textContent='Aucun joueur associé à ce match.';
    return;
  }
  if (!roster.teams.some(team=>team.team_id===state.teamId)) state.teamId=roster.teams[0].team_id;
  for (const team of roster.teams) {
    const button=document.createElement('button');button.type='button';button.className='team-option';
    button.classList.toggle('active',team.team_id===state.teamId);
    button.setAttribute('aria-pressed',String(team.team_id===state.teamId));
    button.style.setProperty('--team-color',teamColor(team.team_id));
    const dot=document.createElement('i');dot.setAttribute('aria-hidden','true');
    const name=document.createElement('span');name.textContent=team.name;
    button.append(dot,name);
    button.addEventListener('click',()=>{state.teamId=team.team_id;renderRosterPanel();});
    toggle.append(button);
  }
  const team=roster.teams.find(item=>item.team_id===state.teamId);
  const players=Array.isArray(team.players) ? team.players : [];
  const name=document.createElement('h3');name.textContent=team.name;
  const count=document.createElement('span');count.textContent=players.length+' joueurs';
  heading.append(name,count);
  for (const player of players) {
    const button=document.createElement('button');button.type='button';button.className='roster-player';
    button.style.setProperty('--team-color',teamColor(team.team_id));
    button.setAttribute('aria-label','Voir le bilan final de '+player.name);
    const badge=document.createElement('span');badge.className='roster-number';badge.textContent='#'+player.number;
    const playerName=document.createElement('span');playerName.className='roster-player-name';playerName.textContent=player.name;
    const arrow=document.createElement('span');arrow.className='roster-arrow';arrow.setAttribute('aria-hidden','true');arrow.textContent='›';
    button.append(badge,playerName,arrow);
    button.addEventListener('click',()=>selectSubject('roster:'+team.team_id+':'+player.number));
    list.append(button);
  }
}
function render(data){
  const row=data.observation;
  $('calibrationStatus').textContent=row.court?.status==='fit'?'Terrain détecté':row.court?.status==='propagated'?'Terrain suivi':'Terrain indisponible';
  $('frameDetails').textContent=`${row.persons?.length||0} personnes détectées${row.included?'':' · hors analyse'}`;
  drawOverlay(row);renderCourt(row);
}
function renderMarkers(){
  const target=$('timelineMarkers');target.replaceChildren();
  const used=new Set();
  for(const marker of (state.meta?.markers||[]).filter(marker=>marker.type==='cut')){
    const key=`${marker.index}:${marker.type}`;if(used.has(key))continue;used.add(key);
    const button=document.createElement('button');button.className=marker.type;button.type='button';
    button.style.left=`${100*marker.time/Math.max(duration(),.001)}%`;
    button.title=`${fmtTime(marker.time)} · ${marker.label}`;button.setAttribute('aria-label',button.title);
    button.addEventListener('click',()=>seekToFrame(marker.index));target.append(button);
  }
}
function seekToFrame(index){
  if(!state.meta)return;
  index=Math.max(0,Math.min(index,state.meta.timestamps.length-1));
  video.currentTime=state.meta.timestamps[index];updateClock();
  state.target=index;state.current=-1;state.frame=null;ctx.clearRect(0,0,canvas.width,canvas.height);$('courtPoints').replaceChildren();fetchBatch(index);
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
  state.teamId=location.team.team_id;
  renderRosterPanel();
  if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}
}
function backToRoster(){
  state.selected=null;
  renderRosterPanel();
  if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}
}
$('runSelect').addEventListener('change',e=>loadRun(e.target.value));
$('rosterBack').addEventListener('click',backToRoster);
$('playButton').addEventListener('click',()=>{if(!state.meta?.video_available)return toast('Vidéo source indisponible.');if(video.paused){if(video.currentTime>=duration())video.currentTime=0;video.play().catch(err=>toast(err.message));}else video.pause();});
video.addEventListener('play',()=>{updateClock();displayAt(video.currentTime);});
video.addEventListener('pause',()=>updateClock());
video.addEventListener('timeupdate',()=>updateClock());
video.addEventListener('seeked',()=>{updateClock();displayAt(video.currentTime,true);});
video.addEventListener('loadedmetadata',resizeOverlay);
video.addEventListener('error',()=>{if(state.loaded)toast('La vidéo ne peut pas être lue par ce navigateur.');});
$('seek').addEventListener('input',e=>{video.currentTime=Number(e.target.value);updateClock();ctx.clearRect(0,0,canvas.width,canvas.height);});
$('prevFrame').addEventListener('click',()=>stepFrame(-1));
$('nextFrame').addEventListener('click',()=>stepFrame(1));
$('prevMarker').addEventListener('click',()=>stepMarker(-1));
$('nextMarker').addEventListener('click',()=>stepMarker(1));
$('focusVideo').addEventListener('click',()=>{const focused=$('workspace').classList.toggle('focus-video');$('focusVideo').textContent=focused?'Afficher les panneaux':'Agrandir la vidéo';$('focusVideo').setAttribute('aria-pressed',String(focused));resizeOverlay();});
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
document.querySelectorAll('[data-filter]').forEach(input=>input.addEventListener('change',()=>{if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}}));
$('identityMode').addEventListener('change',()=>{document.querySelector('[data-filter="referees"]').disabled=$('identityMode').value==='identified';if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement?.tagName))return;
  if(e.code==='Space'){e.preventDefault();$('playButton').click();}
  else if(e.code==='ArrowRight'||e.code==='ArrowLeft'){e.preventDefault();if(e.shiftKey)stepFrame(e.code==='ArrowRight'?1:-1);else seekToFrame(nearestFrame(video.currentTime)+(e.code==='ArrowRight'?30:-30));}
  else if(e.code==='KeyN'){stepMarker(1);}else if(e.code==='KeyP'){stepMarker(-1);}
});
drawCourtLines();init();
