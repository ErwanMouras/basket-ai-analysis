const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const C = { player: '#c9f579', referee: '#ffbd78', ball: '#ffe579', pose: '#b9a6ff', court: '#e3b1ab' };
const state = { runs: [], name: null, meta: null, frame: null, metrics: [], cache: new Map(), batches: new Map(),
  target: -1, current: -1, generation: 0, sort: 'distance', loaded: false, selected: null, selectedRead: null,
  reads: [], lastStatsRender: 0 };
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
    if (!runs.length) { const opt = new Option('Aucun run terminé', ''); select.add(opt); $('videoEmpty').querySelector('span:last-child').textContent = 'Aucun run terminé dans runs/analysis.'; return; }
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
  video.pause(); state.loaded = false; state.name = name; state.meta = null; state.frame = null; state.metrics = [];
  state.cache.clear(); state.batches.clear(); state.target = -1; state.current = -1; state.selected = null; state.reads = [];
  const generation = ++state.generation; ctx.clearRect(0, 0, canvas.width, canvas.height); $('courtPoints').replaceChildren(); renderSelection();
  $('videoEmpty').classList.remove('hidden'); $('videoEmpty').querySelector('strong').textContent = 'Chargement du run…'; $('videoEmpty').querySelector('span:last-child').textContent = name;
  try {
    const data = await json(`/api/run?${params({ name })}`);
    if (generation !== state.generation) return;
    state.meta = data;
    state.loaded = true;
    history.replaceState(null, '', `?${params({ run: name })}`);
    const run = state.runs.find(r => r.name === name);
    const named=(data.statistics?.players||[]).filter(player=>player.player_name).length;
    $('runMeta').textContent = [run?.match, run?.date, `${data.timestamps.length} images`,
      data.run.identity_resolution?.mode==='posthoc' ? 'Effectif appliqué après analyse'
        : run?.roster ? 'Effectif utilisé par le run' : data.reference_roster ? 'Run sans effectif · référence vidéo disponible' : 'Run sans effectif',
      `${named} joueur${named>1?'s':''} identifié${named>1?'s':''}`,
      'N°* = confirmé ailleurs sur la track',
      data.reference_roster ? 'Boîte pointillée = équipe déduite du n°' : null].filter(Boolean).join(' · ');
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
      state.current = index; state.frame = data.observation; state.metrics = data.metrics; render(data, force);
    }
  } else {
    state.current = -1; state.frame = null; ctx.clearRect(0, 0, canvas.width, canvas.height); $('courtPoints').replaceChildren();
    fetchBatch(index);
    if (force) fetchOne(index);
  }
  if (!video.paused) fetchBatch(Math.min(index + 30, state.meta.timestamps.length - 1));
}
function updateClock(time = video.currentTime || 0) {
  const max = duration(); time = Math.min(time, max);
  $('currentTime').textContent = fmtTime(time);
  $('seek').value = time;
  $('seekFill').style.width = `${max ? (time / max) * 100 : 0}%`;
  $('playIcon').textContent = video.paused ? '▶' : '❚❚';
  $('statsTimeLabel').textContent = `Jusqu'à ${fmtTime(time)}`;
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
    const showBox = (role === 'player' && on.players) || (role === 'referee' && on.referees);
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
      const base = person.resolved_identity?.player_name || (!Object.hasOwn(person,'majority_jersey_number') && !person.jersey_number_suppressed && person.jersey?.player_name) || (number ? `#${number}` : `${role === 'referee' ? 'REF' : 'T'} ${person.track_id ?? '—'}`);
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
    const [x,y] = row.ball.position_px;
    ctx.strokeStyle = C.ball;ctx.fillStyle = C.ball;ctx.lineWidth = 2.5;
    ctx.beginPath();ctx.arc(x,y,11,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.moveTo(x-18,y);ctx.lineTo(x-13,y);ctx.moveTo(x+13,y);ctx.lineTo(x+18,y);ctx.moveTo(x,y-18);ctx.lineTo(x,y-13);ctx.moveTo(x,y+13);ctx.lineTo(x,y+18);ctx.stroke();
  }
  if (on.court && row.court?.keypoints) {
    ctx.fillStyle = C.court;ctx.strokeStyle = C.court;ctx.lineWidth = 1.6;
    row.court.keypoints.forEach((p,i) => { if (!p || p[2] < .5) return;ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(p[0],p[1],2,0,Math.PI*2);ctx.fill();ctx.font='bold 12px Manrope,sans-serif';ctx.fillText(String(i),p[0]+9,p[1]-8); });
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
  const support=row.court?.support_polygon_m;
  if (Array.isArray(support) && support.length>2) svg('polygon',{points:support.filter(p=>safeNumber(p?.[0])&&safeNumber(p?.[1])).map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'#c8f57e',opacity:.09,stroke:'#c8f57e','stroke-width':2},g);
  let points=0;
  for(const person of row.persons||[]) {
    const p=person.position_m;
    if(!Array.isArray(p)||!safeNumber(p[0])||!safeNumber(p[1])||p[0]<0||p[0]>court.L||p[1]<0||p[1]>court.W)continue;
    points++;
    const px=x(p[0]),py=y(p[1]), color=personColor(person,row.segment_id);
    const subject=personSubject(person,row.segment_id);
    const selected=state.selected?.subject===subject;
    svg('circle',{cx:px,cy:py,r:9,fill:color,opacity:.14},g);
    const point=svg('circle',{cx:px,cy:py,r:selected?7:4.2,fill:color,stroke:'#132216','stroke-width':1.5,class:'court-point'},g);
    point.addEventListener('click',()=>selectSubject(subject,person.track_id,row.segment_id));
    if(person.track_id!=null) svg('text',{x:px+8,y:py-7,fill:'#e6f5d7','font-size':9,'font-weight':800},g).textContent=person.resolved_identity?.number ? `#${person.resolved_identity.number}` : String(person.track_id);
  }
  const status=row.court?.status;
  $('courtStatus').textContent=points?`${points} position${points>1?'s':''} projetée${points>1?'s':''} · ${status==='fit'?'calibration directe':status==='propagated'?'calibration propagée':status||'terrain'}`:'Aucune position projetée sur cette image';
  $('courtStatusDot').classList.toggle('off',!points);
}
function playerInfo(subject) {
  const p=state.meta?.statistics?.players?.find(p=>p.subject_id===subject);
  const observed=p?.warnings?.includes('jersey_number_suppressed')?[]:
    (p?.track_refs||[]).map(ref=>observedJersey(ref.track_id,ref.segment_id)).filter(Boolean);
  const observedNumbers=[...new Set(observed.map(o=>o.number).filter(Boolean))];
  const number=p?.jersey_number || (observedNumbers.length===1?observedNumbers[0]:null);
  const name=p?.player_name || (number ? `Joueur n° ${number}`
    : `Track ${p?.track_refs?.[0]?.track_id ?? subject.match(/t(\d+)$/)?.[1] ?? '—'}`);
  const numberSource=p?.jersey_number?'statistics':number?'observations':null;
  const groups=[...new Set(observed.map(o=>o.team_group).filter(Boolean))];
  const group=p?.team_group || (groups.length===1?groups[0]:null);
  const candidates=rosterCandidates(number);
  const roster=state.meta?.roster||state.meta?.reference_roster;
  const confirmedTeam=p?.team_id ? roster?.teams?.find(t=>t.team_id===p.team_id) : null;
  const team=p?.team_id ? confirmedTeam?.name||p.team_id.replaceAll('_',' ')
    : candidates.length===1 ? `${candidates[0].team_name} (n° seul)`
    : candidates.length>1 ? `${candidates.map(c=>c.team_name).join(' ou ')} (n° partagé)`
    : group ? `Groupe couleur ${group.slice(-1)}` : 'Équipe indéterminée';
  const palette=state.meta?.run?.display_colors || {};
  const reference=state.meta?.reference_display_colors || {};
  const color=palette.teams?.[p?.team_id] || palette.groups?.[group]
    || (candidates.length===1 ? reference.teams?.[candidates[0].team_id]||palette.teams?.[candidates[0].team_id] : null) || C.player;
  return {name,team,number,numberSource,group,candidates,track:p?.track_refs?.[0]?.track_id,
    status:p?.measurement_status || 'unavailable',color,record:p};
}
function cell(tr, content, className='') { const td=document.createElement('td');td.className=className;if(content instanceof Node)td.append(content);else td.textContent=content;tr.append(td);return td; }
function renderStats() {
  const body=$('statsBody');body.replaceChildren();
  const all=[...state.metrics].map(m=>({...m,info:playerInfo(m.subject_id)}));
  const search=$('statsSearch').value.trim().toLocaleLowerCase('fr');
  const filter=$('statsFilter').value;
  const rows=all.filter(r=>{
    if(filter==='confirmed'&&r.info.record?.identity_status!=='confirmed')return false;
    if(filter==='anonymous'&&r.info.record?.identity_status==='confirmed')return false;
    if(filter==='measured'&&r.distance_m==null)return false;
    return !search || [r.info.name,r.info.team,r.subject_id,r.info.number].some(v=>String(v||'').toLocaleLowerCase('fr').includes(search));
  });
  if(state.sort==='name')rows.sort((a,b)=>a.info.name.localeCompare(b.info.name,'fr'));
  else rows.sort((a,b)=>(b.distance_m??-1)-(a.distance_m??-1));
  $('statsCount').textContent=`${rows.length}/${all.length}`;
  if(!rows.length){const tr=document.createElement('tr');cell(tr,'Aucun joueur pour ce filtre.','table-empty').colSpan=6;body.append(tr);return;}
  const max=Math.max(1,...rows.map(r=>r.distance_m || 0));
  const fragment=document.createDocumentFragment();
  for(const r of rows){
    const tr=document.createElement('tr');
    tr.classList.toggle('selected',state.selected?.subject===r.subject_id);
    tr.tabIndex=0;tr.addEventListener('click',()=>{const ref=r.info.record?.track_refs?.[0];selectSubject(r.subject_id,ref?.track_id,ref?.segment_id);});
    tr.addEventListener('keydown',e=>{if(e.key==='Enter')tr.click();});
    const player=document.createElement('div');player.className='player-cell';
    const avatar=document.createElement('span');avatar.className='player-avatar';avatar.style.setProperty('--team-color',r.info.color);avatar.textContent=r.info.number?`#${r.info.number}`:`${r.info.track??'?'}`;
    const label=document.createElement('span');const name=document.createElement('span');name.className='player-name';name.textContent=r.info.name;const secondary=document.createElement('span');secondary.className='player-secondary';secondary.textContent=r.info.record?.identity_status==='confirmed'?`N° ${r.info.number}`:r.info.number?'Identité à confirmer':r.subject_id;label.append(name,secondary);player.append(avatar,label);cell(tr,player);
    const team=document.createElement('span');team.className='team-chip';team.style.setProperty('--team-color',r.info.color);const dot=document.createElement('i');team.append(dot,document.createTextNode(r.info.team));cell(tr,team);
    const distance=document.createElement('span');distance.className='distance-number';distance.textContent=r.distance_m==null?'—':fmtDistance(r.distance_m);const wrap=document.createElement('span');wrap.append(distance);
    if(r.distance_m!=null){const unit=document.createElement('span');unit.className='distance-unit-small';unit.textContent=' m';const bar=document.createElement('span');bar.className='distance-bar';bar.style.width=`${Math.max(2,Math.round(74*r.distance_m/max))}px`;wrap.append(unit,bar);}cell(tr,wrap);
    cell(tr,r.distance_m==null?'—':`${fmtDistance(r.measured_duration_s)} s`);
    const coverage=r.info.record?.coverage_ratio;cell(tr,coverage==null?'—':`${Math.round(coverage*100)} %`,coverage!=null&&coverage<.5?'coverage-low':'coverage-good');
    const status=document.createElement('span');status.className=`status-chip ${r.info.status}`;status.textContent=r.info.status==='ok'?'Complet':r.info.status==='partial'?'Partiel':'Indisponible';cell(tr,status);fragment.append(tr);
  }
  body.append(fragment);
}
function render(data, force=false){
  const row=data.observation;
  $('frameBadge').textContent=`FRAME ${String(row.frame_index).padStart(4,'0')}`;
  $('playersCount').textContent=String((row.persons||[]).filter(p=>p.role==='player').length);
  $('projectedCount').textContent=String((row.persons||[]).filter(p=>p.position_m).length);
  $('totalDistance').textContent=data.total_distance_m==null?'—':fmtDistance(data.total_distance_m);
  $('calibrationStatus').textContent=row.court?.status==='fit'?'Calibration terrain validée':row.court?.status==='propagated'?'Calibration terrain propagée':'Calibration terrain indisponible';
  $('frameDetails').textContent=`${row.persons?.length||0} détections · segment ${row.segment_id??'—'}${row.included?'':' · hors analyse'}`;
  drawOverlay(row);renderCourt(row);
  if(force||performance.now()-state.lastStatsRender>300){renderStats();state.lastStatsRender=performance.now();}
}
function renderMarkers(){
  const target=$('timelineMarkers');target.replaceChildren();
  const used=new Set();
  for(const marker of state.meta?.markers||[]){
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
  const markers=[...new Set((state.meta?.markers||[]).map(m=>m.index))].sort((a,b)=>a-b);
  const current=nearestFrame(video.currentTime);
  const next=delta>0?markers.find(i=>i>current):markers.reverse().find(i=>i<current);
  if(next!=null)seekToFrame(next);
}
async function selectSubject(subject,track,segment){
  state.selected={subject,track,segment};state.selectedRead=null;state.reads=[];
  renderSelection();if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}renderStats();
  if(track==null||segment==null)return;
  const generation=state.generation;
  try{const {reads}=await json(`/api/reads?${params({name:state.name,track,segment})}`);
    if(generation===state.generation&&state.selected?.subject===subject&&state.selected?.track===track){state.reads=reads;renderSelection();}
  }catch(err){if(generation===state.generation)toast(err.message);}
}
function renderSelection(){
  const selected=state.selected, details=$('selectionDetails'), list=$('ocrReads');list.replaceChildren();
  if(!selected){details.textContent='Cliquez une boîte, un point du terrain ou une ligne de statistiques.';$('ocrCrop').classList.remove('visible');return;}
  const info=playerInfo(selected.subject), p=info.record;
  const coverage=p?.coverage_ratio==null?'—':`${Math.round(p.coverage_ratio*100)} %`;
  const excluded=Object.entries(p?.excluded_intervals||{}).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([key,count])=>`${key} (${count})`).join(', ');
  const handoff=state.meta?.spatial_handoffs?.find(h=>h.segment_id===selected.segment
    && h.track_id===selected.track && h.start_frame<=state.frame?.frame_index
    && state.frame?.frame_index<=h.end_frame);
  const provenance=handoff?`Identité reliée par continuité spatiale depuis la track ${handoff.source_track_id}${handoff.direction==='backward'?' (apparue ensuite)':''} ; aucun joueur proche ou indice contradictoire à cette transition`
    :p?.team_id?(state.meta?.run?.identity_resolution?.mode==='posthoc'
    ? 'Identité résolue après analyse avec les lectures et la couleur du maillot'
    : 'Équipe validée dans le run'):info.candidates.length===1?'Équipe déduite du numéro avec l’effectif de référence, non validée par le run':info.candidates.length>1?'Numéro partagé par les deux équipes : identité non résolue':'Aucune équipe déduite';
  const candidateNames=!p?.team_id&&info.candidates.length?`\nCandidats : ${info.candidates.map(c=>`${c.player_name} (${c.team_name})`).join(' · ')}`:'';
  const identityLine=info.record?.identity_status==='confirmed'
    ? `${handoff?'Joueur':'Maillot'} n° ${info.number||'inconnu'}`
    : `Track ${selected.track??'—'} · segment ${selected.segment??'—'} · maillot ${info.number||'inconnu'}${info.numberSource==='observations'?' (confirmé dans les observations)':''}`;
  const trackVotes=state.meta?.track_votes?.[`${selected.segment}:${selected.track}`];
  const numericVotes=trackVotes?Object.values(trackVotes.number_votes||{}).reduce((sum,count)=>sum+count,0):0;
  const numberEvidence=trackVotes?`\nVote numéro : ${trackVotes.number_majority?`#${trackVotes.number_majority}`:'indécis'} (${trackVotes.number_majority?(trackVotes.number_votes?.[trackVotes.number_majority]||0):0}/${numericVotes} lectures fiables ; ${trackVotes.number_attempts} images examinées)`:'';
  const teamVotes=trackVotes?Object.values(trackVotes.team_votes||{}).reduce((sum,count)=>sum+count,0):0;
  const teamName=state.meta?.roster?.teams?.find(team=>team.team_id===trackVotes?.team_id)?.name||trackVotes?.team_id||'indécise';
  const teamEvidence=trackVotes?`\nVote équipe : ${teamName} (${trackVotes.team_id?(trackVotes.team_votes?.[trackVotes.team_id]||0):0}/${teamVotes} couleurs lisibles ; ${trackVotes.visible_frames} images de la track)`:'';
  details.textContent=`${info.name} · ${info.team}\n${identityLine}\n${provenance}${candidateNames}${numberEvidence}${teamEvidence}\nCouverture finale : ${coverage} · état : ${info.status}\nExclusions : ${excluded||'aucune'}`;
  if(!state.reads.length){list.textContent='Aucune lecture OCR pour ce passage.';return;}
  const votes=state.reads.filter(r=>r.status==='vote').length;
  const count=document.createElement('div');count.className='section-desc';count.textContent=`${state.reads.length} lectures · ${votes} retenues · ${state.reads.length-votes} rejetées`;list.append(count);
  for(const read of state.reads){
    const button=document.createElement('button');button.type='button';button.className='ocr-read';button.classList.toggle('active',state.selectedRead===read);
    const title=document.createElement('strong');title.textContent=`${fmtTime(read.timestamp_seconds)} · ${read.text||'—'} · ${read.status==='vote'?'vote':'rejet'}`;
    const sub=document.createElement('small');sub.textContent=`Confiance ${safeNumber(read.confidence)?Math.round(read.confidence*100)+' %':'—'} · ${read.rejection_reason||read.region||'—'}`;
    button.append(title,sub);button.addEventListener('click',()=>{video.pause();state.selectedRead=read;seekToFrame(read.frame_index);renderSelection();});list.append(button);
  }
}
function renderCrop(){
  const read=state.selectedRead, crop=$('ocrCrop');
  if(!read?.crop_bbox||video.readyState<2)return;
  const [x1,y1,x2,y2]=read.crop_bbox;
  if(x2<=x1||y2<=y1)return;
  const context=crop.getContext('2d');context.clearRect(0,0,crop.width,crop.height);
  try{context.drawImage(video,x1,y1,x2-x1,y2-y1,0,0,crop.width,crop.height);crop.classList.add('visible');}catch(_err){crop.classList.remove('visible');}
}
$('runSelect').addEventListener('change',e=>loadRun(e.target.value));
$('playButton').addEventListener('click',()=>{if(!state.meta?.video_available)return toast('Vidéo source indisponible.');if(video.paused){if(video.currentTime>=duration())video.currentTime=0;video.play().catch(err=>toast(err.message));}else video.pause();});
video.addEventListener('play',()=>{updateClock();displayAt(video.currentTime);});
video.addEventListener('pause',()=>updateClock());
video.addEventListener('timeupdate',()=>updateClock());
video.addEventListener('seeked',()=>{updateClock();displayAt(video.currentTime,true);renderCrop();});
video.addEventListener('loadedmetadata',resizeOverlay);
video.addEventListener('error',()=>{if(state.loaded)toast('La vidéo ne peut pas être lue par ce navigateur.');});
$('seek').addEventListener('input',e=>{video.currentTime=Number(e.target.value);updateClock();ctx.clearRect(0,0,canvas.width,canvas.height);});
$('prevFrame').addEventListener('click',()=>stepFrame(-1));
$('nextFrame').addEventListener('click',()=>stepFrame(1));
$('prevMarker').addEventListener('click',()=>stepMarker(-1));
$('nextMarker').addEventListener('click',()=>stepMarker(1));
$('clearSelection').addEventListener('click',()=>{state.selected=null;state.reads=[];state.selectedRead=null;renderSelection();if(state.frame){drawOverlay(state.frame);renderCourt(state.frame);}renderStats();});
$('focusVideo').addEventListener('click',()=>{const focused=$('workspace').classList.toggle('focus-video');$('focusVideo').textContent=focused?'Afficher les panneaux':'Agrandir la vidéo';$('focusVideo').setAttribute('aria-pressed',String(focused));resizeOverlay();});
canvas.addEventListener('click',e=>{
  if(!state.frame)return;
  const bounds=canvas.getBoundingClientRect(), px=(e.clientX-bounds.left)*canvas.width/bounds.width, py=(e.clientY-bounds.top)*canvas.height/bounds.height;
  const matches=(state.frame.persons||[]).filter(p=>p.track_id!=null&&p.bbox&&px>=p.bbox[0]&&px<=p.bbox[2]&&py>=p.bbox[1]&&py<=p.bbox[3]);
  matches.sort((a,b)=>(a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1])-(b.bbox[2]-b.bbox[0])*(b.bbox[3]-b.bbox[1]));
  if(matches[0])selectSubject(personSubject(matches[0],state.frame.segment_id),matches[0].track_id,state.frame.segment_id);
});
$('speed').addEventListener('change',e=>video.playbackRate=Number(e.target.value));
$('fullscreen').addEventListener('click',()=>{const el=$('videoStage');if(document.fullscreenElement)document.exitFullscreen();else el.requestFullscreen?.();});
document.addEventListener('fullscreenchange',resizeOverlay);window.addEventListener('resize',resizeOverlay);
document.querySelectorAll('[data-filter]').forEach(input=>input.addEventListener('change',()=>{if(state.frame)drawOverlay(state.frame);}));
$('sortDistance').addEventListener('click',()=>{state.sort='distance';$('sortDistance').classList.add('active');$('sortName').classList.remove('active');renderStats();});
$('sortName').addEventListener('click',()=>{state.sort='name';$('sortName').classList.add('active');$('sortDistance').classList.remove('active');renderStats();});
$('statsSearch').addEventListener('input',renderStats);
$('statsFilter').addEventListener('change',renderStats);
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement?.tagName))return;
  if(e.code==='Space'){e.preventDefault();$('playButton').click();}
  else if(e.code==='ArrowRight'||e.code==='ArrowLeft'){e.preventDefault();if(e.shiftKey)stepFrame(e.code==='ArrowRight'?1:-1);else seekToFrame(nearestFrame(video.currentTime)+(e.code==='ArrowRight'?30:-30));}
  else if(e.code==='KeyN'){stepMarker(1);}else if(e.code==='KeyP'){stepMarker(-1);}
});
drawCourtLines();init();
