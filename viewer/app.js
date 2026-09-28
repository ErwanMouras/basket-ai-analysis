const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const C = { player: '#c9f579', referee: '#ffbd78', ball: '#ffe579', pose: '#b9a6ff', court: '#e3b1ab' };
const state = { runs: [], name: null, meta: null, frame: null, metrics: [], cache: new Map(), requested: -1, requestId: 0, sort: 'distance', loaded: false, lastTime: 0 };
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const filters = () => Object.fromEntries([...document.querySelectorAll('[data-filter]')].map(input => [input.dataset.filter, input.checked]));
const params = (values) => new URLSearchParams(values).toString();
let toastTimer;
function toast(message) { const el = $('toast'); el.textContent = message; el.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 4200); }
async function json(url) { const res = await fetch(url); const data = await res.json(); if (!res.ok) throw Error(data.error || `Erreur HTTP ${res.status}`); return data; }
const fmtTime = (n) => { n = Math.max(0, Number(n) || 0); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${String(Math.floor(n % 60)).padStart(2, '0')}`; };
const fmtDistance = (n) => (n || 0).toLocaleString('fr-FR', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const safeNumber = (n) => typeof n === 'number' && Number.isFinite(n);
const shortName = (path) => path?.split('/').pop() || '—';
function nearestFrame(time) {
  const a = state.meta?.timestamps || [];
  if (!a.length) return -1;
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] <= time + 0.012) lo = mid + 1; else hi = mid; }
  return Math.max(0, Math.min(a.length - 1, lo - 1));
}
function duration() { const a = state.meta?.timestamps || []; return a.length ? a[a.length - 1] : 0; }
async function init() {
  try {
    const { runs } = await json('/api/runs'); state.runs = runs;
    const select = $('runSelect'); select.replaceChildren();
    if (!runs.length) { const opt = new Option('Aucun run terminé', ''); select.add(opt); $('videoEmpty').querySelector('span:last-child').textContent = 'Aucun run terminé dans runs/analysis.'; return; }
    for (const run of runs) select.add(new Option(`${run.name} · ${run.frames} images`, run.name));
    const requested = new URLSearchParams(location.search).get('run');
    const initial = runs.find(r => r.name === requested)?.name || runs.find(r => r.name === 'validation/color-groups')?.name || runs.find(r => r.name === 'validation/optimized-roster')?.name || runs[0].name;
    select.value = initial;
    await loadRun(initial);
  } catch (err) { toast(err.message); }
}
async function loadRun(name) {
  if (!name) return;
  video.pause(); state.loaded = false; state.name = name; state.meta = null; state.frame = null; state.metrics = []; state.cache.clear(); state.requested = -1; state.requestId++;
  $('videoEmpty').classList.remove('hidden'); $('videoEmpty').querySelector('strong').textContent = 'Chargement du run…'; $('videoEmpty').querySelector('span:last-child').textContent = name;
  try {
    const data = await json(`/api/run?${params({ name })}`);
    if (state.name !== name) return;
    state.meta = data;
    state.loaded = true;
    history.replaceState(null, '', `?${params({ run: name })}`);
    $('videoFilename').textContent = shortName(data.run.source?.path);
    $('totalTime').textContent = fmtTime(duration());
    $('seek').max = Math.max(duration(), .001);
    $('seek').value = 0;
    $('seekFill').style.width = '0%';
    $('videoEmpty').classList.add('hidden');
    if (data.video_available) { video.src = `/api/video?${params({ name })}`; video.load(); }
    else { video.removeAttribute('src'); video.load(); $('videoEmpty').classList.remove('hidden'); $('videoEmpty').querySelector('strong').textContent = 'Vidéo source indisponible'; $('videoEmpty').querySelector('span:last-child').textContent = 'Vérifiez le chemin source et --video-root.'; }
    await showFrame(0);
  } catch (err) { $('videoEmpty').querySelector('strong').textContent = 'Impossible de charger le run'; $('videoEmpty').querySelector('span:last-child').textContent = err.message; toast(err.message); }
}
async function showFrame(index) {
  if (!state.meta || index < 0 || index >= state.meta.timestamps.length || index === state.requested) return;
  state.requested = index;
  const token = ++state.requestId;
  try {
    let data = state.cache.get(index);
    if (!data) {
      data = await json(`/api/frame?${params({ name: state.name, index })}`);
      state.cache.set(index, data);
      if (state.cache.size > 180) state.cache.delete(state.cache.keys().next().value);
    }
    if (token !== state.requestId) return;
    state.frame = data.observation;
    state.metrics = data.metrics;
    render(data);
  } catch (err) { if (token === state.requestId) { state.requested = -1; toast(err.message); } }
}
function updateClock() {
  const max = duration(), time = Math.min(video.currentTime || 0, max);
  $('currentTime').textContent = fmtTime(time);
  $('seek').value = time;
  $('seekFill').style.width = `${max ? (time / max) * 100 : 0}%`;
  $('playIcon').textContent = video.paused ? '▶' : '❚❚';
  $('statsTimeLabel').textContent = `Jusqu'à ${fmtTime(time)}`;
  if (state.meta && video.currentTime > max + .04 && !video.paused) { video.pause(); video.currentTime = max; }
  const index = nearestFrame(time);
  if (index !== state.requested) showFrame(index);
}
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
function drawOverlay(row) {
  resizeCanvasOnly(row);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const on = filters();
  for (const person of row.persons || []) {
    const role = person.role;
    const showBox = (role === 'player' && on.players) || (role === 'referee' && on.referees);
    const box = person.bbox;
    if (!Array.isArray(box) || box.length !== 4) continue;
    const color = role === 'referee' ? C.referee : C.player;
    const [x1,y1,x2,y2] = box;
    const size = Math.max(13, Math.min(28, (x2-x1)*.16));
    if (showBox) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.fillStyle = role === 'referee' ? '#ffbd7812' : '#c9f5790e';
      ctx.fillRect(x1,y1,x2-x1,y2-y1);
      // Draw short corners so the frame remains legible over game footage.
      ctx.beginPath(); ctx.moveTo(x1+size,y1);ctx.lineTo(x1,y1);ctx.lineTo(x1,y1+size);ctx.moveTo(x2-size,y1);ctx.lineTo(x2,y1);ctx.lineTo(x2,y1+size);ctx.moveTo(x1,y2-size);ctx.lineTo(x1,y2);ctx.lineTo(x1+size,y2);ctx.moveTo(x2-size,y2);ctx.lineTo(x2,y2);ctx.lineTo(x2,y2-size);ctx.stroke();
    }
    if (on.labels && showBox) {
      const base = person.jersey?.player_name || (person.jersey?.number ? `#${person.jersey.number}` : `${role === 'referee' ? 'REF' : 'T'} ${person.track_id ?? '—'}`);
      const title = person.jersey?.team_group ? `${base} · G${person.jersey.team_group.slice(-1)}` : base;
      ctx.font = 'bold 15px Manrope, sans-serif';
      const labelW = ctx.measureText(title).width + 18;
      const top = Math.max(0, y1 - 27);
      ctx.fillStyle = '#0c1512dd'; ctx.fillRect(x1, top, labelW, 24);
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
    const px=x(p[0]),py=y(p[1]), color=person.role==='referee'?C.referee:C.player;
    svg('circle',{cx:px,cy:py,r:9,fill:color,opacity:.14},g);
    svg('circle',{cx:px,cy:py,r:4.2,fill:color,stroke:'#132216','stroke-width':1.5},g);
    if(person.track_id!=null) svg('text',{x:px+8,y:py-7,fill:'#e6f5d7','font-size':9,'font-weight':800},g).textContent=String(person.track_id);
  }
  const status=row.court?.status;
  $('courtStatus').textContent=points?`${points} position${points>1?'s':''} projetée${points>1?'s':''} · ${status==='fit'?'calibration directe':status==='propagated'?'calibration propagée':status||'terrain'}`:'Aucune position projetée sur cette image';
  $('courtStatusDot').classList.toggle('off',!points);
}
function playerInfo(subject) {
  const p=state.meta?.statistics?.players?.find(p=>p.subject_id===subject);
  const name=p?.player_name || `Track ${p?.track_refs?.[0]?.track_id ?? subject.match(/t(\d+)$/)?.[1] ?? '—'}`;
  const team=p?.team_id ? state.meta?.roster?.teams?.find(t=>t.team_id===p.team_id)?.name || p.team_id.replaceAll('_',' ') : p?.team_group ? `Groupe couleur ${p.team_group.slice(-1)}` : 'Non attribuée';
  return {name,team,number:p?.jersey_number,track:p?.track_refs?.[0]?.track_id, status:p?.measurement_status || 'partial'};
}
function cell(tr, content, className='') { const td=document.createElement('td');td.className=className;if(content instanceof Node)td.append(content);else td.textContent=content;tr.append(td);return td; }
function renderStats() {
  const body=$('statsBody');body.replaceChildren();
  const rows=[...state.metrics].map(m=>({...m,info:playerInfo(m.subject_id)}));
  if(state.sort==='name')rows.sort((a,b)=>a.info.name.localeCompare(b.info.name,'fr'));
  $('statsCount').textContent=String(rows.length);
  if(!rows.length){const tr=document.createElement('tr');cell(tr,'Les statistiques apparaîtront pendant la lecture.','table-empty').colSpan=5;body.append(tr);return;}
  const max=Math.max(1,...rows.map(r=>r.distance_m || 0));
  for(const r of rows.slice(0,60)){
    const tr=document.createElement('tr');
    const player=document.createElement('div');player.className='player-cell';
    const avatar=document.createElement('span');avatar.className='player-avatar';avatar.textContent=r.info.number?`#${r.info.number}`:`${r.info.track??'?'}`;
    const label=document.createElement('span');const name=document.createElement('span');name.className='player-name';name.textContent=r.info.name;const secondary=document.createElement('span');secondary.className='player-secondary';secondary.textContent=r.subject_id;label.append(name,secondary);player.append(avatar,label);cell(tr,player);
    const team=document.createElement('span');team.className='team-chip'+(r.info.team==='Non attribuée'?' alt':'');const dot=document.createElement('i');team.append(dot,document.createTextNode(r.info.team));cell(tr,team);
    const distance=document.createElement('span');distance.className='distance-number';distance.textContent=r.distance_m==null?'—':fmtDistance(r.distance_m);const wrap=document.createElement('span');wrap.append(distance);
    if(r.distance_m!=null){const unit=document.createElement('span');unit.className='distance-unit-small';unit.textContent=' m';const bar=document.createElement('span');bar.className='distance-bar';bar.style.width=`${Math.max(2,Math.round(74*r.distance_m/max))}px`;wrap.append(unit,bar);}cell(tr,wrap);
    cell(tr,r.distance_m==null?'—':`${fmtDistance(r.measured_duration_s)} s`);
    const status=document.createElement('span');status.className='status-chip';status.textContent=r.distance_m==null?'En attente':'Mesuré';cell(tr,status);body.append(tr);
  }
}
function render(data){
  const row=data.observation;
  $('frameBadge').textContent=`FRAME ${String(row.frame_index).padStart(4,'0')}`;
  $('playersCount').textContent=String((row.persons||[]).filter(p=>p.role==='player').length);
  $('projectedCount').textContent=String((row.persons||[]).filter(p=>p.position_m).length);
  $('totalDistance').textContent=data.total_distance_m==null?'—':fmtDistance(data.total_distance_m);
  $('calibrationStatus').textContent=row.court?.status==='fit'?'Calibration terrain validée':row.court?.status==='propagated'?'Calibration terrain propagée':'Calibration terrain indisponible';
  $('frameDetails').textContent=`${row.persons?.length||0} détections · segment ${row.segment_id??'—'}${row.included?'':' · hors analyse'}`;
  drawOverlay(row);renderCourt(row);renderStats();
}
$('runSelect').addEventListener('change',e=>loadRun(e.target.value));
$('playButton').addEventListener('click',()=>{if(!state.meta?.video_available)return toast('Vidéo source indisponible.');if(video.paused){if(video.currentTime>=duration())video.currentTime=0;video.play().catch(err=>toast(err.message));}else video.pause();});
video.addEventListener('play',updateClock);video.addEventListener('pause',updateClock);video.addEventListener('timeupdate',updateClock);video.addEventListener('seeked',updateClock);video.addEventListener('loadedmetadata',resizeOverlay);video.addEventListener('error',()=>{if(state.loaded)toast('La vidéo ne peut pas être lue par ce navigateur.');});
$('seek').addEventListener('input',e=>{video.currentTime=Number(e.target.value);updateClock();});
$('speed').addEventListener('change',e=>video.playbackRate=Number(e.target.value));
$('fullscreen').addEventListener('click',()=>{const el=$('videoStage');if(document.fullscreenElement)document.exitFullscreen();else el.requestFullscreen?.();});
document.addEventListener('fullscreenchange',resizeOverlay);window.addEventListener('resize',resizeOverlay);
document.querySelectorAll('[data-filter]').forEach(input=>input.addEventListener('change',()=>{if(state.frame)drawOverlay(state.frame);}));
$('sortDistance').addEventListener('click',()=>{state.sort='distance';$('sortDistance').classList.add('active');$('sortName').classList.remove('active');renderStats();});
$('sortName').addEventListener('click',()=>{state.sort='name';$('sortName').classList.add('active');$('sortDistance').classList.remove('active');renderStats();});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(document.activeElement?.tagName))return;if(e.code==='Space'){e.preventDefault();$('playButton').click();}if(e.code==='ArrowRight'||e.code==='ArrowLeft'){e.preventDefault();video.currentTime=Math.max(0,Math.min(duration(),video.currentTime+(e.code==='ArrowRight'?1:-1)));updateClock();}});
drawCourtLines();init();
