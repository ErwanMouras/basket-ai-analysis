/* Review a player's shots while their statistics remain open underneath.
   Reuse the annotated player so video, masks and presentation timestamps stay together. */
window.playerShotReview = (() => {
  const dialog = $('playerShotDialog');
  const entry = $('playerShotStat');
  const candidates = $('playerShotCandidates');
  const mounts = [
    [$('layerControls'), $('playerShotLayers')],
    [$('videoStage'), $('playerShotStage')],
    [document.querySelector('.player-controls'), $('playerShotControls')],
  ];
  let subject = null, generation = null, active = null, clip = null, anchors = [], playRequest = 0;
  const isOpen = () => dialog.open;
  const allEvents = () => state.selected && state.meta?.roster && state.meta?.shots?.available
    ? (state.meta.shots.events || []).filter(event => event.shooter?.subject_id === state.selected?.subject)
      .sort((a, b) => a.time_s - b.time_s)
    : [];
  const events = () => allEvents().filter(event => candidates.checked || event.status === 'probable');
  const status = event => t(event.status === 'probable' ? 'shotProbable' : 'shotCandidate');

  function movePlayer() {
    anchors = mounts.map(([node, target]) => {
      const anchor = document.createComment('annotated player home');
      node.before(anchor);
      target.append(node);
      return [anchor, node];
    });
    resizeOverlay();
  }
  function restorePlayer() {
    playRequest++;
    video.pause();
    for (const [anchor, node] of anchors) anchor.replaceWith(node);
    anchors = [];
    subject = null; active = null; clip = null;
    resizeOverlay();
  }
  function reset() {
    if (!isOpen() && !anchors.length) return;
    video.pause();
    dialog.close();
    // Restore synchronously, including when switching runs before the close event.
    restorePlayer();
  }
  function translate() {
    const player = rosterPlayer(subject)?.player;
    if (!player) return;
    $('playerShotTitle').textContent = `${player.name} · #${player.number}`;
    $('playerShotLanguage').textContent = language === 'fr' ? 'EN' : 'FR';
    const all = allEvents();
    $('playerShotSummary').textContent = t('playerShotSummary', {
      probable: all.filter(event => event.status === 'probable').length,
      candidates: all.filter(event => event.status === 'candidate').length,
    });
    $('playerShotCandidatesField').hidden = !all.some(event => event.status === 'candidate')
      || !all.some(event => event.status === 'probable');
    renderList();
    update(video.currentTime);
  }
  function sync() {
    const available = allEvents();
    entry.disabled = !state.loaded || !state.meta?.video_available || !available.length;
    if (!isOpen()) return;
    if (generation !== state.generation || subject !== state.selected?.subject
        || state.mode !== 'statistics' || state.statsView !== 'players' || entry.disabled) {
      reset();
      return;
    }
    translate();
  }
  function renderList() {
    const list = $('playerShotList');
    const scroll = list.scrollTop;
    const focused = document.activeElement?.closest('[data-player-shot]')?.dataset.playerShot;
    list.replaceChildren();
    for (const [index, event] of events().entries()) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = `player-shot-row ${event.status}`;
      button.dataset.playerShot = event.id;
      button.setAttribute('aria-current', String(event.id === active));
      const timing = document.createElement('span'); timing.className = 'shot-timing';
      const time = document.createElement('strong'); time.className = 'shot-time'; time.textContent = fmtTime(event.time_s);
      const frame = document.createElement('span'); frame.className = 'shot-frame';
      frame.textContent = t('shotFrame', {frame: event.frame_index}); timing.append(time, frame);
      const badge = document.createElement('span'); badge.className = 'shot-status'; badge.textContent = status(event);
      button.append(timing, badge);
      button.setAttribute('aria-label', `${index + 1} · ${fmtTime(event.time_s)} · ${status(event)} · ${t('shotReview')}`);
      button.addEventListener('click', () => select(event.id));
      list.append(button);
      if (event.id === focused) button.focus({preventScroll: true});
    }
    list.scrollTop = scroll;
    const index = events().findIndex(event => event.id === active);
    $('playerShotIndex').textContent = `${Math.max(0, index + 1)} / ${events().length}`;
    $('previousPlayerShot').disabled = index <= 0;
    $('nextPlayerShot').disabled = index < 0 || index >= events().length - 1;
  }
  function select(id) {
    const event = events().find(event => event.id === id);
    if (!event) return;
    video.pause();
    active = event.id;
    // Keep the lead-in and enough of the flight to review the attempt.
    const first = nearestFrame(Math.max(0, event.start_s));
    const last = Math.max(first, nearestFrame(Math.min(duration(), Math.max(event.end_s, event.time_s + 2))));
    clip = {first, last, start: state.meta.timestamps[first], end: state.meta.timestamps[last]};
    $('playerShotSeek').min = clip.start;
    $('playerShotSeek').max = clip.end;
    $('playerShotSeek').disabled = clip.end <= clip.start;
    renderList();
    seekToFrame(first);
    play();
  }
  function play() {
    if (!clip || !isOpen()) return;
    if (video.currentTime >= clip.end - .001 || video.currentTime < clip.start)
      seekToFrame(clip.first);
    const request = ++playRequest;
    video.play().catch(error => {
      if (isOpen() && request === playRequest && error.name !== 'AbortError') toast(error.message);
    });
  }
  function togglePlayback() { if (video.paused) play(); else video.pause(); }
  function open() {
    if (entry.disabled || isOpen() || !state.loaded || !state.selected
        || state.mode !== 'statistics' || !state.meta?.video_available || !allEvents().length) return;
    video.pause();
    subject = state.selected.subject; generation = state.generation;
    candidates.checked = !allEvents().some(event => event.status === 'probable');
    dialog.showModal();
    movePlayer();
    translate();
    select(events()[0]?.id);
  }
  function update(time) {
    if (!isOpen() || !clip) return;
    const bounded = Math.max(clip.start, Math.min(clip.end, time));
    $('playerShotSeek').value = bounded;
    $('playerShotSeek').setAttribute('aria-valuetext', fmtTime(bounded));
    $('playerShotSeekFill').style.width = `${clip.end > clip.start ? 100 * (bounded - clip.start) / (clip.end - clip.start) : 0}%`;
    $('playerShotTime').textContent = `${fmtTime(bounded - clip.start)} / ${fmtTime(clip.end - clip.start)}`;
    if (time >= clip.end && !video.paused) {
      video.pause();
      // The presented frame can lead currentTime slightly. Land inside the
      // final frame so pause, replay and frame stepping share the same position.
      video.currentTime = clip.end + .001;
    }
  }
  function clampFrame(index) { return isOpen() && clip ? Math.max(clip.first, Math.min(clip.last, index)) : index; }
  function keydown(event) {
    if (['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    if (event.code === 'Space') { event.preventDefault(); togglePlayback(); }
    if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
      event.preventDefault();
      const direction = event.code === 'ArrowRight' ? 1 : -1;
      if (event.shiftKey) stepFrame(direction); else stepSecond(direction);
    }
  }
  entry.addEventListener('click', open);
  $('closePlayerShots').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    // A queued close event from a reset must not affect a newly opened review.
    if (isOpen() || !anchors.length) return;
    restorePlayer();
    if (entry.isConnected && !entry.disabled && state.mode === 'statistics') entry.focus({preventScroll: true});
  });
  $('playerShotLanguage').addEventListener('click', () => setLanguage(language === 'fr' ? 'en' : 'fr'));
  candidates.addEventListener('change', () => {
    if (!events().some(event => event.id === active)) select(events()[0]?.id);
    else renderList();
  });
  $('playerShotSeek').addEventListener('input', event => {
    video.pause(); seekToFrame(nearestFrame(Number(event.target.value)));
  });
  $('playerShotStage').addEventListener('click', () => {
    $('playerShotStage').focus({preventScroll: true}); togglePlayback();
  });
  $('replayPlayerShot').addEventListener('click', () => { if (clip) { seekToFrame(clip.first); play(); } });
  for (const [id, offset] of [['previousPlayerShot', -1], ['nextPlayerShot', 1]]) {
    $(id).addEventListener('click', () => {
      const list = events(); const index = list.findIndex(event => event.id === active);
      select(list[index + offset]?.id);
    });
  }
  sync();
  return {isOpen, sync, reset, update, clampFrame, togglePlayback, keydown};
})();
