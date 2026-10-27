/* Shot proposals are computed once by the server, independently of playback. */
window.shotStudio = (() => {
  const panel = $("shotsPanel");
  const candidates = $("shotsCandidates");
  let active = null;
  const events = () =>
    (state.meta?.shots?.events || []).filter(
      (event) => candidates.checked || event.status === "probable",
    );
  const label = (event) =>
    t(event.status === "probable" ? "shotProbable" : "shotCandidate");
  function shooter(event) {
    const identity = event.shooter || {};
    const teams = state.meta?.roster?.teams || [];
    const team = teams.find((team) => team.team_id === identity.team_id);
    const player = team?.players.find(
      (player) =>
        `roster:${team.team_id}:${player.number}` === identity.subject_id,
    );
    return player
      ? `${player.name} · #${player.number}`
      : team?.name || t("shotUnassigned");
  }
  function reasons(event) {
    const e = event.evidence;
    return [
      e.release_observations >= 2 ? t("shotRelease") : t("shotReleaseHidden"),
      e.arc ? t("shotArc") : t("shotPartialArc"),
      e.rim_approach ? t("shotRim") : t("shotRimUnknown"),
      e.camera_compensated ? t("shotCamera") : t("shotRawCamera"),
    ].join(" · ");
  }
  function review(event) {
    video.pause();
    seekToFrame(nearestFrame(event.start_s));
  }
  function render() {
    const data = state.meta?.shots;
    panel.hidden = !state.loaded;
    $("shotsLegend").hidden = !data?.events?.length;
    $("shotsSummary").textContent = !data?.available
      ? t("shotsUnavailable")
      : data.events.length
        ? t("shotsSummary", {
            probable: data.probable_count,
            candidates: data.candidate_count,
          })
        : t("shotsNone");
    $("shotsHelp").textContent = t(
      data?.available ? "shotsHelp" : "shotsUnavailableHelp",
    );
    $("shotsCandidatesField").hidden = !data?.candidate_count;
    $("shotsList").replaceChildren();
    document
      .querySelectorAll("#timelineMarkers .shot")
      .forEach((node) => node.remove());
    for (const event of events()) {
      const marker = document.createElement("button");
      marker.className = `shot ${event.status}`;
      marker.type = "button";
      marker.style.left = `${(100 * event.time_s) / Math.max(duration(), 0.001)}%`;
      marker.title = `${fmtTime(event.time_s)} · ${label(event)} · ${shooter(event)}`;
      marker.setAttribute("aria-label", marker.title);
      marker.addEventListener("click", () => review(event));
      $("timelineMarkers").append(marker);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `shot-row ${event.status}`;
      button.dataset.shotId = event.id;
      const time = document.createElement("span");
      time.className = "shot-time";
      time.textContent = fmtTime(event.time_s);
      const description = document.createElement("span");
      const name = document.createElement("strong");
      name.textContent = shooter(event);
      const evidence = document.createElement("small");
      evidence.textContent = reasons(event);
      description.append(name, evidence);
      const badge = document.createElement("span");
      badge.className = "shot-status";
      badge.textContent = label(event);
      button.append(time, description, badge);
      button.title = t("shotReviewHelp", {
        start: fmtTime(event.release_window_s[0]),
        end: fmtTime(event.release_window_s[1]),
      });
      button.addEventListener("click", () => review(event));
      $("shotsList").append(button);
    }
    active = null;
    update(video.currentTime || 0);
  }
  function update(time) {
    const event = events().find(
      (event) => event.time_s <= time && time <= event.end_s,
    );
    const id = event?.id || "";
    if (id === active) return;
    active = id;
    document.querySelectorAll(".shot-row").forEach((node) => {
      node.classList.toggle("active", node.dataset.shotId === id);
      node.setAttribute(
        "aria-current",
        node.dataset.shotId === id ? "true" : "false",
      );
    });
  }
  candidates.addEventListener("change", render);
  render();
  return { render, update };
})();
