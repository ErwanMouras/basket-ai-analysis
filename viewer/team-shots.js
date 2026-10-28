/* Team shot maps count attempts at player ground positions, never airborne balls. */
window.teamShots = (() => {
  const canvas = $("teamShotCanvas");
  const cache = new WeakMap();
  let current = null;
  function box() {
    const margin = 24;
    return {
      x: margin,
      y: margin,
      width: canvas.width - 2 * margin,
      height: canvas.height - 2 * margin,
    };
  }
  function render() {
    const data = state.meta?.shots;
    const team = state.meta?.roster?.teams.find(
      (team) => team.team_id === state.teamId,
    );
    const maps = data?.shot_maps;
    current = maps?.maps.find((map) => map.team_id === state.teamId) || null;
    const available = Boolean(
      team && data?.available && maps?.available && current,
    );
    $("teamShotMap").hidden = !state.loaded;
    canvas.hidden = !available || !current.located_count;
    $("teamShotSummary").textContent =
      !team || !data?.available || !current
        ? t("teamShotUnavailable")
        : t("teamShotSummary", {
            count: current?.detected_count ?? 0,
            located: current?.located_count ?? 0,
            probable: current?.probable_count ?? 0,
            candidates: current?.candidate_count ?? 0,
          });
    const missing = [];
    if (current?.unlocated_count) {
      missing.push(t("teamShotMissing", { count: current.unlocated_count }));
      const grouped = {};
      for (const [reason, count] of Object.entries(current.unlocated_reasons || {})) {
        const key = reason === 'direction_unavailable' ? 'teamShotReasonDirection'
          : reason === 'outside_attacking_half' ? 'teamShotReasonHalf'
          : reason === 'outside_court' ? 'teamShotReasonCourt' : 'teamShotReasonPosition';
        grouped[key] = (grouped[key] || 0) + count;
      }
      for (const [key, count] of Object.entries(grouped)) missing.push(t(key, {count}));
    }
    $("teamShotMissing").textContent = missing.join(" ");
    $("teamShotMissing").hidden = !missing.length;
    $("teamShotInspect").textContent = "";
    if (!available || canvas.hidden) return;
    const geometry = maps.court;
    canvas.width = 1440;
    canvas.height =
      Math.round(((canvas.width - 48) * geometry.width) / geometry.length) + 48;
    canvas.setAttribute(
      "aria-label",
      `${t("teamShotTitle")} · ${team.name} · ${t("teamShotSummary", { count: current.detected_count, located: current.located_count, probable: current.probable_count, candidates: current.candidate_count })}`,
    );
    const ctx = canvas.getContext("2d");
    const bounds = box();
    ctx.fillStyle = "#171a1d";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (const map of maps.maps) {
      if (!cache.has(map))
        cache.set(
          map,
          HeatmapRenderer.density(
            map.grid,
            map.nx,
            map.ny,
            geometry.length,
            geometry.width,
          ),
        );
    }
    const maximum = Math.max(
      0,
      ...maps.maps.map((map) => HeatmapRenderer.maximum(cache.get(map))),
    );
    const field = cache.get(current);
    HeatmapRenderer.paint(ctx, bounds, field, maximum, false);
    HeatmapRenderer.court(ctx, bounds, geometry);
    for (const point of current.points) {
      const x = bounds.x + (point.x / geometry.length) * bounds.width;
      const y = bounds.y + (point.y / geometry.width) * bounds.height;
      ctx.beginPath();
      ctx.arc(x, y, 18, 0, Math.PI * 2);
      ctx.fillStyle = point.status === "candidate" ? "#171a1d" : "#e4c2ff";
      ctx.fill();
      ctx.lineWidth = 5;
      ctx.strokeStyle = "#e4c2ff";
      ctx.stroke();
    }
  }
  canvas.addEventListener("pointermove", (event) => {
    if (!current || canvas.hidden) return;
    const geometry = state.meta.shots.shot_maps.court,
      bounds = box(),
      rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
    if (
      x < bounds.x ||
      x > bounds.x + bounds.width ||
      y < bounds.y ||
      y > bounds.y + bounds.height
    ) {
      $("teamShotInspect").textContent = "";
      return;
    }
    const px = ((x - bounds.x) / bounds.width) * geometry.length;
    const py = ((y - bounds.y) / bounds.height) * geometry.width;
    const near = current.points.filter(
      (point) => Math.hypot(point.x - px, point.y - py) < 0.65,
    );
    const value =
      current.grid[
        Math.min(
          current.ny - 1,
          Math.floor(((y - bounds.y) / bounds.height) * current.ny),
        ) *
          current.nx +
          Math.min(
            current.nx - 1,
            Math.floor(((x - bounds.x) / bounds.width) * current.nx),
          )
      ];
    $("teamShotInspect").textContent = near.length
      ? `${t("teamShotHere", { count: near.length })} · ${near.map((point) => `${fmtTime(point.time_s)} (${t(point.status === 'candidate' ? 'shotCandidate' : 'shotProbable')})`).join(" / ")}`
      : t("teamShotCell", { count: value });
  });
  canvas.addEventListener(
    "pointerleave",
    () => ($("teamShotInspect").textContent = ""),
  );
  render();
  return { render };
})();
