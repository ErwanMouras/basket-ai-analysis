/* Team shot maps count attempts at player ground positions, never airborne balls. */
window.teamShots = (() => {
  const canvas = $("teamShotCanvas");
  const cache = new WeakMap();
  let current = null;
  let maximum = 0;
  const number = (value) =>
    Number(value).toLocaleString(locale(), { maximumFractionDigits: 1 });
  function box() {
    const margin = 48;
    return {
      x: margin,
      y: margin,
      width: canvas.width - 2 * margin,
      height: canvas.height - 2 * margin,
    };
  }
  function resize() {
    const geometry = state.meta?.shots?.shot_maps?.court;
    if (!geometry || canvas.hidden) return;
    const height = $("teamPanel").clientHeight;
    canvas.style.maxWidth = `${Math.max(150, Math.min(360, ((height - 80) * geometry.length) / geometry.width))}px`;
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
    $("exportTeamShots").disabled = canvas.hidden;
    $("teamShotSummary").textContent =
      !team || !data?.available || !current
        ? t("teamShotUnavailable")
        : t("teamShotSummary", {
            count: current?.detected_count ?? 0,
            located: current?.located_count ?? 0,
          });
    const missing = [];
    if (current?.unlocated_count)
      missing.push(t("teamShotMissing", { count: current.unlocated_count }));
    if (data?.unassigned_probable_count)
      missing.push(
        t("teamShotUnassigned", { count: data.unassigned_probable_count }),
      );
    $("teamShotMissing").textContent = missing.join(" ");
    $("teamShotMissing").hidden = !missing.length;
    $("teamShotInspect").textContent = "";
    $("teamShotLegend").textContent = canvas.hidden ? "" : t("teamShotLegend");
    if (!available || canvas.hidden) return;
    const geometry = maps.court;
    canvas.width = 900;
    canvas.height =
      Math.round(((canvas.width - 96) * geometry.width) / geometry.length) + 96;
    canvas.setAttribute(
      "aria-label",
      `${t("teamShotTitle")} · ${team.name} · ${t("teamShotSummary", { count: current.detected_count, located: current.located_count })}`,
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
    maximum = Math.max(
      0,
      ...maps.maps.map((map) => HeatmapRenderer.maximum(cache.get(map))),
    );
    const field = cache.get(current);
    HeatmapRenderer.paint(ctx, bounds, field, maximum, false);
    $("teamShotLegend").textContent =
      `${t("teamShotLegend")} · 0 – ${number(maximum)} ${t("teamShotDensity")}`;
    HeatmapRenderer.court(ctx, bounds, geometry);
    for (const point of current.points) {
      const x = bounds.x + (point.x / geometry.length) * bounds.width;
      const y = bounds.y + (point.y / geometry.width) * bounds.height;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = "#171a1d";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#fff2cb";
      ctx.stroke();
    }
    ctx.fillStyle = "#d4cec6";
    ctx.font = "24px Arial";
    ctx.textAlign = "center";
    ctx.fillText(t("hmBasketAttack"), canvas.width / 2, canvas.height - 14);
    resize();
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
      ? `${t("teamShotHere", { count: near.length })} · ${near.map((point) => fmtTime(point.time_s)).join(" / ")}`
      : t("teamShotCell", { count: value });
  });
  canvas.addEventListener(
    "pointerleave",
    () => ($("teamShotInspect").textContent = ""),
  );
  $("exportTeamShots").addEventListener("click", () => {
    if (!current || canvas.hidden) return;
    const team = state.meta.roster.teams.find(
      (team) => team.team_id === state.teamId,
    );
    const output = document.createElement("canvas");
    output.width = canvas.width;
    output.height = canvas.height + 200;
    const ctx = output.getContext("2d");
    ctx.fillStyle = "#171a1d";
    ctx.fillRect(0, 0, output.width, output.height);
    ctx.fillStyle = "#f3f0ea";
    ctx.font = "bold 26px Arial";
    ctx.fillText(`${t("teamShotTitle")} · ${team.name}`, 30, 40, 840);
    ctx.font = "19px Arial";
    ctx.fillText(
      t("teamShotSummary", {
        count: current.detected_count,
        located: current.located_count,
      }),
      30,
      74,
      840,
    );
    ctx.drawImage(canvas, 0, 100);
    ctx.font = "17px Arial";
    ctx.fillText(
      `0 – ${number(maximum)} ${t("teamShotDensity")}`,
      30,
      output.height - 70,
      840,
    );
    ctx.fillText(t("teamShotExportScope"), 30, output.height - 44, 840);
    ctx.fillText(t("teamShotExportPositions"), 30, output.height - 18, 840);
    const a = document.createElement("a");
    a.download = `shot-map-${team.team_id.replace(/[^a-z0-9-]/gi, "-")}.png`;
    a.href = output.toDataURL("image/png");
    a.click();
  });
  new ResizeObserver(resize).observe($("teamPanel"));
  render();
  return { render };
})();
