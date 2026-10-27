/* Spatial analysis is separate from the frame/segmentation playback loop. */
function createHeatmapStudio({inline = false} = {}) {
  const dialog = document.getElementById("heatmapDialog");
  const root = inline ? document.getElementById("teamHeatmapStudio") : dialog;
  if (inline) {
    const layout = dialog.querySelector('.hm-layout').cloneNode(true);
    for (const node of layout.querySelectorAll('[id]')) node.id = `team-${node.id}`;
    for (const node of layout.querySelectorAll('[for], [aria-labelledby], [aria-controls]')) {
      for (const attr of ['for', 'aria-labelledby', 'aria-controls']) {
        if (node.hasAttribute(attr)) node.setAttribute(attr, node.getAttribute(attr).split(' ').map(id => `team-${id}`).join(' '));
      }
    }
    layout.querySelector('[data-heatmap-mode="comparison"]').remove();
    const maps = document.createElement('div');
    maps.className = 'team-map-grid';
    const presence = document.createElement('section');
    presence.className = 'team-presence-map';
    const title = document.createElement('h4');
    title.id = 'team-mapTitle';
    presence.setAttribute('aria-labelledby', title.id);
    presence.append(
      title,
      layout.querySelector('#team-heatmapStatus'),
      layout.querySelector('#team-heatmapCharts'),
      layout.querySelector('#team-heatmapPhaseSummary'),
    );
    maps.append(presence, document.getElementById('teamShotMap'));
    layout.querySelector('.hm-content').append(maps);
    root.append(layout);
  }
  const $ = id => document.getElementById(inline ? `team-${id}` : id);
  const isOpen = () => inline || dialog.open;
  let selectionKey = null;
  const modes = ["presence", "zones", "comparison"];
  const modeKeys = {
    presence: "hmPresence",
    zones: "hmZones",
    comparison: "hmComparison",
  };
  const densityCache = new WeakMap();
  let metadata = null;
  let metadataRun = null;
  let result = null;
  let mode = "presence";
  let phase = "offense";
  let request = 0;
  let opener = null;
  let busy = false;
  let statusKey = null;
  const number = (value, digits = 1) =>
    Number(value).toLocaleString(locale(), { maximumFractionDigits: digits });
  const share = (value, total) => (total > 0 ? (100 * value) / total : 0);
  const selectedSubject = () =>
    metadata?.subjects.find((item) => item.id === $("heatmapSubject").value);
  const phaseKey = (phase) =>
    ({ offense: "hmOffense", defense: "hmDefense" })[phase];
  const selectedPhase = () => phase;
  const selectedPeriod = () =>
    (metadata?.scoreboard?.periods || []).find(
      (_, index) => String(index) === $("heatmapPeriod").value,
    );
  const periodBounds = () =>
    selectedPeriod()
      ? {
          start_s: selectedPeriod().start_s,
          end_s: Math.min(metadata.bounds.end_s, selectedPeriod().end_s),
        }
      : metadata?.bounds;
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }
  function status(key) {
    statusKey = key;
    $("heatmapStatus").textContent = key ? t(key) : "";
    $("heatmapStatus").hidden = !key;
  }
  function clearCharts() {
    result = null;
    $("heatmapCharts").replaceChildren();
    renderPhases();
  }
  function subjectOptions(select, subjects, value) {
    select.replaceChildren();
    const teams = subjects.filter((item) => item.kind === "team");
    const players = subjects.filter((item) => item.kind === "player");
    for (const [label, items] of [
      [t("hmTeamOpen"), teams],
      [t("players"), players],
    ]) {
      if (!items.length) continue;
      const group = element("optgroup");
      group.label = label;
      for (const subject of items) {
        const team = metadata.subjects.find(
          (item) => item.kind === "team" && item.team_id === subject.team_id,
        );
        const suffix =
          subject.kind === "player"
            ? ` · ${team?.label || ""} #${subject.number}`
            : "";
        group.append(new Option(subject.label + suffix, subject.id));
      }
      select.append(group);
    }
    if (subjects.some((item) => item.id === value)) select.value = value;
  }
  const comparisonTeams = () =>
    metadata?.subjects.filter((item) => item.kind === "team") || [];
  const querySubject = () =>
    mode === "comparison" ? comparisonTeams()[0] : selectedSubject();

  function updateControls() {
    const comparing = mode === "comparison";
    $("heatmapSubjectField").hidden = inline || comparing;
    $("heatmapMatchup").hidden = !comparing;
    $("heatmapMatchup").textContent = comparisonTeams()
      .map((team) => team.label)
      .join(" / ");
    $("heatmapSmoothField").hidden = mode === "zones";
    $("heatmapRenderNote").hidden = mode === "zones";
  }
  function periodOptions(value = "") {
    const periods = metadata?.scoreboard?.periods || [];
    $("heatmapPeriodField").hidden = !periods.length;
    const select = $("heatmapPeriod");
    select.replaceChildren(new Option(t("allGamePeriods"), ""));
    periods.forEach((period, index) =>
      select.add(
        new Option(
          `${periodLabel(period.period)} · ${fmtTime(period.start_s)} – ${fmtTime(Math.min(metadata.bounds.end_s, period.end_s))}`,
          String(index),
        ),
      ),
    );
    select.value = value;
  }
  async function open(context = "court", trigger = document.activeElement) {
    if (!state.loaded || !state.name) return toast(t("hmNoRun"));
    opener = trigger;
    video.pause();
    mode = "presence";
    clearCharts();
    if (!inline && !dialog.open) dialog.showModal();
    const token = ++request;
    busy = true;
    status("hmLoading");
    try {
      const name = state.name;
      if (!metadata || metadataRun !== name) {
        const data = await json(`/api/heatmaps?${params({ name })}`);
        if (token !== request || name !== state.name || !isOpen()) return;
        metadata = data;
        metadataRun = name;
      }
      if (!metadata.available) {
        busy = false;
        status(metadata.reason === "geometry" ? "hmNoGeometry" : "hmNoRoster");
        translate();
        return;
      }
      const subject =
        context === "player"
          ? state.selected?.subject
          : context === "team"
            ? `team:${state.teamId}`
            : state.selected?.subject || `team:${state.teamId}`;
      subjectOptions($("heatmapSubject"), metadata.subjects, subject);
      phase = "offense";
      periodOptions();
      updateControls();
      busy = false;
      translate();
      await calculate();
    } catch (error) {
      if (token !== request) return;
      busy = false;
      status("hmError");
    }
  }

  async function calculate() {
    updateModeButtons();
    updateIntro();
    if (!metadata?.available || !isOpen()) return;
    const bounds = periodBounds();
    const subject = querySubject();
    if (!bounds || !subject) {
      clearCharts();
      status("hmNoData");
      return;
    }
    if (mode === "comparison" && comparisonTeams().length !== 2) {
      ++request;
      busy = false;
      clearCharts();
      status("hmNoPair");
      return;
    }
    const token = ++request;
    busy = true;
    clearCharts();
    status("hmLoading");
    const name = state.name;
    try {
      const data = await json(
        `/api/heatmaps?${params({
          name,
          subject: subject.id,
          phase: selectedPhase(),
          start: bounds.start_s,
          end: bounds.end_s,
          mode,
        })}`,
      );
      if (token !== request || name !== state.name || !isOpen()) return;
      result = data;
      busy = false;
      status(null);
      render();
    } catch (error) {
      if (token !== request || !isOpen()) return;
      busy = false;
      status("hmError");
    }
  }
  function updateModeButtons() {
    root.querySelectorAll("[data-heatmap-mode]").forEach((button) => {
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.heatmapMode === mode),
      );
      button.disabled = !metadata?.available;
    });
    if (inline) {
      const compare = document.getElementById('compareTeamHeatmaps');
      compare.disabled = !metadata?.available || comparisonTeams().length !== 2;
      compare.setAttribute('aria-pressed', String(mode === 'comparison'));
    }
  }
  function updateIntro() {
    $("heatmapModeTitle").textContent = t(modeKeys[mode]);
    if (inline) document.getElementById('team-mapTitle').textContent = t(modeKeys[mode]);
    const subject = selectedSubject();
    $("heatmapScope").textContent = [
      mode === "comparison"
        ? comparisonTeams()
            .map((team) => team.label)
            .join(" / ")
        : subject?.label,
      selectedPeriod() ? periodLabel(selectedPeriod().period) : null,
      t(phaseKey(selectedPhase())) + ` · ${t("hmEstimated")}`,
    ]
      .filter(Boolean)
      .join(" / ");
    if (!inline) {
      $("heatmapLanguage").textContent = language === "fr" ? "EN" : "FR";
      $("heatmapLanguage").setAttribute("aria-label", t("hmCloseLanguage"));
    }
  }
  function translate() {
    if (!isOpen()) return;
    if (metadata?.available) {
      subjectOptions(
        $("heatmapSubject"),
        metadata.subjects,
        $("heatmapSubject").value,
      );
      periodOptions($("heatmapPeriod").value);
      updateControls();
    }
    const disabled = !metadata?.available;
    for (const id of [
      "heatmapSubject",
      "heatmapPeriod",
    ])
      $(id).disabled = disabled;
    root
      .querySelectorAll("[data-heatmap-mode]")
      .forEach((button) => (button.disabled = disabled));
    updateModeButtons();
    updateIntro();
    renderPhases();
    if (statusKey) status(statusKey);
    if (result && !busy) render();
  }

  function renderPhases() {
    const summary = $("heatmapPhaseSummary");
    summary.setAttribute("role", "group");
    summary.setAttribute("aria-label", t("hmPhase"));
    for (const value of ["offense", "defense"]) {
      let button = summary.querySelector(`[data-heatmap-phase="${value}"]`);
      if (!button) {
        button = element("button", `hm-phase-chip hm-phase-${value}`);
        button.type = "button";
        button.dataset.heatmapPhase = value;
        button.append(element("span"), element("strong"));
        button.addEventListener("click", () => changePhase(value));
        summary.append(button);
      }
      button.disabled = !metadata?.available;
      button.setAttribute("aria-pressed", String(phase === value));
      button.querySelector("span").textContent = t(phaseKey(value));
      const seconds = mode === 'comparison' ? null : result?.phase_analysis?.[`${value}_s`];
      button.querySelector("strong").textContent = seconds == null ? '' : `${number(seconds)} s`;
    }
  }
  function changePhase(value) {
    if (!["offense", "defense"].includes(value)) return;
    phase = value;
    calculate();
  }
  function render() {
    if (!result) return;
    renderPhases();
    const charts = $("heatmapCharts");
    charts.replaceChildren();
    charts.classList.toggle("hm-multiple", result.maps.length > 1);
    charts.classList.toggle("hm-comparison", mode === "comparison");
    const continuous = $("heatmapSmooth").checked && mode !== "zones";
    charts.classList.toggle("hm-continuous", continuous);
    const fields = result.maps.map((map) => {
      if (!continuous) return null;
      if (!densityCache.has(map)) {
        densityCache.set(
          map,
          HeatmapRenderer.density(
            map.grid.map((value) => share(value, map.total)),
            map.nx,
            map.ny,
            result.court.length,
            result.court.width,
          ),
        );
      }
      return densityCache.get(map);
    });
    const maximum = Math.max(
      0,
      ...result.maps.map((map, index) =>
        fields[index]
          ? HeatmapRenderer.maximum(fields[index])
          : Math.max(0, ...map.grid.map((value) => share(value, map.total))),
      ),
    );
    result.maps.forEach((map, index) => {
      const title =
        mode === "comparison"
          ? `${index ? "B" : "A"} · ${map.subject.label}`
          : map.subject.label;
      charts.append(mapCard(map, title, maximum, fields[index]));
    });
    if (mode === "comparison") {
      const [first, second] = result.maps;
      const available = first.total > 0 && second.total > 0;
      const grid = first.grid.map((value, index) =>
        available
          ? share(value, first.total) - share(second.grid[index], second.total)
          : 0,
      );
      const difference = {
        ...first,
        grid,
        zones: first.zones.map((value, index) =>
          available
            ? share(value, first.total) -
              share(second.zones[index], second.total)
            : 0,
        ),
        total: available ? 1 : 0,
        difference: true,
        available,
      };
      const field = continuous
        ? HeatmapRenderer.subtract(fields[0], fields[1], available)
        : null;
      charts.append(
        mapCard(
          difference,
          t("hmDifference"),
          field
            ? HeatmapRenderer.maximum(field)
            : Math.max(0, ...grid.map(Math.abs)),
          field,
        ),
      );
    }
  }
  function mapCard(map, title, maximum, field) {
    const card = element("article", "hm-map-card");
    const time = `${fmtTime(map.start_s)} – ${fmtTime(map.end_s)}`;
    const legendMaximum =
      mode === "zones"
        ? Math.max(0, ...map.zones.map((value) => share(value, map.total)))
        : maximum;
    if (mode === "comparison") card.append(element("h4", "hm-map-title", title));
    const mapCanvas = element("canvas", "hm-canvas");
    mapCanvas.width = 1440;
    mapCanvas.height =
      Math.round(((mapCanvas.width - 48) * result.court.width) / result.court.length) + 48;
    mapCanvas.setAttribute("role", "img");
    mapCanvas.setAttribute(
      "aria-label",
      `${t(modeKeys[mode])} · ${title} · ${time}`,
    );
    paintMap(mapCanvas, map, maximum, field);
    mapCanvas.dataset.rendering = field ? "contours" : "cells";
    const displayUnit = field
      ? map.difference
        ? "pp/m²"
        : "%/m²"
      : map.difference
        ? "pp"
        : "%";
    const legendKey = map.difference
      ? field
        ? "hmDensityDifferenceLegend"
        : "hmDifferenceLegend"
      : mode === "zones"
        ? "hmZonesLegend"
        : field
          ? "hmDensityLegend"
          : "hmLegend";
    card.append(mapCanvas);
    const empty = map.difference ? !map.available : map.total <= 0;
    if (empty)
      card.append(
        element(
          "p",
          "hm-empty",
          t(map.difference ? "hmNoDifference" : "hmNoData"),
        ),
      );
    const legend = element("div", "hm-map-legend");
    if (map.difference) {
      legend.append(
        element("span", "", `−${number(maximum)} ${displayUnit}`),
        element("i", "hm-gradient hm-diverging"),
        element("span", "", `+${number(maximum)} ${displayUnit}`),
      );
      card.append(element("p", "hm-legend-caption", t(legendKey)));
    } else {
      legend.append(
        element("span", "", "0"),
        element("i", "hm-gradient"),
        element("span", "", `${number(legendMaximum)} ${displayUnit}`),
      );
      card.append(element("p", "hm-legend-caption", t(legendKey)));
    }
    card.append(legend);
    return card;
  }
  function courtBox(canvas) {
    const geometry = result.court;
    const margin = 24;
    const width = Math.min(
      canvas.width - margin * 2,
      ((canvas.height - margin * 2) * geometry.length) / geometry.width,
    );
    const height = (width * geometry.width) / geometry.length;
    return {
      x: (canvas.width - width) / 2,
      y: (canvas.height - height) / 2,
      width,
      height,
    };
  }
  function paintMap(canvas, map, maximum, field) {
    const ctx = canvas.getContext("2d");
    const box = courtBox(canvas),
      geometry = result.court;
    ctx.fillStyle = "#171a1d";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#171c20";
    ctx.fillRect(box.x, box.y, box.width, box.height);
    const values = mode === "zones" ? map.zones : map.grid;
    const nx = mode === "zones" ? 3 : map.nx,
      ny = mode === "zones" ? 3 : map.ny;
    const normalized = values.map((value) =>
      map.difference ? value : share(value, map.total),
    );
    const max = mode === "zones" ? Math.max(0, ...normalized) : maximum;
    if (field) {
      HeatmapRenderer.paint(ctx, box, field, maximum, map.difference);
    } else {
      const raster = document.createElement("canvas");
      raster.width = nx;
      raster.height = ny;
      const paint = raster.getContext("2d");
      normalized.forEach((value, index) => {
        const amount = max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
        paint.fillStyle = color(amount, map.difference && value < 0);
        paint.fillRect(index % nx, Math.floor(index / nx), 1, 1);
      });
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(raster, box.x, box.y, box.width, box.height);
    }
    HeatmapRenderer.court(ctx, box, geometry);
    if (mode === "zones" || !$("heatmapSmooth").checked) {
      ctx.strokeStyle = "#eee0bf40";
      ctx.lineWidth = 1;
      for (let i = 1; i < nx; i++) {
        ctx.beginPath();
        ctx.moveTo(box.x + (i * box.width) / nx, box.y);
        ctx.lineTo(box.x + (i * box.width) / nx, box.y + box.height);
        ctx.stroke();
      }
      for (let i = 1; i < ny; i++) {
        ctx.beginPath();
        ctx.moveTo(box.x, box.y + (i * box.height) / ny);
        ctx.lineTo(box.x + box.width, box.y + (i * box.height) / ny);
        ctx.stroke();
      }
    }
    if (mode === "zones") {
      ctx.font = "bold 32px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      normalized.forEach((value, index) => {
        const xx = box.x + (((index % 3) + 0.5) * box.width) / 3,
          yy = box.y + ((Math.floor(index / 3) + 0.5) * box.height) / 3;
        ctx.fillStyle = "#111214d9";
        ctx.fillRect(xx - 55, yy - 23, 110, 46);
        ctx.fillStyle = "#f3f0ea";
        ctx.fillText(map.total ? `${number(value)} %` : "—", xx, yy);
      });
    }
  }
  function color(amount, blue) {
    const stops = blue
      ? [
          [23, 28, 32],
          [60, 123, 182],
          [193, 228, 253],
        ]
      : [
          [23, 28, 32],
          [209, 97, 43],
          [255, 229, 162],
        ];
    const part = amount < 0.55 ? 0 : 1;
    const mix = part ? (amount - 0.55) / 0.45 : amount / 0.55;
    const rgb = stops[part].map((value, index) =>
      Math.round(value + mix * (stops[part + 1][index] - value)),
    );
    return `rgb(${rgb.join(",")})`;
  }

  if (!inline) document
    .querySelectorAll("[data-heatmap-context]")
    .forEach((button) =>
      button.addEventListener("click", () =>
        open(button.dataset.heatmapContext, button),
      ),
    );
  let singleMode = 'presence';
  function changeMode(next) {
    if (!modes.includes(next)) return;
    if (next !== 'comparison') singleMode = next;
    mode = next;
    updateControls();
    calculate();
  }
  root.querySelectorAll("[data-heatmap-mode]").forEach(button =>
    button.addEventListener('click', () => changeMode(button.dataset.heatmapMode)),
  );
  if (inline) document.getElementById('compareTeamHeatmaps').addEventListener('click', () =>
    changeMode(mode === 'comparison' ? singleMode : 'comparison'),
  );
  $("heatmapSubject").addEventListener("change", calculate);
  $("heatmapPeriod").addEventListener("change", calculate);
  $("heatmapSmooth").addEventListener("change", render);
  if (!inline) {
    $("heatmapLanguage").addEventListener("click", () => setLanguage(language === "fr" ? "en" : "fr"));
    $("closeHeatmaps").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
      ++request;
      busy = false;
      if (opener?.isConnected) opener.focus();
    });
  }
  function sync() {
    if (!inline || state.mode !== 'statistics' || state.statsView !== 'team') return;
    const key = JSON.stringify([state.name, state.teamId, state.loaded, Boolean(state.meta?.roster)]);
    if (selectionKey === key) return;
    selectionKey = key;
    if (!state.loaded || !state.name || !state.teamId) {
      ++request;
      metadata = null;
      busy = false;
      clearCharts();
      status(state.loaded ? 'hmNoRoster' : 'hmNoRun');
      translate();
      return;
    }
    if (!metadata || metadataRun !== state.name) {
      open('team');
      return;
    }
    if (!metadata.available) return;
    $("heatmapSubject").value = `team:${state.teamId}`;
    translate();
    calculate();
  }
  function reset() {
    ++request;
    metadata = null;
    metadataRun = null;
    selectionKey = null;
    busy = false;
    clearCharts();
    if (!inline && dialog.open) dialog.close();
  }
  updateControls();
  if (inline) translate();
  return { open, translate, reset, sync };
}
window.heatmapStudio = createHeatmapStudio();
window.teamHeatmaps = createHeatmapStudio({inline: true});
translateDocument();
window.teamHeatmaps.sync();
