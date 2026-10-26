/* Spatial analysis is separate from the frame/segmentation playback loop. */
window.heatmapStudio = (() => {
  const dialog = $("heatmapDialog");
  const modes = ["presence", "zones", "evolution", "comparison"];
  const modeKeys = {
    presence: "hmPresence",
    zones: "hmZones",
    comparison: "hmComparison",
    evolution: "hmEvolution",
  };
  const densityCache = new WeakMap();
  let metadata = null;
  let metadataRun = null;
  let result = null;
  let mode = "presence";
  let request = 0;
  let opener = null;
  let busy = false;
  let statusKey = null;
  const number = (value, digits = 1) =>
    Number(value).toLocaleString(locale(), { maximumFractionDigits: digits });
  const share = (value, total) => (total > 0 ? (100 * value) / total : 0);
  const unitLabel = (unit) =>
    t(
      unit === "player_s"
        ? "hmPlayerSeconds"
        : unit === "m"
          ? "hmMetres"
          : "hmSeconds",
    );
  const selectedSubject = () =>
    metadata?.subjects.find((item) => item.id === $("heatmapSubject").value);
  const phaseKey = (phase) =>
    ({ offense: "hmOffense", defense: "hmDefense" })[phase];
  const selectedPhase = () => $("heatmapPhase").value;
  const selectedPassage = () =>
    metadata?.phase_analysis?.episodes.find(
      (item) => String(item.id) === $("heatmapPassage").value,
    );
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
  const scopeBounds = () => selectedPassage() || periodBounds();
  const matchingRole = (episode) =>
    (!selectedPeriod() ||
      (episode.period === selectedPeriod().period &&
        episode.start_s < periodBounds().end_s &&
        episode.end_s > periodBounds().start_s)) &&
    (mode === "comparison" ||
      (selectedPhase() === "offense"
        ? episode.attacking_team_id === selectedSubject()?.team_id
        : selectedPhase() === "defense" &&
          episode.attacking_team_id !== selectedSubject()?.team_id));

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
    $("heatmapTables").replaceChildren();
    $("heatmapData").hidden = true;
    $("heatmapInspect").textContent = "";
    $("heatmapPhaseSummary").replaceChildren();
    $("heatmapOrientation").textContent = "";
    $("heatmapPassages").hidden = true;
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
    $("heatmapSubjectField").hidden = comparing;
    $("heatmapPassageField").hidden = comparing;
    $("heatmapMatchup").hidden = !comparing;
    $("heatmapMatchup").textContent = comparisonTeams()
      .map((team) => team.label)
      .join(" / ");
    $("heatmapSmoothField").hidden = mode === "zones";
    $("heatmapRenderNote").hidden = mode === "zones";
  }
  function passageOptions(value = "") {
    const select = $("heatmapPassage");
    select.replaceChildren(new Option(t("hmAllPassages"), ""));
    for (const passage of metadata?.phase_analysis?.episodes || []) {
      if (!matchingRole(passage)) continue;
      select.add(
        new Option(
          `${passage.period ? `${periodLabel(passage.period)} · ` : ""}${fmtTime(passage.start_s)} – ${fmtTime(passage.end_s)}${passage.orientation_ready ? "" : ` · ${t("hmNoDirectionShort")}`}`,
          passage.id,
        ),
      );
    }
    select.value = [...select.options].some(
      (option) => option.value === String(value),
    )
      ? value
      : "";
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
  function resetRange() {
    const bounds = scopeBounds();
    if (!bounds) return;
    for (const id of ["heatmapStart", "heatmapEnd"]) {
      $(id).min = bounds.start_s;
      $(id).max = bounds.end_s;
    }
    $("heatmapStart").value = numberForInput(bounds.start_s);
    $("heatmapEnd").value = numberForInput(bounds.end_s);
  }
  const numberForInput = (value) => String(Number(value.toFixed(6)));

  async function open(context = "court", trigger = document.activeElement) {
    if (!state.loaded || !state.name) return toast(t("hmNoRun"));
    opener = trigger;
    video.pause();
    mode = "presence";
    clearCharts();
    if (!dialog.open) dialog.showModal();
    const token = ++request;
    busy = true;
    status("hmLoading");
    try {
      const name = state.name;
      if (!metadata || metadataRun !== name) {
        const data = await json(`/api/heatmaps?${params({ name })}`);
        if (token !== request || name !== state.name || !dialog.open) return;
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
      $("heatmapPhase").value = "offense";
      periodOptions();
      passageOptions();
      resetRange();
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
    if (!metadata?.available || !dialog.open) return;
    const bounds = scopeBounds();
    const subject = querySubject();
    if (!bounds || !subject) {
      clearCharts();
      status("hmNoData");
      return;
    }
    const rawStart = $("heatmapStart").value,
      rawEnd = $("heatmapEnd").value;
    const a = Math.max(bounds.start_s, Number(rawStart));
    const b = Math.min(bounds.end_s, Number(rawEnd));
    if (
      !rawStart ||
      !rawEnd ||
      !Number.isFinite(a) ||
      !Number.isFinite(b) ||
      b <= a
    ) {
      ++request;
      busy = false;
      clearCharts();
      status("hmBadRange");
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
          ...(selectedPassage() ? { episode: selectedPassage().id } : {}),
          start: a,
          end: b,
          mode,
        })}`,
      );
      if (token !== request || name !== state.name || !dialog.open) return;
      result = data;
      busy = false;
      status(null);
      render();
    } catch (error) {
      if (token !== request || !dialog.open) return;
      busy = false;
      status("hmError");
    }
  }
  function updateModeButtons() {
    document.querySelectorAll("[data-heatmap-mode]").forEach((button) => {
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.heatmapMode === mode),
      );
      button.disabled = !metadata?.available;
    });
  }
  function updateIntro() {
    $("heatmapModeTitle").textContent = t(modeKeys[mode]);
    $("heatmapExplanation").textContent = t(`${modeKeys[mode]}Help`);
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
    $("heatmapLanguage").textContent = language === "fr" ? "EN" : "FR";
    $("heatmapLanguage").setAttribute("aria-label", t("hmCloseLanguage"));
  }
  function translate() {
    if (!dialog.open) return;
    if (metadata?.available) {
      subjectOptions(
        $("heatmapSubject"),
        metadata.subjects,
        $("heatmapSubject").value,
      );
      passageOptions($("heatmapPassage").value);
      periodOptions($("heatmapPeriod").value);
      updateControls();
    }
    const disabled = !metadata?.available;
    for (const id of [
      "heatmapSubject",
      "heatmapPhase",
      "heatmapPassage",
      "heatmapPeriod",
      "heatmapStart",
      "heatmapEnd",
    ])
      $(id).disabled = disabled;
    document
      .querySelectorAll("[data-heatmap-mode]")
      .forEach((button) => (button.disabled = disabled));
    updateModeButtons();
    updateIntro();
    if (statusKey) status(statusKey);
    if (result && !busy) render();
  }

  function renderPhases() {
    const summary = $("heatmapPhaseSummary");
    summary.replaceChildren();
    const durations = result.phase_analysis;
    summary.hidden = mode === "comparison";
    if (!durations) return;
    summary.setAttribute("role", "group");
    summary.setAttribute("aria-label", t("hmPhaseBreakdown"));
    for (const phase of ["offense", "defense"]) {
      const button = element("button", `hm-phase-chip hm-phase-${phase}`);
      button.type = "button";
      button.setAttribute("aria-pressed", String(selectedPhase() === phase));
      button.append(
        element("span", "", t(phaseKey(phase))),
        element("strong", "", `${number(durations[phase + "_s"])} s`),
      );
      button.addEventListener("click", () => changePhase(phase));
      summary.append(button);
    }
    const missing =
      durations.orientation_missing_s > 0.05
        ? ` · ${t("hmDirectionMissing", { value: number(durations.orientation_missing_s) })}`
        : "";
    $("heatmapOrientation").textContent =
      t(selectedPhase() === "offense" ? "hmAttackBasket" : "hmDefendBasket") +
      missing +
      (metadata.phase_analysis.period_directions?.length
        ? ` · ${t("hmQuarterLock")}`
        : "");
    const bounds = result.scope;
    const passages = metadata.phase_analysis.episodes.filter(matchingRole);
    const list = $("heatmapPassagesList");
    list.replaceChildren();
    for (const passage of passages) {
      const a = Math.max(bounds.start_s, passage.start_s),
        b = Math.min(bounds.end_s, passage.end_s);
      if (
        b - a < 0.2 ||
        (selectedPassage() && passage.id !== selectedPassage().id)
      )
        continue;
      const button = element("button", "hm-passage-review");
      button.type = "button";
      const team =
        mode === "comparison"
          ? comparisonTeams().find(
              (item) =>
                (item.team_id === passage.attacking_team_id) ===
                (selectedPhase() === "offense"),
            )
          : null;
      const label = [team?.label, `${fmtTime(a)} – ${fmtTime(b)}`]
        .filter(Boolean)
        .join(" · ");
      button.append(
        element("span", "", label),
        element("small", "", t("hmReviewVideo")),
      );
      button.addEventListener("click", () => {
        dialog.close();
        video.pause();
        seekToFrame(nearestFrame(a + 0.01));
      });
      list.append(button);
    }
    if (!list.children.length) list.append(element("p", "", t("hmNoPassages")));
    $("heatmapPassagesTitle").textContent = t("hmReviewPassages");
    $("heatmapPassages").hidden = false;
    if ($("heatmapPassages").dataset.phase !== selectedPhase())
      $("heatmapPassages").open = false;
    $("heatmapPassages").dataset.phase = selectedPhase();
  }
  function changePhase(phase) {
    if (!["offense", "defense"].includes(phase)) return;
    $("heatmapPhase").value = phase;
    $("heatmapPhase").focus();
    passageOptions();
    resetRange();
    calculate();
  }
  function render() {
    if (!result) return;
    renderPhases();
    const charts = $("heatmapCharts");
    charts.replaceChildren();
    const tables = $("heatmapTables");
    tables.replaceChildren();
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
        mode === "evolution"
          ? t(index ? "hmSecond" : "hmFirst")
          : mode === "comparison"
            ? `${index ? "B" : "A"} · ${map.subject.label}`
            : map.subject.label;
      charts.append(mapCard(map, title, maximum, fields[index]));
      tables.append(zoneTable(map, title));
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
      tables.append(zoneTable(difference, t("hmDifference")));
    }
    $("heatmapData").hidden = !result.maps.some((map) => map.total > 0);
    $("heatmapInspect").textContent = t("hmHover");
  }
  function stat(label, value, hint, unit) {
    const item = element("div", "hm-stat");
    const name = element("span", "", t(label));
    if (hint) name.title = t(hint);
    const output = element("strong", "", value);
    if (unit) output.append(element("small", "", ` ${unit}`));
    item.append(name, output);
    return item;
  }
  function mapCard(map, title, maximum, field) {
    const card = element("article", "hm-map-card");
    const heading = element("div", "hm-map-heading");
    const exportButton = element("button", "hm-export", t("hmExport"));
    exportButton.type = "button";
    exportButton.setAttribute("aria-label", `${t("hmExportLabel")} · ${title}`);
    const time = `${fmtTime(map.start_s)} – ${fmtTime(map.end_s)}`;
    const legendMaximum =
      mode === "zones"
        ? Math.max(0, ...map.zones.map((value) => share(value, map.total)))
        : maximum;
    const labels = element("div");
    labels.append(
      element("h4", "", title),
      element("span", "hm-map-period", time),
    );
    heading.append(labels, exportButton);
    card.append(heading);
    const mapCanvas = element("canvas", "hm-canvas");
    mapCanvas.width = 1100;
    mapCanvas.height =
      Math.round((1010 * result.court.width) / result.court.length) + 90;
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
    if (!map.difference) {
      const stats = element("div", "hm-map-stats");
      stats.append(
        stat("hmTotal", number(map.total), null, unitLabel(map.unit)),
        stat(
          "hmCoverage",
          map.coverage == null ? "—" : number(map.coverage * 100, 0),
          "hmCoverageHelp",
          map.coverage == null ? null : "%",
        ),
        stat("hmVideoTime", number(map.half_court_video_s), "hmVideoHelp", "s"),
      );
      card.insertBefore(stats, mapCanvas);
      if (map.outside_half_s > 0.05)
        card.append(
          element(
            "p",
            "hm-measure-note",
            t("hmOutsideHalf", {
              value: number(map.outside_half_s),
              unit: unitLabel(map.unit),
            }),
          ),
        );
    } else {
      const stats = element("div", "hm-map-stats hm-difference-stats");
      stats.append(
        stat(
          "hmMaxDifference",
          map.available ? `±${number(maximum)}` : "—",
          null,
          map.available ? displayUnit : null,
        ),
      );
      card.insertBefore(stats, mapCanvas);
    }
    let hovered = null;
    mapCanvas.addEventListener("pointermove", (event) => {
      if (empty) return;
      const bounds = mapCanvas.getBoundingClientRect();
      const px =
        ((event.clientX - bounds.left) / bounds.width) * mapCanvas.width;
      const py =
        ((event.clientY - bounds.top) / bounds.height) * mapCanvas.height;
      const box = courtBox(mapCanvas);
      if (
        px < box.x ||
        px > box.x + box.width ||
        py < box.y ||
        py > box.y + box.height
      )
        return;
      const nx = mode === "zones" ? 3 : map.nx,
        ny = mode === "zones" ? 3 : map.ny;
      const x = Math.min(nx - 1, Math.floor(((px - box.x) / box.width) * nx));
      const y = Math.min(ny - 1, Math.floor(((py - box.y) / box.height) * ny));
      const key = `${x}:${y}`;
      if (hovered === key) return;
      hovered = key;
      const value = (mode === "zones" ? map.zones : map.grid)[y * nx + x];
      $("heatmapInspect").textContent = t(
        map.difference ? "hmDeltaCell" : "hmCell",
        {
          x: x + 1,
          y: y + 1,
          value: number(value, 2),
          unit: unitLabel(map.unit),
          share: number(share(value, map.total), 2),
        },
      );
    });
    exportButton.addEventListener("click", () => {
      const output = document.createElement("canvas");
      output.width = mapCanvas.width;
      output.height = mapCanvas.height + 220;
      const ctx = output.getContext("2d");
      ctx.fillStyle = "#1a1b1e";
      ctx.fillRect(0, 0, output.width, output.height);
      ctx.fillStyle = "#f3f0ea";
      ctx.font = "bold 27px Arial";
      ctx.fillText(`${t(modeKeys[mode])} · ${title}`, 35, 45, 1030);
      ctx.font = "18px Arial";
      ctx.fillStyle = "#b8b2aa";
      ctx.fillText(
        `${selectedPeriod() ? `${periodLabel(selectedPeriod().period)} · ` : ""}${t(phaseKey(selectedPhase()))} · ${t("hmEstimated")} · ${time}`,
        35,
        77,
      );
      ctx.fillText(
        map.difference
          ? `A · ${result.maps[0].subject.label} / B · ${result.maps[1].subject.label}`
          : `${t("hmTotal")} : ${number(map.total)} ${unitLabel(map.unit)} · ${t("hmCoverage")} : ${map.coverage == null ? "—" : `${number(map.coverage * 100, 0)} %`}`,
        35,
        107,
        1030,
      );
      ctx.drawImage(mapCanvas, 0, 130);
      ctx.font = "15px Arial";
      ctx.fillText(
        `${t(legendKey)} · ${map.difference ? "±" : "0 – "}${number(legendMaximum)} ${displayUnit}`,
        35,
        output.height - 52,
        1030,
      );
      ctx.fillText(t("hmExportCaveat"), 35, output.height - 22, 1030);
      const link = document.createElement("a");
      link.download = `heatmap-${mode}-${selectedPhase()}${selectedPeriod() ? `-q${selectedPeriod().period}` : ""}-${map.subject.id.replace(/[^a-z0-9-]/gi, "-")}.png`;
      link.href = output.toDataURL("image/png");
      link.click();
    });
    return card;
  }
  function zoneTable(map, title) {
    const section = element("div", "hm-zone-table");
    const table = element("table");
    const caption = element("caption", "", title);
    table.append(caption);
    const head = element("thead"),
      header = element("tr");
    for (const key of ["hmZone", "hmValue", "hmShare"]) {
      const th = element("th", "", t(key));
      th.scope = "col";
      header.append(th);
    }
    head.append(header);
    table.append(head);
    const body = element("tbody");
    map.zones.forEach((value, index) => {
      const row = element("tr");
      row.append(
        element("th", "", t(`hmZone${index}`)),
        element(
          "td",
          "",
          `${number(value, 2)} ${map.difference ? "pp" : unitLabel(map.unit)}`,
        ),
        element(
          "td",
          "",
          map.total
            ? map.difference
              ? "—"
              : `${number(share(value, map.total))} %`
            : "—",
        ),
      );
      row.firstChild.scope = "row";
      body.append(row);
    });
    table.append(body);
    section.append(table);
    return section;
  }

  function courtBox(canvas) {
    const geometry = result.court;
    const margin = 45;
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
    const sx = box.width / geometry.length,
      sy = box.height / geometry.width;
    const x = (value) => box.x + value * sx,
      y = (value) => box.y + value * sy;
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
    ctx.strokeStyle = "#eee0bf99";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.width, box.height);
    ctx.clip();
    if (geometry.league === "NBA") {
      const circle = (cx, cy, radius) => {
        ctx.beginPath();
        ctx.arc(x(cx), y(cy), radius * sx, 0, Math.PI * 2);
        ctx.stroke();
      };
      circle(0, geometry.width / 2, 1.8288);
      const rim = geometry.length - 1.6002;
      const foul = geometry.length - 5.7912;
      ctx.strokeRect(
        x(foul),
        y((geometry.width - 4.8768) / 2),
        5.7912 * sx,
        4.8768 * sy,
      );
      circle(foul, geometry.width / 2, 1.8288);
      circle(rim, geometry.width / 2, 0.23);
      const radius = 7.239,
        half = geometry.width / 2,
        corner = 0.9144;
      const join = Math.sqrt(radius * radius - (half - corner) ** 2);
      for (const yy of [corner, geometry.width - corner]) {
        ctx.beginPath();
        ctx.moveTo(x(geometry.length), y(yy));
        ctx.lineTo(x(rim - join), y(yy));
        ctx.stroke();
      }
      ctx.beginPath();
      for (let i = 0; i <= 80; i++) {
        const yy = corner + ((geometry.width - 2 * corner) * i) / 80;
        const dx = Math.sqrt(Math.max(0, radius * radius - (yy - half) ** 2));
        if (!i) ctx.moveTo(x(rim - dx), y(yy));
        else ctx.lineTo(x(rim - dx), y(yy));
      }
      ctx.stroke();
    }
    ctx.restore();
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
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#b8b2aa";
    ctx.font = "28px Arial";
    ctx.fillText(t("hmHalfCourt"), canvas.width / 2, box.y - 15);
    ctx.fillText(
      t(selectedPhase() === "offense" ? "hmBasketAttack" : "hmBasketDefend"),
      canvas.width / 2,
      box.y + box.height + 27,
    );
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

  document
    .querySelectorAll("[data-heatmap-context]")
    .forEach((button) =>
      button.addEventListener("click", () =>
        open(button.dataset.heatmapContext, button),
      ),
    );
  document.querySelectorAll("[data-heatmap-mode]").forEach((button) =>
    button.addEventListener("click", () => {
      if (!modes.includes(button.dataset.heatmapMode)) return;
      mode = button.dataset.heatmapMode;
      if (mode === "comparison" && selectedPassage()) {
        $("heatmapPassage").value = "";
        resetRange();
      }
      passageOptions($("heatmapPassage").value);
      updateControls();
      calculate();
    }),
  );
  $("heatmapSubject").addEventListener("change", () => {
    passageOptions();
    resetRange();
    updateControls();
    calculate();
  });
  $("heatmapPhase").addEventListener("change", () =>
    changePhase(selectedPhase()),
  );
  $("heatmapPassage").addEventListener("change", () => {
    resetRange();
    calculate();
  });
  $("heatmapPeriod").addEventListener("change", () => {
    passageOptions();
    resetRange();
    calculate();
  });
  for (const id of ["heatmapStart", "heatmapEnd"])
    $(id).addEventListener("change", calculate);
  $("heatmapSmooth").addEventListener("change", render);
  $("heatmapLanguage").addEventListener("click", () =>
    setLanguage(language === "fr" ? "en" : "fr"),
  );
  $("closeHeatmaps").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    ++request;
    busy = false;
    if (opener?.isConnected) opener.focus();
  });
  function reset() {
    ++request;
    metadata = null;
    metadataRun = null;
    clearCharts();
    if (dialog.open) dialog.close();
  }
  return { open, translate, reset };
})();
