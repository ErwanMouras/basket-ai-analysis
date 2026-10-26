/* Spatial analysis is separate from the frame/segmentation playback loop. */
window.heatmapStudio = (() => {
  const dialog = $("heatmapDialog");
  const modes = ["presence", "movement", "zones", "comparison", "evolution"];
  const modeKeys = {
    presence: "hmPresence",
    movement: "hmMovement",
    zones: "hmZones",
    comparison: "hmComparison",
    evolution: "hmEvolution",
  };
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
  const selectedPlan = () =>
    metadata?.plans.find((item) => String(item.id) === $("heatmapPlan").value);

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
  function updateCompare() {
    const subject = selectedSubject();
    const candidates =
      metadata?.subjects.filter(
        (item) => item.kind === subject?.kind && item.id !== subject?.id,
      ) || [];
    subjectOptions($("heatmapCompare"), candidates, $("heatmapCompare").value);
    $("heatmapCompareField").hidden = mode !== "comparison";
    $("heatmapSmoothField").hidden = mode === "zones";
  }
  function planOptions(value) {
    const select = $("heatmapPlan");
    select.replaceChildren();
    for (const plan of metadata?.plans || []) {
      const option = new Option(
        `${t("hmPlanLabel", { count: plan.id + 1 })} · ${fmtTime(plan.start_s)} – ${fmtTime(plan.end_s)}`,
        plan.id,
      );
      option.disabled = plan.end_s <= plan.start_s;
      select.add(option);
    }
    if (value != null) select.value = value;
  }
  function resetRange() {
    const plan = selectedPlan();
    if (!plan) return;
    for (const id of ["heatmapStart", "heatmapEnd"]) {
      $(id).min = plan.start_s;
      $(id).max = plan.end_s;
    }
    $("heatmapStart").value = numberForInput(plan.start_s);
    $("heatmapEnd").value = numberForInput(plan.end_s);
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
      const plans = metadata.plans.filter((plan) => plan.end_s > plan.start_s);
      const plan =
        [...plans]
          .reverse()
          .find((item) => item.start_s <= (video.currentTime || 0)) || plans[0];
      planOptions(plan?.id);
      resetRange();
      updateCompare();
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
    const plan = selectedPlan();
    const subject = selectedSubject();
    if (!plan || !subject) {
      clearCharts();
      status("hmNoData");
      return;
    }
    const rawStart = $("heatmapStart").value,
      rawEnd = $("heatmapEnd").value;
    const a = Math.max(plan.start_s, Number(rawStart));
    const b = Math.min(plan.end_s, Number(rawEnd));
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
    if (mode === "comparison" && !$("heatmapCompare").value) {
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
          plan: plan.id,
          start: a,
          end: b,
          mode,
          compare: $("heatmapCompare").value,
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
    document
      .querySelectorAll("[data-heatmap-mode]")
      .forEach((button) =>
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.heatmapMode === mode),
        ),
      );
  }
  function updateIntro() {
    $("heatmapModeTitle").textContent = t(modeKeys[mode]);
    $("heatmapExplanation").textContent = t(`${modeKeys[mode]}Help`);
    const subject = selectedSubject();
    const plan = selectedPlan();
    $("heatmapScope").textContent = [
      subject?.label,
      plan ? t("hmPlanLabel", { count: plan.id + 1 }) : "",
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
      planOptions($("heatmapPlan").value);
      updateCompare();
    }
    const disabled = !metadata?.available;
    for (const id of [
      "heatmapSubject",
      "heatmapPlan",
      "heatmapStart",
      "heatmapEnd",
      "heatmapCompare",
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

  function render() {
    if (!result) return;
    const charts = $("heatmapCharts");
    charts.replaceChildren();
    const tables = $("heatmapTables");
    tables.replaceChildren();
    charts.classList.toggle("hm-multiple", result.maps.length > 1);
    charts.classList.toggle("hm-comparison", mode === "comparison");
    const maximum = Math.max(
      0,
      ...result.maps.flatMap((map) =>
        map.grid.map((value) => share(value, map.total)),
      ),
    );
    result.maps.forEach((map, index) => {
      const title =
        mode === "evolution"
          ? t(index ? "hmSecond" : "hmFirst")
          : mode === "comparison"
            ? `${index ? "B" : "A"} · ${map.subject.label}`
            : map.subject.label;
      charts.append(mapCard(map, title, maximum));
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
      charts.append(
        mapCard(
          difference,
          t("hmDifference"),
          Math.max(0, ...grid.map(Math.abs)),
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
  function mapCard(map, title, maximum) {
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
    mapCanvas.height = 650;
    mapCanvas.setAttribute("role", "img");
    mapCanvas.setAttribute(
      "aria-label",
      `${t(modeKeys[mode])} · ${title} · ${time}`,
    );
    paintMap(mapCanvas, map, maximum);
    card.append(mapCanvas);
    const empty = map.difference ? !map.available : map.total <= 0;
    if (empty)
      card.append(
        element(
          "p",
          "hm-empty",
          t(
            map.difference
              ? "hmNoDifference"
              : mode === "movement" && map.measured_s > 0
                ? "hmNoMovement"
                : "hmNoData",
          ),
        ),
      );
    const legend = element("div", "hm-map-legend");
    if (map.difference) {
      legend.append(
        element("span", "", `−${number(maximum)} pp`),
        element("i", "hm-gradient hm-diverging"),
        element("span", "", `+${number(maximum)} pp`),
      );
      card.append(element("p", "hm-legend-caption", t("hmDifferenceLegend")));
    } else {
      legend.append(
        element("span", "", "0"),
        element("i", "hm-gradient"),
        element("span", "", `${number(legendMaximum)} %`),
      );
      card.append(
        element(
          "p",
          "hm-legend-caption",
          t(
            mode === "movement"
              ? "hmMovementLegend"
              : mode === "zones"
                ? "hmZonesLegend"
                : "hmLegend",
          ),
        ),
      );
    }
    card.append(legend);
    if (!map.difference) {
      const stats = element("div", "hm-map-stats");
      stats.append(
        stat(
          mode === "movement" ? "hmTotalMovement" : "hmTotal",
          number(map.total),
          null,
          unitLabel(map.unit),
        ),
        stat(
          "hmCoverage",
          map.coverage == null ? "—" : number(map.coverage * 100, 0),
          "hmCoverageHelp",
          map.coverage == null ? null : "%",
        ),
        stat("hmVideoTime", number(map.localized_video_s), "hmVideoHelp", "s"),
      );
      card.insertBefore(stats, mapCanvas);
      if (mode === "movement")
        card.append(
          element(
            "p",
            "hm-measure-note",
            t("hmMovementCoverage", {
              value: number(100 * (map.movement_coverage || 0), 0),
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
          map.available ? "pp" : null,
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
        `${t("hmPlanLabel", { count: result.plan.id + 1 })} · ${time}`,
        35,
        77,
      );
      ctx.fillText(
        map.difference
          ? `A · ${result.maps[0].subject.label} / B · ${result.maps[1].subject.label}`
          : `${t(mode === "movement" ? "hmTotalMovement" : "hmTotal")} : ${number(map.total)} ${unitLabel(map.unit)} · ${t("hmCoverage")} : ${map.coverage == null ? "—" : `${number(map.coverage * 100, 0)} %`}`,
        35,
        107,
        1030,
      );
      ctx.drawImage(mapCanvas, 0, 130);
      ctx.font = "15px Arial";
      ctx.fillText(
        map.difference
          ? t("hmDifferenceLegend")
          : `${t(mode === "movement" ? "hmMovementLegend" : mode === "zones" ? "hmZonesLegend" : "hmLegend")} · 0 – ${number(legendMaximum)} %`,
        35,
        output.height - 52,
        1030,
      );
      ctx.fillText(t("hmExportCaveat"), 35, output.height - 22, 1030);
      const link = document.createElement("a");
      link.download = `heatmap-${mode}-${map.subject.id.replace(/[^a-z0-9-]/gi, "-")}.png`;
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
  function paintMap(canvas, map, maximum) {
    const ctx = canvas.getContext("2d");
    const box = courtBox(canvas),
      geometry = result.court;
    const sx = box.width / geometry.length,
      sy = box.height / geometry.width;
    const x = (value) => box.x + value * sx,
      y = (value) => box.y + value * sy;
    ctx.fillStyle = "#1d1c1b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#292620";
    ctx.fillRect(box.x, box.y, box.width, box.height);
    const values = mode === "zones" ? map.zones : map.grid;
    const nx = mode === "zones" ? 3 : map.nx,
      ny = mode === "zones" ? 3 : map.ny;
    const normalized = values.map((value) =>
      map.difference ? value : share(value, map.total),
    );
    const max = mode === "zones" ? Math.max(0, ...normalized) : maximum;
    const raster = document.createElement("canvas");
    raster.width = nx;
    raster.height = ny;
    const paint = raster.getContext("2d");
    normalized.forEach((value, index) => {
      const amount = max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
      paint.fillStyle = color(amount, map.difference && value < 0);
      paint.fillRect(index % nx, Math.floor(index / nx), 1, 1);
    });
    ctx.imageSmoothingEnabled = $("heatmapSmooth").checked && mode !== "zones";
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(raster, box.x, box.y, box.width, box.height);
    ctx.strokeStyle = "#eee0bf99";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.beginPath();
    ctx.moveTo(x(geometry.length / 2), box.y);
    ctx.lineTo(x(geometry.length / 2), box.y + box.height);
    ctx.stroke();
    if (geometry.league === "NBA") {
      const circle = (cx, cy, radius) => {
        ctx.beginPath();
        ctx.arc(x(cx), y(cy), radius * sx, 0, Math.PI * 2);
        ctx.stroke();
      };
      circle(geometry.length / 2, geometry.width / 2, 1.8288);
      for (const side of [0, 1]) {
        const rim = side ? geometry.length - 1.6002 : 1.6002;
        const foul = side ? geometry.length - 5.7912 : 5.7912;
        ctx.strokeRect(
          x(side ? foul : 0),
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
          ctx.moveTo(x(side ? geometry.length : 0), y(yy));
          ctx.lineTo(x(rim + (side ? -join : join)), y(yy));
          ctx.stroke();
        }
        ctx.beginPath();
        for (let i = 0; i <= 80; i++) {
          const yy = corner + ((geometry.width - 2 * corner) * i) / 80;
          const dx = Math.sqrt(Math.max(0, radius * radius - (yy - half) ** 2));
          if (!i) ctx.moveTo(x(rim + (side ? -dx : dx)), y(yy));
          else ctx.lineTo(x(rim + (side ? -dx : dx)), y(yy));
        }
        ctx.stroke();
      }
    }
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
    ctx.font = "16px Arial";
    ctx.fillText(t("hmSideFar"), canvas.width / 2, box.y - 15);
    ctx.fillText(t("hmSideNear"), canvas.width / 2, box.y + box.height + 27);
  }
  function color(amount, blue) {
    const stops = blue
      ? [
          [41, 38, 32],
          [60, 123, 182],
          [193, 228, 253],
        ]
      : [
          [41, 38, 32],
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
      updateCompare();
      calculate();
    }),
  );
  $("heatmapSubject").addEventListener("change", () => {
    updateCompare();
    calculate();
  });
  $("heatmapPlan").addEventListener("change", () => {
    resetRange();
    calculate();
  });
  for (const id of ["heatmapStart", "heatmapEnd", "heatmapCompare"])
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
