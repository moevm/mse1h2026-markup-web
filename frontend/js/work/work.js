function initWork() {
  let detections = [
    {
      id: 1,
      label: "Машина",
      cls: "red",
      conf: 0.98,
      x: 33,
      y: 10,
      w: 18,
      h: 45,
    },
    {
      id: 2,
      label: "Человек",
      cls: "green",
      conf: 0.45,
      x: 8,
      y: 35,
      w: 14,
      h: 35,
    },
    {
      id: 3,
      label: "Машина",
      cls: "blue",
      conf: 0.91,
      x: 55,
      y: 50,
      w: 12,
      h: 18,
    },
    { id: 4, label: "Машина", cls: "red", conf: 0.87, x: 42, y: 55, w: 8, h: 14 },
  ];

  let nextId = 10;

  const scene = document.getElementById("scene");
  const layer = document.getElementById("detection-layer");
  const popup = document.getElementById("edit-popup");
  const btnSelect = document.getElementById("btn-select");
  const btnDraw = document.getElementById("btn-draw");
  const btnDelSel = document.getElementById("btn-delete-sel");

  const popLabel = document.getElementById("pop-label");
  const popClass = document.getElementById("pop-class");
  const popConf = document.getElementById("pop-conf");
  const popX = document.getElementById("pop-x");
  const popY = document.getElementById("pop-y");
  const popW = document.getElementById("pop-w");
  const popH = document.getElementById("pop-h");

  let mode = "select";
  let selectedId = null;

  let dragState = null;
  let drawState = null;

  let ghost = null;

  function sceneRect() {
    return scene.getBoundingClientRect();
  }

  function toPercent(px, dir) {
    const r = sceneRect();
    return dir === "x" ? (px / r.width) * 100 : (px / r.height) * 100;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function getDetection(id) {
    return detections.find((d) => d.id === id);
  }

  function updateDetection(id, patch) {
    const d = getDetection(id);
    if (!d) return;
    Object.assign(d, patch);
    refreshBbox(id);
    if (selectedId === id) syncPopup(d);
  }

  const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

  function createBboxEl(d) {
    const box = document.createElement("div");
    box.className = "section-workspace__bbox";
    box.dataset.id = d.id;
    box.dataset.class = d.cls;

    box.innerHTML = `
      <div class="section-workspace__bbox-border"></div>
      <div class="section-workspace__bbox-label"></div>
      <div class="section-workspace__bbox-conf-bar"></div>
      ${HANDLES.map((dir) => `<div class="section-workspace__bbox-handle" data-dir="${dir}"></div>`).join("")}
      <button class="section-workspace__bbox-delete" title="Удалить">✕</button>
    `;

    box.querySelector(".section-workspace__bbox-border").addEventListener("mousedown", (e) => {
      if (mode !== "select") return;
      e.stopPropagation();
      selectBbox(d.id);
      startMove(e, d.id);
    });

    box.querySelector(".section-workspace__bbox-border").addEventListener("dblclick", (e) => {
      e.stopPropagation();
      openPopup(d.id, e.clientX, e.clientY);
    });

    box.querySelectorAll(".section-workspace__bbox-handle").forEach((h) => {
      h.addEventListener("mousedown", (e) => {
        if (mode !== "select") return;
        e.stopPropagation();
        selectBbox(d.id);
        startResize(e, d.id, h.dataset.dir);
      });
    });

    box.querySelector(".section-workspace__bbox-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteBbox(d.id);
    });

    box.addEventListener("mousedown", (e) => {
      if (mode !== "select") return;
      e.stopPropagation();
      selectBbox(d.id);
    });

    layer.appendChild(box);
    return box;
  }

  function refreshBbox(id) {
    const d = getDetection(id);
    const el = layer.querySelector(`[data-id="${id}"]`);
    if (!el || !d) return;

    el.dataset.class = d.cls;
    Object.assign(el.style, {
      left: d.x + "%",
      top: d.y + "%",
      width: d.w + "%",
      height: d.h + "%",
    });

    el.querySelector(".section-workspace__bbox-label").textContent =
      `${d.label}: ${Math.round(d.conf * 100)}%`;
    el.querySelector(".section-workspace__bbox-conf-bar").style.width = d.conf * 100 + "%";
  }

  function renderAll() {
    layer.innerHTML = "";
    detections.forEach((d) => {
      createBboxEl(d);
      refreshBbox(d.id);
    });
  }

  function selectBbox(id) {
    if (selectedId === id) return;
    deselectAll();
    selectedId = id;
    layer.querySelector(`[data-id="${id}"]`)?.classList.add("section-workspace__bbox--selected");
    const d = getDetection(id);
  }

  function deselectAll() {
    selectedId = null;
    layer
      .querySelectorAll(".section-workspace__bbox--selected")
      .forEach((el) => el.classList.remove("section-workspace__bbox--selected"));
    closePopup();
  }

  function deleteBbox(id) {
    detections = detections.filter((d) => d.id !== id);
    layer.querySelector(`[data-id="${id}"]`)?.remove();
    if (selectedId === id) {
      selectedId = null;
      closePopup();
    }
  }

  function startMove(e, id) {
    const d = getDetection(id);
    dragState = {
      type: "move",
      id,
      startMx: e.clientX,
      startMy: e.clientY,
      startX: d.x,
      startY: d.y,
    };
    document.body.style.cursor = "move";
  }

  function startResize(e, id, dir) {
    const d = getDetection(id);
    dragState = {
      type: "resize",
      id,
      dir,
      startMx: e.clientX,
      startMy: e.clientY,
      startX: d.x,
      startY: d.y,
      startW: d.w,
      startH: d.h,
    };
    document.body.style.cursor = e.target.style.cursor;
  }

  function getRelPos(e) {
    const r = sceneRect();
    return {
      px: clamp(e.clientX - r.left, 0, r.width),
      py: clamp(e.clientY - r.top, 0, r.height),
    };
  }

  scene.addEventListener("mousedown", (e) => {
    if (mode === "draw") {
      const { px, py } = getRelPos(e);
      ghost = document.createElement("div");
      ghost.className = "section-workspace__draw-ghost";
      ghost.style.left = toPercent(px, "x") + "%";
      ghost.style.top = toPercent(py, "y") + "%";
      ghost.style.width = "0";
      ghost.style.height = "0";
      layer.appendChild(ghost);
      drawState = { startPx: px, startPy: py };
      return;
    }
    if (mode === "select") deselectAll();
  });

  document.addEventListener("mousemove", (e) => {
    if (dragState?.type === "move") {
      const r = sceneRect();
      const dx = ((e.clientX - dragState.startMx) / r.width) * 100;
      const dy = ((e.clientY - dragState.startMy) / r.height) * 100;
      const d = getDetection(dragState.id);
      updateDetection(dragState.id, {
        x: clamp(dragState.startX + dx, 0, 100 - d.w),
        y: clamp(dragState.startY + dy, 0, 100 - d.h),
      });
      return;
    }

    if (dragState?.type === "resize") {
      const r = sceneRect();
      const dx = ((e.clientX - dragState.startMx) / r.width) * 100;
      const dy = ((e.clientY - dragState.startMy) / r.height) * 100;
      const dir = dragState.dir;
      let { startX: nx, startY: ny, startW: nw, startH: nh } = dragState;
      const MIN = 2;

      if (dir.includes("e")) nw = Math.max(MIN, dragState.startW + dx);
      if (dir.includes("s")) nh = Math.max(MIN, dragState.startH + dy);
      if (dir.includes("w")) {
        nw = Math.max(MIN, dragState.startW - dx);
        nx = dragState.startX + dragState.startW - nw;
      }
      if (dir.includes("n")) {
        nh = Math.max(MIN, dragState.startH - dy);
        ny = dragState.startY + dragState.startH - nh;
      }

      updateDetection(dragState.id, {
        x: clamp(nx, 0, 100),
        y: clamp(ny, 0, 100),
        w: Math.min(nw, 100 - nx),
        h: Math.min(nh, 100 - ny),
      });
      return;
    }

    if (drawState && ghost) {
      const { px, py } = getRelPos(e);
      const x0 = Math.min(drawState.startPx, px);
      const y0 = Math.min(drawState.startPy, py);
      const w = Math.abs(px - drawState.startPx);
      const h = Math.abs(py - drawState.startPy);
      const r = sceneRect();
      ghost.style.left = (x0 / r.width) * 100 + "%";
      ghost.style.top = (y0 / r.height) * 100 + "%";
      ghost.style.width = (w / r.width) * 100 + "%";
      ghost.style.height = (h / r.height) * 100 + "%";
    }
  });

  document.addEventListener("mouseup", (e) => {
    document.body.style.cursor = "";

    if (dragState) {
      dragState = null;
      return;
    }

    if (drawState && ghost) {
      const r = sceneRect();
      const x0 = parseFloat(ghost.style.left);
      const y0 = parseFloat(ghost.style.top);
      const w = parseFloat(ghost.style.width);
      const h = parseFloat(ghost.style.height);
      ghost.remove();
      ghost = null;
      drawState = null;

      if (w < 1 || h < 1) return;

      const newDet = {
        id: nextId++,
        label: "Объект",
        cls: "green",
        conf: 1.0,
        x: x0,
        y: y0,
        w,
        h,
      };
      detections.push(newDet);
      createBboxEl(newDet);
      refreshBbox(newDet.id);
      selectBbox(newDet.id);
      openPopup(newDet.id, e.clientX, e.clientY);
      setMode("select");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "Delete" || e.key === "Backspace") {
      if (selectedId !== null) deleteBbox(selectedId);
    }
    if (e.key === "d" || e.key === "D") setMode("draw");
    if (e.key === "s" || e.key === "S" || e.key === "Escape") setMode("select");
  });

  function setMode(m) {
    mode = m;
    scene.className = `section-workspace__detection-scene .section-workspace__detection-scene--mode-${m}`;
    btnSelect.classList.toggle("section-workspace__tool-btn--active", m === "select");
    btnDraw.classList.toggle("section-workspace__tool-btn--active", m === "draw");
  }

  btnSelect.addEventListener("click", () => setMode("select"));
  btnDraw.addEventListener("click", () => setMode("draw"));

  function openPopup(id, cx, cy) {
    const d = getDetection(id);
    if (!d) return;
    syncPopup(d);
    popup.classList.add("section-workspace__edit-popup--open");

    const pw = 220,
    ph = 200;
    popup.style.left = Math.min(cx + 10, window.innerWidth - pw - 10) + "px";
    popup.style.top = Math.min(cy + 10, window.innerHeight - ph - 10) + "px";

    const handler = () => {
      updateDetection(id, {
        label: popLabel.value || "Объект",
        cls: popClass.value,
        conf: clamp(parseFloat(popConf.value) || 1, 0, 1),
        x: clamp(parseFloat(popX.value) || 0, 0, 100),
        y: clamp(parseFloat(popY.value) || 0, 0, 100),
        w: clamp(parseFloat(popW.value) || 1, 0.5, 100),
        h: clamp(parseFloat(popH.value) || 1, 0.5, 100),
      });
    };

    [popLabel, popClass, popConf, popX, popY, popW, popH].forEach((el) => {
      el.removeEventListener("input", el._handler);
      el._handler = handler;
      el.addEventListener("input", handler);
    });
  }

  function syncPopup(d) {
    popLabel.value = d.label;
    popClass.value = d.cls;
    popConf.value = d.conf.toFixed(2);
    popX.value = d.x.toFixed(1);
    popY.value = d.y.toFixed(1);
    popW.value = d.w.toFixed(1);
    popH.value = d.h.toFixed(1);
  }

  function closePopup() {
    popup.classList.remove("section-workspace__edit-popup--open");
  }

  document.getElementById("pop-close").addEventListener("click", closePopup);

  btnDelSel.addEventListener("click", () => {
    if (selectedId !== null) deleteBbox(selectedId);
  });

  renderAll();
}

initWork();