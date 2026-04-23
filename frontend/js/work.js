let detections = [
  {
    id: 1,
    class_id: 0,
    label: "Машина",
    cls: "red",
    conf: 0.98,
    x1: 330,
    y1: 80,
    x2: 510,
    y2: 440,
  },
  {
    id: 2,
    class_id: 1,
    label: "Человек",
    cls: "green",
    conf: 0.45,
    x1: 80,
    y1: 280,
    x2: 220,
    y2: 560,
  },
  {
    id: 3,
    class_id: 0,
    label: "Машина",
    cls: "blue",
    conf: 0.91,
    x1: 550,
    y1: 400,
    x2: 670,
    y2: 544,
  },
  {
    id: 4,
    class_id: 0,
    label: "Машина",
    cls: "red",
    conf: 0.87,
    x1: 420,
    y1: 440,
    x2: 500,
    y2: 552,
  },
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

function pxToPercentX(px) {
  const r = sceneRect();
  return r.width > 0 ? (px / r.width) * 100 : 0;
}

function pxToPercentY(px) {
  const r = sceneRect();
  return r.height > 0 ? (px / r.height) * 100 : 0;
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
  box.className = "bbox";
  box.dataset.id = d.id;
  box.dataset.class = d.cls;

  box.innerHTML = `
    <div class="bbox__border"></div>
    <div class="bbox__label"></div>
    <div class="bbox__conf-bar"></div>
    ${HANDLES.map((dir) => `<div class="bbox__handle" data-dir="${dir}"></div>`).join("")}
    <button class="bbox__delete" title="Удалить">✕</button>
  `;

  box.querySelector(".bbox__border").addEventListener("mousedown", (e) => {
    if (mode !== "select") return;
    e.stopPropagation();
    selectBbox(d.id);
    startMove(e, d.id);
  });

  box.querySelector(".bbox__border").addEventListener("dblclick", (e) => {
    e.stopPropagation();
    openPopup(d.id, e.clientX, e.clientY);
  });

  box.querySelectorAll(".bbox__handle").forEach((h) => {
    h.addEventListener("mousedown", (e) => {
      if (mode !== "select") return;
      e.stopPropagation();
      selectBbox(d.id);
      startResize(e, d.id, h.dataset.dir);
    });
  });

  box.querySelector(".bbox__delete").addEventListener("click", (e) => {
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

  const widthPx = d.x2 - d.x1;
  const heightPx = d.y2 - d.y1;

  Object.assign(el.style, {
    left: pxToPercentX(d.x1) + "%",
    top: pxToPercentY(d.y1) + "%",
    width: pxToPercentX(widthPx) + "%",
    height: pxToPercentY(heightPx) + "%",
  });

  el.querySelector(".bbox__label").textContent =
    `${d.label}: ${Math.round(d.conf * 100)}%`;
  el.querySelector(".bbox__conf-bar").style.width = d.conf * 100 + "%";
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
  layer.querySelector(`[data-id="${id}"]`)?.classList.add("selected");
}

function deselectAll() {
  selectedId = null;
  layer
    .querySelectorAll(".bbox.selected")
    .forEach((el) => el.classList.remove("selected"));
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
    id: d.id,
    startMx: e.clientX,
    startMy: e.clientY,
    startX1: d.x1,
    startY1: d.y1,
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
    startX1: d.x1,
    startY1: d.y1,
    startX2: d.x2,
    startY2: d.y2,
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
    ghost.className = "draw-ghost";
    ghost.style.left = pxToPercentX(px) + "%";
    ghost.style.top = pxToPercentY(py) + "%";
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
  const dx = e.clientX - dragState.startMx;
  const dy = e.clientY - dragState.startMy;

  const d = getDetection(dragState.id);
  const boxW = d.x2 - d.x1;
  const boxH = d.y2 - d.y1;

  let nx1 = dragState.startX1 + dx;
  let ny1 = dragState.startY1 + dy;

  nx1 = clamp(nx1, 0, r.width - boxW);
  ny1 = clamp(ny1, 0, r.height - boxH);

  updateDetection(dragState.id, {
    x1: nx1,
    y1: ny1,
    x2: nx1 + boxW,
    y2: ny1 + boxH,
  });

  return;
}

  if (dragState?.type === "resize") {
  const r = sceneRect();
  const dx = e.clientX - dragState.startMx;
  const dy = e.clientY - dragState.startMy;
  const dir = dragState.dir;

  let x1 = dragState.startX1;
  let y1 = dragState.startY1;
  let x2 = dragState.startX2;
  let y2 = dragState.startY2;

  const MIN = 8;

  if (dir.includes("e")) x2 = dragState.startX2 + dx;
  if (dir.includes("s")) y2 = dragState.startY2 + dy;
  if (dir.includes("w")) x1 = dragState.startX1 + dx;
  if (dir.includes("n")) y1 = dragState.startY1 + dy;

  x1 = clamp(x1, 0, r.width);
  y1 = clamp(y1, 0, r.height);
  x2 = clamp(x2, 0, r.width);
  y2 = clamp(y2, 0, r.height);

  if (x2 - x1 < MIN) {
    if (dir.includes("w")) x1 = x2 - MIN;
    else x2 = x1 + MIN;
  }

  if (y2 - y1 < MIN) {
    if (dir.includes("n")) y1 = y2 - MIN;
    else y2 = y1 + MIN;
  }

  x1 = clamp(x1, 0, r.width);
  y1 = clamp(y1, 0, r.height);
  x2 = clamp(x2, 0, r.width);
  y2 = clamp(y2, 0, r.height);

  updateDetection(dragState.id, { x1, y1, x2, y2 });
  return;
}

  if (drawState && ghost) {
    const { px, py } = getRelPos(e);
    const x0 = Math.min(drawState.startPx, px);
    const y0 = Math.min(drawState.startPy, py);
    const w = Math.abs(px - drawState.startPx);
    const h = Math.abs(py - drawState.startPy);

    ghost.style.left = pxToPercentX(x0) + "%";
    ghost.style.top = pxToPercentY(y0) + "%";
    ghost.style.width = pxToPercentX(w) + "%";
    ghost.style.height = pxToPercentY(h) + "%";
}
});

document.addEventListener("mouseup", (e) => {
  document.body.style.cursor = "";

  if (dragState) {
    dragState = null;
    return;
  }

  if (drawState && ghost) {
    const { px, py } = getRelPos(e);

    const x1 = Math.min(drawState.startPx, px);
    const y1 = Math.min(drawState.startPy, py);
    const x2 = Math.max(drawState.startPx, px);
    const y2 = Math.max(drawState.startPy, py);

    ghost.remove();
    ghost = null;
    drawState = null;

    if (x2 - x1 < 8 || y2 - y1 < 8) return;

    const newDet = {
      id: nextId++,
      class_id: 0,
      label: "Объект",
      cls: "green",
      conf: 1.0,
      x1,
      y1,
      x2,
      y2,
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
  scene.className = `detection-scene mode-${m}`;
  btnSelect.classList.toggle("section-workspace__tool-btn--active", m === "select");
  btnDraw.classList.toggle("section-workspace__tool-btn--active", m === "draw");
}

btnSelect.addEventListener("click", () => setMode("select"));
btnDraw.addEventListener("click", () => setMode("draw"));

function openPopup(id, cx, cy) {
  const d = getDetection(id);
  if (!d) return;
  syncPopup(d);
  popup.classList.add("open");

  const pw = 220,
  ph = 200;
  popup.style.left = Math.min(cx + 10, window.innerWidth - pw - 10) + "px";
  popup.style.top = Math.min(cy + 10, window.innerHeight - ph - 10) + "px";

  const handler = () => {
    const r = sceneRect();

    const x1 = clamp(parseFloat(popX.value) || 0, 0, r.width);
    const y1 = clamp(parseFloat(popY.value) || 0, 0, r.height);
    const w = Math.max(1, parseFloat(popW.value) || 1);
    const h = Math.max(1, parseFloat(popH.value) || 1);

    updateDetection(id, {
      label: popLabel.value || "Объект",
      cls: popClass.value,
      conf: clamp(parseFloat(popConf.value) || 1, 0, 1),
      x1,
      y1,
      x2: clamp(x1 + w, 0, r.width),
      y2: clamp(y1 + h, 0, r.height),
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
  popX.value = d.x1.toFixed(0);
  popY.value = d.y1.toFixed(0);
  popW.value = (d.x2 - d.x1).toFixed(0);
  popH.value = (d.y2 - d.y1).toFixed(0);
}

function closePopup() {
  popup.classList.remove("open");
}

document.getElementById("pop-close").addEventListener("click", closePopup);

btnDelSel.addEventListener("click", () => {
  if (selectedId !== null) deleteBbox(selectedId);
});

window.DetectionOverlay = {
  load(data) {
    detections = data.map((d, index) => ({
      id: d.id ?? index + 1,
      class_id: d.class_id ?? 0,
      label: d.label ?? "Объект",
      cls: d.cls ?? "green",
      conf: d.conf ?? 1,
      x1: d.x1,
      y1: d.y1,
      x2: d.x2,
      y2: d.y2,
  }));

  nextId =
    detections.length > 0
      ? Math.max(...detections.map((d) => d.id)) + 1
      : 1;

  renderAll();
},
  hide(id) {
    layer.querySelector(`[data-id="${id}"]`)?.classList.add("hidden");
  },
  show(id) {
    layer.querySelector(`[data-id="${id}"]`)?.classList.remove("hidden");
  },
  select(id) {
    selectBbox(id);
  },
  delete(id) {
    deleteBbox(id);
  },
  getAll() {
    return JSON.parse(JSON.stringify(detections));
  },
  setImage(src) {
    document.getElementById("scene-img").src = src;
  },
};

renderAll();
