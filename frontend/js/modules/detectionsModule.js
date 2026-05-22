// Detections Module - manages bounding boxes on images

class DetectionsModule {
  constructor() {
    this.detections = [];
    this.nextId = 1;
    this.selectedId = null;
    this.mode = 'select';
    this.dragState = null;
    this.drawState = null;
    this.ghost = null;
    
    this.scene = null;
    this.layer = null;
    this.popup = null;
    
    this.HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
  }

  init(sceneId, layerId, popupId, datasetId) {
    this.datasetId = datasetId;
    this.scene = document.getElementById(sceneId);
    this.layer = document.getElementById(layerId);
    this.popup = document.getElementById(popupId);
    
    if (!this.scene || !this.layer || !this.popup) {
      console.error('Required elements not found');
      return;
    }
    
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Scene events
    this.scene.addEventListener('mousedown', (e) => this.onSceneMouseDown(e));
    
    // Document events
    document.addEventListener('mousemove', (e) => this.onDocumentMouseMove(e));
    document.addEventListener('mouseup', (e) => this.onDocumentMouseUp(e));
    document.addEventListener('keydown', (e) => this.onKeyDown(e));
    
    // Popup close
    document.getElementById('pop-close')?.addEventListener('click', () => this.closePopup());
  }

  sceneRect() {
    return this.scene.getBoundingClientRect();
  }

  pxToPercentX(px) {
    const r = this.sceneRect();
    return r.width > 0 ? (px / r.width) * 100 : 0;
  }

  pxToPercentY(px) {
    const r = this.sceneRect();
    return r.height > 0 ? (px / r.height) * 100 : 0;
  }

  clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  getDetection(id) {
    return this.detections.find((d) => d.id === id);
  }

  updateDetection(id, patch) {
    const d = this.getDetection(id);
    if (!d) return;
    Object.assign(d, patch);
    this.refreshBbox(id);
    if (this.selectedId === id) this.syncPopup(d);
    this.onUpdate?.();
  }

  createBboxEl(d) {
    const box = document.createElement('div');
    box.className = 'bbox';
    box.dataset.id = d.id;
    box.dataset.class = d.cls;

    box.innerHTML = `
      <div class="bbox__border"></div>
      <div class="bbox__label"></div>
      <div class="bbox__conf-bar"></div>
      ${this.HANDLES.map((dir) => `<div class="bbox__handle" data-dir="${dir}"></div>`).join('')}
      <button class="bbox__delete" title="Удалить">✕</button>
    `;

    box.querySelector('.bbox__border').addEventListener('mousedown', (e) => {
      if (this.mode !== 'select') return;
      e.stopPropagation();
      this.selectBbox(d.id);
      this.startMove(e, d.id);
    });

    box.querySelector('.bbox__border').addEventListener('dblclick', (e) => {
      e.stopPropagation();
      this.openPopup(d.id, e.clientX, e.clientY);
    });

    box.querySelectorAll('.bbox__handle').forEach((h) => {
      h.addEventListener('mousedown', (e) => {
        if (this.mode !== 'select') return;
        e.stopPropagation();
        this.selectBbox(d.id);
        this.startResize(e, d.id, h.dataset.dir);
      });
    });

    box.querySelector('.bbox__delete').addEventListener('click', (e) => {
      e.stopPropagation();
      this.deleteBbox(d.id);
    });

    box.addEventListener('mousedown', (e) => {
      if (this.mode !== 'select') return;
      e.stopPropagation();
      this.selectBbox(d.id);
    });

    this.layer.appendChild(box);
    return box;
  }

  refreshBbox(id) {
    const d = this.getDetection(id);
    const el = this.layer.querySelector(`[data-id="${id}"]`);
    if (!el || !d) return;

    el.dataset.class = d.cls;

    const widthPx = d.x2 - d.x1;
    const heightPx = d.y2 - d.y1;

    Object.assign(el.style, {
      left: this.pxToPercentX(d.x1) + '%',
      top: this.pxToPercentY(d.y1) + '%',
      width: this.pxToPercentX(widthPx) + '%',
      height: this.pxToPercentY(heightPx) + '%',
    });

    el.querySelector('.bbox__label').textContent = `${d.label}: ${Math.round(d.conf * 100)}%`;
    el.querySelector('.bbox__conf-bar').style.width = d.conf * 100 + '%';
  }

  renderAll() {
    this.layer.innerHTML = '';
    this.detections.forEach((d) => {
      this.createBboxEl(d);
      this.refreshBbox(d.id);
    });
  }

  selectBbox(id) {
    if (this.selectedId === id) return;
    this.deselectAll();
    this.selectedId = id;
    this.layer.querySelector(`[data-id="${id}"]`)?.classList.add('selected');
  }

  deselectAll() {
    this.selectedId = null;
    this.layer.querySelectorAll('.bbox.selected').forEach((el) => el.classList.remove('selected'));
    this.closePopup();
  }

  deleteBbox(id) {
    this.detections = this.detections.filter((d) => d.id !== id);
    this.layer.querySelector(`[data-id="${id}"]`)?.remove();
    if (this.selectedId === id) {
      this.selectedId = null;
      this.closePopup();
    }
    this.onDelete?.();
  }

  startMove(e, id) {
    const d = this.getDetection(id);
    this.dragState = {
      type: 'move',
      id: d.id,
      startMx: e.clientX,
      startMy: e.clientY,
      startX1: d.x1,
      startY1: d.y1,
    };
    document.body.style.cursor = 'move';
  }

  startResize(e, id, dir) {
    const d = this.getDetection(id);
    this.dragState = {
      type: 'resize',
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

  getRelPos(e) {
    const r = this.sceneRect();
    return {
      px: this.clamp(e.clientX - r.left, 0, r.width),
      py: this.clamp(e.clientY - r.top, 0, r.height),
    };
  }

  onSceneMouseDown(e) {
    if (this.mode === 'draw') {
      const { px, py } = this.getRelPos(e);
      this.ghost = document.createElement('div');
      this.ghost.className = 'draw-ghost';
      this.ghost.style.left = this.pxToPercentX(px) + '%';
      this.ghost.style.top = this.pxToPercentY(py) + '%';
      this.ghost.style.width = '0';
      this.ghost.style.height = '0';
      this.layer.appendChild(this.ghost);
      this.drawState = { startPx: px, startPy: py };
      return;
    }
    if (this.mode === 'select') this.deselectAll();
  }

  onDocumentMouseMove(e) {
    if (this.dragState?.type === 'move') {
      const r = this.sceneRect();
      const dx = e.clientX - this.dragState.startMx;
      const dy = e.clientY - this.dragState.startMy;

      const d = this.getDetection(this.dragState.id);
      const boxW = d.x2 - d.x1;
      const boxH = d.y2 - d.y1;

      let nx1 = this.dragState.startX1 + dx;
      let ny1 = this.dragState.startY1 + dy;

      nx1 = this.clamp(nx1, 0, r.width - boxW);
      ny1 = this.clamp(ny1, 0, r.height - boxH);

      this.updateDetection(this.dragState.id, {
        x1: nx1,
        y1: ny1,
        x2: nx1 + boxW,
        y2: ny1 + boxH,
      });
      return;
    }

    if (this.dragState?.type === 'resize') {
      const r = this.sceneRect();
      const dx = e.clientX - this.dragState.startMx;
      const dy = e.clientY - this.dragState.startMy;
      const dir = this.dragState.dir;

      let x1 = this.dragState.startX1;
      let y1 = this.dragState.startY1;
      let x2 = this.dragState.startX2;
      let y2 = this.dragState.startY2;

      const MIN = 8;

      if (dir.includes('e')) x2 = this.dragState.startX2 + dx;
      if (dir.includes('s')) y2 = this.dragState.startY2 + dy;
      if (dir.includes('w')) x1 = this.dragState.startX1 + dx;
      if (dir.includes('n')) y1 = this.dragState.startY1 + dy;

      x1 = this.clamp(x1, 0, r.width);
      y1 = this.clamp(y1, 0, r.height);
      x2 = this.clamp(x2, 0, r.width);
      y2 = this.clamp(y2, 0, r.height);

      if (x2 - x1 < MIN) {
        if (dir.includes('w')) x1 = x2 - MIN;
        else x2 = x1 + MIN;
      }

      if (y2 - y1 < MIN) {
        if (dir.includes('n')) y1 = y2 - MIN;
        else y2 = y1 + MIN;
      }

      x1 = this.clamp(x1, 0, r.width);
      y1 = this.clamp(y1, 0, r.height);
      x2 = this.clamp(x2, 0, r.width);
      y2 = this.clamp(y2, 0, r.height);

      this.updateDetection(this.dragState.id, { x1, y1, x2, y2 });
      return;
    }

    if (this.drawState && this.ghost) {
      const { px, py } = this.getRelPos(e);
      const x0 = Math.min(this.drawState.startPx, px);
      const y0 = Math.min(this.drawState.startPy, py);
      const w = Math.abs(px - this.drawState.startPx);
      const h = Math.abs(py - this.drawState.startPy);

      this.ghost.style.left = this.pxToPercentX(x0) + '%';
      this.ghost.style.top = this.pxToPercentY(y0) + '%';
      this.ghost.style.width = this.pxToPercentX(w) + '%';
      this.ghost.style.height = this.pxToPercentY(h) + '%';
    }
  }

async onDocumentMouseUp(e) {
  document.body.style.cursor = '';

  if (this.dragState) {
    this.dragState = null;
    return;
  }

  if (this.drawState && this.ghost) {
    const { px, py } = this.getRelPos(e);

    const x1 = Math.min(this.drawState.startPx, px);
    const y1 = Math.min(this.drawState.startPy, py);
    const x2 = Math.max(this.drawState.startPx, px);
    const y2 = Math.max(this.drawState.startPy, py);

    this.ghost.remove();
    this.ghost = null;
    this.drawState = null;

    if (x2 - x1 < 8 || y2 - y1 < 8) return;

    // Загружаем классы, берём первый если есть
    const classes = await this.loadDatasetClasses(this.datasetId);
    const firstClass = classes[0];

    const newDet = {
      id: this.nextId++,
      class_id: firstClass?.class_id ?? 0,
      label: firstClass?.name ?? 'Объект',
      cls: firstClass?.color ?? 'green',
      conf: 1.0,
      x1,
      y1,
      x2,
      y2,
    };

    this.detections.push(newDet);
    this.createBboxEl(newDet);
    this.refreshBbox(newDet.id);
    this.selectBbox(newDet.id);
    this.openPopup(newDet.id, e.clientX, e.clientY);
    this.setMode('select');
    this.onAdd?.();
  }
}

  onKeyDown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (this.selectedId !== null) this.deleteBbox(this.selectedId);
    }
    if (e.key === 'd' || e.key === 'D') this.setMode('draw');
    if (e.key === 's' || e.key === 'S' || e.key === 'Escape') this.setMode('select');
  }

  setMode(m) {
    this.mode = m;
    this.scene.className = `detection-scene mode-${m}`;
    
    const btnSelect = document.getElementById('btn-select');
    const btnDraw = document.getElementById('btn-draw');
    
    btnSelect?.classList.toggle('section-workspace__tool-btn--active', m === 'select');
    btnDraw?.classList.toggle('section-workspace__tool-btn--active', m === 'draw');
  }

  async loadDatasetClasses(datasetId) {
    try {
      const res = await fetch(`http://localhost:8000/api/datasets/${datasetId}/classes`);
      if (!res.ok) return [];
      return await res.json(); // [{id, class_id, name, color}]
    } catch { return []; }
  }

async openPopup(id, cx, cy) {
  const d = this.getDetection(id);
  if (!d) return;

  // Загружаем актуальные классы перед открытием
  const classes = await this.loadDatasetClasses(this.datasetId);

  this.syncPopup(d, classes);
  this.popup.classList.add('open');

  const pw = 260, ph = 220;
  this.popup.style.left = Math.min(cx + 10, window.innerWidth - pw - 10) + 'px';
  this.popup.style.top  = Math.min(cy + 10, window.innerHeight - ph - 10) + 'px';

  const handler = () => {
    const r = this.sceneRect();
    const popClassSelect = document.getElementById('pop-class-select');
    const popNewLabel    = document.getElementById('pop-new-label');
    const popClass       = document.getElementById('pop-class');
    const popConf        = document.getElementById('pop-conf');
    const popX = document.getElementById('pop-x');
    const popY = document.getElementById('pop-y');
    const popW = document.getElementById('pop-w');
    const popH = document.getElementById('pop-h');

    const selectedVal = popClassSelect?.value;
    let label, class_id;

    if (selectedVal === '__new__') {
      label    = popNewLabel?.value?.trim() || 'Объект';
      // class_id для нового = максимальный существующий + 1
      class_id = classes.length > 0
        ? Math.max(...classes.map(c => c.class_id)) + 1
        : 0;
    } else {
      const found = classes.find(c => String(c.class_id) === selectedVal);
      label    = found?.name || 'Объект';
      class_id = found?.class_id ?? 0;
    }

    const x1 = this.clamp(parseFloat(popX?.value) || 0, 0, r.width);
    const y1 = this.clamp(parseFloat(popY?.value) || 0, 0, r.height);
    const w  = Math.max(1, parseFloat(popW?.value) || 1);
    const h  = Math.max(1, parseFloat(popH?.value) || 1);

    this.updateDetection(id, {
      label,
      class_id,
      cls: popClass?.value || 'green',
      conf: this.clamp(parseFloat(popConf?.value) || 1, 0, 1),
      x1,
      y1,
      x2: this.clamp(x1 + w, 0, r.width),
      y2: this.clamp(y1 + h, 0, r.height),
    });
  };

  // показать/скрыть поле нового класса
  const popClassSelect = document.getElementById('pop-class-select');
  const popNewLabel    = document.getElementById('pop-new-label');

  popClassSelect?.removeEventListener('change', popClassSelect._changeHandler);
  popClassSelect._changeHandler = () => {
    if (popNewLabel) {
      popNewLabel.style.display = popClassSelect.value === '__new__' ? 'block' : 'none';
    }
    handler();
  };
  popClassSelect?.addEventListener('change', popClassSelect._changeHandler);

  const inputs = ['pop-class-select', 'pop-new-label', 'pop-class', 'pop-conf', 'pop-x', 'pop-y', 'pop-w', 'pop-h'];
  inputs.forEach((inputId) => {
    const el = document.getElementById(inputId);
    if (el) {
      el.removeEventListener('input', el._handler);
      el._handler = handler;
      el.addEventListener('input', handler);
    }
  });
}

syncPopup(d, classes = []) {
  const popClassSelect = document.getElementById('pop-class-select');
  const popNewLabel    = document.getElementById('pop-new-label');
  const popClass       = document.getElementById('pop-class');
  const popConf        = document.getElementById('pop-conf');
  const popX = document.getElementById('pop-x');
  const popY = document.getElementById('pop-y');
  const popW = document.getElementById('pop-w');
  const popH = document.getElementById('pop-h');

  // Заполняем select классами
  if (popClassSelect) {
    popClassSelect.innerHTML = '';
    classes.forEach(c => {
      const opt = document.createElement('option');
      opt.value = String(c.class_id);
      opt.textContent = `${c.name} (id: ${c.class_id})`;
      if (c.class_id === d.class_id) opt.selected = true;
      popClassSelect.appendChild(opt);
    });
    // Опция "новый класс"
    const newOpt = document.createElement('option');
    newOpt.value = '__new__';
    newOpt.textContent = '+ Новый класс';
    if (classes.length === 0) newOpt.selected = true;
    popClassSelect.appendChild(newOpt);

    // Показываем поле нового класса если нужно
    const isNew = popClassSelect.value === '__new__';
    if (popNewLabel) {
      popNewLabel.style.display = isNew ? 'block' : 'none';
      popNewLabel.value = isNew ? d.label : '';
    }
  }

  if (popClass) popClass.value = d.cls;
  if (popConf)  popConf.value  = d.conf.toFixed(2);
  if (popX)     popX.value     = d.x1.toFixed(0);
  if (popY)     popY.value     = d.y1.toFixed(0);
  if (popW)     popW.value     = (d.x2 - d.x1).toFixed(0);
  if (popH)     popH.value     = (d.y2 - d.y1).toFixed(0);
}

  closePopup() {
    this.popup.classList.remove('open');
  }

  // Public API
  load(data) {
    this.detections = data.map((d, index) => ({
      id: d.id ?? index + 1,
      class_id: d.class_id ?? 0,
      label: d.label ?? 'Объект',
      cls: d.cls ?? 'green',
      conf: d.conf ?? 1,
      x1: d.x1,
      y1: d.y1,
      x2: d.x2,
      y2: d.y2,
    }));

    this.nextId = this.detections.length > 0 ? Math.max(...this.detections.map((d) => d.id)) + 1 : 1;
    this.renderAll();
  }

  hide(id) {
    this.layer.querySelector(`[data-id="${id}"]`)?.classList.add('hidden');
  }

  show(id) {
    this.layer.querySelector(`[data-id="${id}"]`)?.classList.remove('hidden');
  }

  select(id) {
    this.selectBbox(id);
  }

  delete(id) {
    this.deleteBbox(id);
  }

  getAll() {
    return JSON.parse(JSON.stringify(this.detections));
  }

  setImage(src) {
    const img = document.getElementById('scene-img');
    if (img) img.src = src;
  }
}

export const detectionsModule = new DetectionsModule();
