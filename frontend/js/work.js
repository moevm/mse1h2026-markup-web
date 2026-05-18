// Check if dataset is selected
import { classesManager } from './managers/classesManager.js';
import { detectionsModule } from './modules/detectionsModule.js';
import { Notify } from './utils/notify.js'

const currentDataset = sessionStorage.getItem('currentDataset') || localStorage.getItem('currentDataset');
if (!currentDataset) {
  window.location.href = '/';
}

const dataset = JSON.parse(currentDataset);

// Display dataset name
document.getElementById('dataset-name').textContent = `Выбранный датасет: ${dataset.name}`;

// Initialize classes manager
classesManager.init(dataset.id);

// Initialize detections module
detectionsModule.init('scene', 'detection-layer', 'edit-popup');

// Setup mode buttons
document.getElementById('btn-select')?.addEventListener('click', () => detectionsModule.setMode('select'));
document.getElementById('btn-draw')?.addEventListener('click', () => detectionsModule.setMode('draw'));
document.getElementById('btn-delete-sel')?.addEventListener('click', () => {
  if (detectionsModule.selectedId !== null) {
    detectionsModule.delete(detectionsModule.selectedId);
  }
});

// Categories for image slider
const CATEGORIES = [
  { id: 1, name: 'В ОЖИДАНИИ', codes: ['unlabeled'] },
  { id: 2, name: 'АВТО-РАЗМЕТКА (ТРЕБУЕТ ПРОВЕРКИ)', codes: ['auto_labeled_pending_review'] },
  { id: 3, name: 'РАЗМЕЧЕНО', codes: ['labeled', 'finalized', 'ready_for_training'] }
];

let currentCategoryIndex = 0;
let currentImages = [];
let currentImageIndex = 0;
let currentImage = null;

// Category slider
const categoryTitle = document.getElementById('current-category');
const prevBtn = document.querySelector('.section-workspace__slider-btn--prev');
const nextBtn = document.querySelector('.section-workspace__slider-btn--next');

function updateCategory() {
  categoryTitle.textContent = CATEGORIES[currentCategoryIndex].name;
  loadImagesForCategory();
  updateDecisionButtons();
}

function updateDecisionButtons() {
   const category = CATEGORIES[currentCategoryIndex];
   
   // Get UI elements
   const decisionBox = document.querySelector('.section-workspace__decision-box');
   const autoMarkupBox = document.querySelector('.section-workspace__auto-markup-box');
   const saveButton = document.querySelector('.section-workspace__toolbar-server-button--recheck');
   
   // Show/hide buttons based on category
   if (category.codes && category.codes.includes('unlabeled')) {
     // В ОЖИДАНИИ: показать авторазметку и сохранение
     if (autoMarkupBox) autoMarkupBox.style.display = 'flex';
     if (saveButton) saveButton.style.display = 'block';
     if (decisionBox) decisionBox.style.display = 'none';
   } else if (category.codes && category.codes.includes('auto_labeled_pending_review')) {
     // АВТО-РАЗМЕТКА: скрыть авторазметку, показать сохранение и принять/отклонить
     if (autoMarkupBox) autoMarkupBox.style.display = 'none';
     if (saveButton) saveButton.style.display = 'block';
     if (decisionBox) decisionBox.style.display = 'flex';
   } else {
     // РАЗМЕЧЕНО: скрыть всё
     if (autoMarkupBox) autoMarkupBox.style.display = 'none';
     if (saveButton) saveButton.style.display = 'none';
     if (decisionBox) decisionBox.style.display = 'none';
   }
 }

prevBtn.addEventListener('click', () => {
  currentCategoryIndex = (currentCategoryIndex - 1 + CATEGORIES.length) % CATEGORIES.length;
  updateCategory();
});

nextBtn.addEventListener('click', () => {
  currentCategoryIndex = (currentCategoryIndex + 1) % CATEGORIES.length;
  updateCategory();
});

async function loadImagesForCategory() {
  const category = CATEGORIES[currentCategoryIndex];
  try {
    // If category has multiple codes, fetch all and merge
    if (category.codes && category.codes.length > 1) {
      const allImages = [];
      for (const code of category.codes) {
        const res = await fetch(`http://localhost:8000/api/datasets/${dataset.id}/images?status=${code}`);
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
        const images = await res.json();
        allImages.push(...images);
      }
      currentImages = allImages;
    } else {
      // Single code
      const code = category.codes ? category.codes[0] : category.code;
      const res = await fetch(`http://localhost:8000/api/datasets/${dataset.id}/images?status=${code}`);
      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      currentImages = await res.json();
    }
    
    currentImageIndex = 0;
    renderImagesList();
    if (currentImages.length > 0) {
      loadImage(currentImages[0]);
    }
  } catch (err) {
    console.error('Failed to load images:', err);
    currentImages = [];
    renderImagesList();
  }
}

function renderImagesList() {
   const list = document.getElementById('images-list');
   list.innerHTML = '';
   
   if (currentImages.length === 0) {
     list.innerHTML = '<li class="section-workspace__no-images">Нет изображений в этой категории</li>';
     return;
   }
   
   currentImages.forEach((img, index) => {
     const li = document.createElement('li');
     li.className = 'section-workspace__image-card';
     if (index === currentImageIndex) {
       li.classList.add('section-workspace__image-card--active');
     }
     
     li.innerHTML = `
       <button class="section-workspace__image-button" data-index="${index}">
         <img 
           class="section-workspace__image-preview" 
           src="http://localhost:8000${img.preview_url}" 
           alt="${img.filename}"
         />
         <h4 class="section-workspace__image-title">${img.filename}</h4>
       </button>
     `;
     
     li.querySelector('button').addEventListener('click', () => {
       currentImageIndex = index;
       loadImage(img);
       renderImagesList();
     });
     
     list.appendChild(li);
   });
 }

function renderObjectList() {
    const list = document.querySelector('.section-workspace__object-list');
    const detections = detectionsModule.getAll();
    
    list.innerHTML = '';
    
    if (detections.length === 0) {
      return;
    }
   
   detections.forEach((detection, index) => {
     const li = document.createElement('li');
     li.className = 'section-workspace__object-item';
     
     // Determine confidence percentage and styling
     const confidencePercent = Math.round(detection.conf * 100);
     let confidenceClass = '';
     if (confidencePercent >= 80) {
       confidenceClass = 'section-workspace__object-sub-data--high-percent';
     } else if (confidencePercent >= 50) {
       confidenceClass = 'section-workspace__object-sub-data--medium-percent';
     } else {
       confidenceClass = 'section-workspace__object-sub-data--low-percent';
     }
     
li.innerHTML = `
        <button class="section-workspace__object-button" data-id="${detection.id}">
          <div class="section-workspace__object-main-data">
            <h4 class="section-workspace__object-title">${detection.label} #${index + 1}</h4>
            <h5 class="section-workspace__object-sub-title">
              ID: det_${detection.id.toString().padStart(4, '0')}
            </h5>
          </div>
          <div class="section-workspace__object-sub-data ${confidenceClass}">
            ${confidencePercent}%
          </div>
        </button>
      `;
     
     // Add click handler to select the detection
     li.querySelector('button').addEventListener('click', () => {
       detectionsModule.select(detection.id);
       // Update active state in object list
       document.querySelectorAll('.section-workspace__object-button').forEach(btn => {
         btn.classList.remove('section-workspace__object-button--selected');
       });
       li.querySelector('button').classList.add('section-workspace__object-button--selected');
     });
     
     list.appendChild(li);
   });
   
   // If there's a selected detection, highlight it in the list
   const selectedDetection = detectionsModule.getDetection(detectionsModule.selectedId);
   if (selectedDetection) {
     const selectedButton = list.querySelector(`.section-workspace__object-button[data-id="${selectedDetection.id}"]`);
     if (selectedButton) {
       selectedButton.classList.add('section-workspace__object-button--selected');
     }
   }
 }

function loadImage(img) {
  currentImage = img;
  detectionsModule.setImage(`http://localhost:8000${img.url}`);
  
  // Create DB record if image doesn't have ID yet
  if (!img.id) {
    createImageRecord(img);
  } else {
    loadDetections(img.id);
  }
}

async function createImageRecord(img) {
  try {
    const res = await fetch(`http://localhost:8000/api/datasets/${dataset.id}/create-image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: img.filename })
    });
    
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    
    const data = await res.json();
    img.id = data.id;
    currentImage = img;
    loadDetections(img.id);
  } catch (err) {
    console.error('Failed to create image record:', err);
    detectionsModule.load([]);
  }
}

async function loadDetections(imageId) {
   try {
     const res = await fetch(`http://localhost:8000/api/images/${imageId}/detections`);
     if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
     const data = await res.json();
     detectionsModule.load(data);
     renderObjectList();
   } catch (err) {
     console.error('Failed to load detections:', err);
     detectionsModule.load([]);
     renderObjectList();
   }
 }

async function saveDetections(imageId) {
  if (!imageId) return;
  
  const detections = detectionsModule.getAll();
  
  try {
    const res = await fetch(`http://localhost:8000/api/images/${imageId}/detections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(detections)
    });
    
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    
    const result = await res.json();
    console.log(`Saved ${result.count} detections`);
    return true;
  } catch (err) {
    console.error('Failed to save detections:', err);
    return false;
  }
}

// Save detections button
document.querySelector('.section-workspace__toolbar-server-button--recheck')?.addEventListener('click', async () => {
  if (!currentImage) {
    Notify.error('Нет выбранного изображения');
    return;
  }
  
  const success = await saveDetections(currentImage.id);
  if (success) {
    Notify.success('Разметка сохранена успешно!');
    // Move to next image
    currentImageIndex++;
    if (currentImageIndex < currentImages.length) {
      loadImage(currentImages[currentImageIndex]);
      renderImagesList();
    } else {
      // Reload category to get fresh images
      loadImagesForCategory();
    }
  } else {
    Notify.error('Ошибка при сохранении разметки');
  }
});

// Auto markup button
document.querySelector('.section-workspace__toolbar-server-button--auto-markup')?.addEventListener('click', async () => {
  const input = document.querySelector('.section-workspace__auto-markup-input');
  const count = parseInt(input?.value) || 1;
  
  if (count <= 0) {
    Notify.warning('Введите корректное количество изображений');
    return;
  }
  
  // Get unlabeled images
  const category = CATEGORIES.find(c => c.codes && c.codes.includes('unlabeled'));
  if (!category) {
    Notify.error('Категория "unlabeled" не найдена');
    return;
  }
  
  try {
    // Fetch unlabeled images
    const res = await fetch(`http://localhost:8000/api/datasets/${dataset.id}/images?status=unlabeled`);
    if (!res.ok) throw new Error(`Failed to fetch images: ${res.status}`);
    
    const images = await res.json();
    
    if (images.length === 0) {
      Notify.warning('Нет неразмеченных изображений');
      return;
    }
    
    // Take first N images
    const imagesToProcess = images.slice(0, Math.min(count, images.length));
    
    Notify.success(`Начинаем авторазметку ${imagesToProcess.length} изображений...`);

    let successCount = 0;
    let errorCount = 0;
    let autoAcceptedCount = 0;
    
    // Process each image
    for (const img of imagesToProcess) {
      try {
        // Call predict API
        const predictRes = await fetch(`http://localhost:8000/api/predict/${dataset.name}/${img.filename}`, {
          method: 'POST'
        });
        
        if (!predictRes.ok) {
          console.error(`Prediction failed for ${img.filename}:`, predictRes.status);
          errorCount++;
          continue;
        }
        
        const predictions = await predictRes.json();
        console.log(`Predictions for ${img.filename}:`, predictions);
        
        // Convert predictions to detections format
        const detections = predictions.map((p, index) => ({
          id: p.id || (index + 1),
          class_id: p.class_id,
          label: p.label || "Object",
          cls: "blue",
          conf: p.conf,
          x1: p.x1,
          y1: p.y1,
          x2: p.x2,
          y2: p.y2
        }));
        
        // Save detections with is_auto=true flag
        const saveRes = await fetch(`http://localhost:8000/api/images/${img.id}/detections?is_auto=true`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(detections)
        });

        if (!saveRes.ok) {
          console.error(`Save failed for ${img.filename}:`, saveRes.status);
          errorCount++;
          continue;
        }

        // Auto-accept if all boxes pass the confidence threshold
        const allAboveThreshold = detections.length > 0 &&
          detections.every(d => d.conf >= confFilterThreshold);

        if (confFilterEnabled && allAboveThreshold) {
          const acceptRes = await fetch(`http://localhost:8000/api/images/${img.id}/accept`, { method: 'POST' });
          if (acceptRes.ok) {
            autoAcceptedCount++;
          }
        }

        successCount++;

      } catch (err) {
        console.error(`Error processing ${img.filename}:`, err);
        errorCount++;
      }
    }
    
    const autoMsg = (confFilterEnabled && autoAcceptedCount > 0)
      ? `\nАвтопринято как GT: ${autoAcceptedCount}`
      : '';
    Notify.success(`Авторазметка завершена!\nУспешно: ${successCount}${autoMsg}\nОшибок: ${errorCount}`);
    
    // Reload current category to show updated images
    loadImagesForCategory();
    
  } catch (err) {
    console.error('Auto markup failed:', err);
    Notify.error(`Ошибка авторазметки: ${err.message}`);
  }
});

// Decision buttons (accept/reject)
document.querySelector('.section-workspace__decision-button--accept')?.addEventListener('click', async () => {
  if (!currentImage) return;
  
  try {
    const res = await fetch(`http://localhost:8000/api/images/${currentImage.id}/accept`, {
      method: 'POST'
    });
    
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    
    // Move to next image
    currentImages.splice(currentImageIndex, 1);
    if (currentImages.length > 0) {
      if (currentImageIndex >= currentImages.length) {
        currentImageIndex = currentImages.length - 1;
      }
      loadImage(currentImages[currentImageIndex]);
      renderImagesList();
    } else {
      loadImagesForCategory();
    }
  } catch (err) {
    console.error('Failed to accept image:', err);
  }
});

document.querySelector('.section-workspace__decision-button--reject')?.addEventListener('click', async () => {
  if (!currentImage) return;
  
  try {
    const res = await fetch(`http://localhost:8000/api/images/${currentImage.id}/reject`, {
      method: 'POST'
    });
    
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    
    // Move to next image
    currentImages.splice(currentImageIndex, 1);
    if (currentImages.length > 0) {
      if (currentImageIndex >= currentImages.length) {
        currentImageIndex = currentImages.length - 1;
      }
      loadImage(currentImages[currentImageIndex]);
      renderImagesList();
    } else {
      loadImagesForCategory();
    }
  } catch (err) {
    console.error('Failed to reject image:', err);
  }
});

// Confidence filter settings
let confFilterEnabled = false;
let confFilterThreshold = 0.85;

async function loadConfFilterConfig() {
  try {
    const res = await fetch(`http://localhost:8000/api/datasets/${dataset.id}/augmentation`);
    if (!res.ok) return;
    const data = await res.json();
    confFilterEnabled = data.augmentation_enabled ?? false;
    confFilterThreshold = data.augmentation_threshold ?? 0.85;

    const checkbox = document.getElementById('conf-filter-enabled');
    const thresholdInput = document.getElementById('conf-filter-threshold');
    if (checkbox) checkbox.checked = confFilterEnabled;
    if (thresholdInput) thresholdInput.value = confFilterThreshold;
    updateConfFilterControlsVisibility();
  } catch (err) {
    console.error('Failed to load conf filter config:', err);
  }
}

function updateConfFilterControlsVisibility() {
  const controls = document.getElementById('conf-filter-controls');
  if (controls) controls.style.display = confFilterEnabled ? 'flex' : 'none';
}

document.getElementById('conf-filter-enabled')?.addEventListener('change', (e) => {
  confFilterEnabled = e.target.checked;
  updateConfFilterControlsVisibility();
});

document.getElementById('conf-filter-save')?.addEventListener('click', async () => {
  const thresholdInput = document.getElementById('conf-filter-threshold');
  const threshold = parseFloat(thresholdInput?.value);
  if (isNaN(threshold) || threshold < 0 || threshold > 1) {
    Notify.error('Порог должен быть числом от 0 до 1');
    return;
  }
  confFilterThreshold = threshold;
  try {
    const res = await fetch(`http://localhost:8000/api/datasets/${dataset.id}/augmentation`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ augmentation_enabled: confFilterEnabled, augmentation_threshold: confFilterThreshold })
    });
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    Notify.success('Настройки порога сохранены');
  } catch (err) {
    Notify.error('Ошибка при сохранении настроек');
  }
});

// Initialize
loadConfFilterConfig();
updateCategory();

// Expose DetectionOverlay for backward compatibility
window.DetectionOverlay = {
  load: (data) => detectionsModule.load(data),
  hide: (id) => detectionsModule.hide(id),
  show: (id) => detectionsModule.show(id),
  select: (id) => detectionsModule.select(id),
  delete: (id) => detectionsModule.delete(id),
  getAll: () => detectionsModule.getAll(),
  setImage: (src) => detectionsModule.setImage(src)
};
