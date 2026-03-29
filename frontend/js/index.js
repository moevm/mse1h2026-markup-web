import { initAnimationText } from "./homePage/animationTextHomePage.js";
import { uploadNewDataset } from "./addNewDataset.js";
import { createPopup } from "./popup.js";

function initLogicChooseDatasetButton() {
  const btn = document.querySelector('.section-datasets-choose__button--choice');
  btn.addEventListener('click', () => {
    window.location.href = '/datasets';
  })
}

const mockDatasets = [
  {
    id: 1,
    name: 'Архитектура Датасет',
    total_size: 1240,
    created: '2 часа назад',
    percent: 65,
    status_text: '65% размечено',
    preview: '/img/homePage/dataset-image-preview-architect.png',
  },
  {
    id: 2,
    name: 'Ночной Датасет',
    total_size: 450,
    created: 'вчера',
    percent: 100,
    status_text: 'завершено',
    preview: '/img/homePage/dataset-image-preview-night.png',
  },
  {
    id: 3,
    name: 'Дроны Датасет',
    total_size: 3800,
    created: '3 дня назад',
    percent: 12,
    status_text: '12% размечено',
    preview: '/img/homePage/dataset-image-preview-drons.png',
  },
  {
    id: 4,
    name: 'Городской Датасет',
    total_size: 2100,
    created: '5 дней назад',
    percent: 45,
    status_text: '45% размечено',
    preview: '/img/homePage/dataset-image-preview-architect.png',
  },
  {
    id: 5,
    name: 'Природа Датасет',
    total_size: 980,
    created: 'неделю назад',
    percent: 80,
    status_text: '80% размечено',
    preview: '/img/homePage/dataset-image-preview-night.png',
  },
  {
    id: 6,
    name: 'Транспорт Датасет',
    total_size: 5600,
    created: '2 недели назад',
    percent: 30,
    status_text: '30% размечено',
    preview: '/img/homePage/dataset-image-preview-drons.png',
  },
  {
    id: 7,
    name: 'Медицина Датасет',
    total_size: 750,
    created: '3 недели назад',
    percent: 100,
    status_text: 'завершено',
    preview: '/img/homePage/dataset-image-preview-architect.png',
  },
  {
    id: 8,
    name: 'Промышленность Датасет',
    total_size: 4200,
    created: 'месяц назад',
    percent: 55,
    status_text: '55% размечено',
    preview: '/img/homePage/dataset-image-preview-night.png',
  },
  {
    id: 9,
    name: 'Спорт Датасет',
    total_size: 1850,
    created: 'месяц назад',
    percent: 0,
    status_text: 'не размечено',
    preview: '/img/homePage/dataset-image-preview-drons.png',
  },
  {
    id: 10,
    name: 'Животные Датасет',
    total_size: 3300,
    created: '2 месяца назад',
    percent: 90,
    status_text: '90% размечено',
    preview: '/img/homePage/dataset-image-preview-architect.png',
  },
];

function renderDatasetItem(dataset) {
  const template = document.getElementById('dataset-item-template');
  const clone = template.content.cloneNode(true);

  clone.querySelector('.section-recent-datasets__image-dataset-preview').src = dataset.preview;
  clone.querySelector('.section-recent-datasets__dataset-title').textContent = dataset.name;
  clone.querySelector('.section-recent-datasets__dataset-count-images').textContent = `${dataset.total_size.toLocaleString()} изображений`;
  clone.querySelector('.section-recent-datasets__dataset-subdata-text--time').textContent = `Создан ${dataset.created}`;
  clone.querySelector('.section-recent-datasets__input-status-markup').value = dataset.percent;
  clone.querySelector('.section-recent-datasets__datasets-status-text').textContent = dataset.status_text;
  clone.querySelector('.section-recent-datasets__button-goto').dataset.id = dataset.id;

  return clone;
}

function initRecentDatasets() {
  const list = document.querySelector('.section-recent-datasets__datasets-list');
  if (!list) return;

  list.innerHTML = '';
  mockDatasets.splice(0,3).forEach(dataset => list.appendChild(renderDatasetItem(dataset)));

  list.querySelectorAll('.section-recent-datasets__button-goto').forEach(btn => {
    btn.addEventListener('click', () => {
      window.location.href = `/datasets/${btn.dataset.id}`;
    });
  });
}

function initShowAllDatasets() {
  const btn = document.querySelector('.section-recent-datasets__button-show-all');
  if (!btn) return;

  btn.addEventListener('click', () => {
    const overlay = createPopup(`
      <h2 class="popup__title">Недавние датасеты</h2>
      <ul class="section-recent-datasets__datasets-list popup-datasets-list"></ul>
      <button class="popup__btn popup__btn--goto-datasets">Перейти к датасетам</button>
    `);

    const list = overlay.querySelector('.popup-datasets-list');
    mockDatasets.forEach(dataset => list.appendChild(renderDatasetItem(dataset)));

    list.querySelectorAll('.section-recent-datasets__button-goto').forEach(b => {
      b.addEventListener('click', () => {
        window.location.href = `/datasets/${b.dataset.id}`;
      });
    });

    overlay.querySelector('.popup__btn--goto-datasets').addEventListener('click', () => {
      window.location.href = '/datasets';
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('loadBtn').addEventListener('click', uploadNewDataset);
  initAnimationText();
  initLogicChooseDatasetButton();
  initRecentDatasets();
  initShowAllDatasets();
})