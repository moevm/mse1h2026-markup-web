import { STATUS_TEMPLATE_MAP, MENU_FILTER_MAP } from "/js/datasetsPage/datasetsEnums.js";
import { Notify } from "./utils/notify.js";
import { uploadNewDataset } from "./utils/addNewDataset.js";
import { datasetsManager } from "./managers/datasetsManager.js";

import { DATASETS_MOCK } from "./develop/mockdata.js";

function formatTotal(n) {
  return n.toLocaleString("ru-RU");
}

function renderDatasetCard(dataset) {
  const statusId = dataset.status?.id ?? dataset.status_id;
  const templateId = STATUS_TEMPLATE_MAP[statusId];
  const template = document.getElementById(templateId);
  
  if (!template) return null;

  const clone = template.content.cloneNode(true);
  const q = (field) => clone.querySelector(`[data-field="${field}"]`);

  const previewImg = q("dataset-card-preview-image");
  if (dataset.preview_image) {
    previewImg.src = `http://localhost:8000${dataset.preview_image}`;
  }
  
  q("count-images-text").textContent = formatTotal(dataset.total_size);
  q("dataset-name").textContent = dataset.name;
  q("dataset-last-activity").textContent = `Обновлено ${dataset.lastactivity || 'недавно'}`;

  const percent = dataset.average_percent_success ?? 0;
  q("dataset-percent").textContent = `${percent}%`;
  q("dataset-count-fraction").textContent =
    `${formatTotal(dataset.inwork_size)} / ${formatTotal(dataset.total_size)} изображений размечено`;

  const range = q("dataset-range");
  if (range) range.value = percent;

  const cardBtn = clone.querySelector(".section-datasets-cards__card-button");
  cardBtn.addEventListener("click", () => {
    datasetsManager.setDataset(JSON.stringify(dataset));
    Notify.success(`Выбран датасет: ${dataset.name}`);
  });

  return clone;
}

function renderDatasets(datasets) {
  const list = document.querySelector(".section-datasets-cards__cards");
  list.innerHTML = "";
  for (const dataset of datasets) {
    const card = renderDatasetCard(dataset);
    if (card) list.appendChild(card);
  }
}

function initFilters(datasets) {
  const buttons = document.querySelectorAll(".section-datasets-cards__menu-button");
  const activeClass = "section-datasets-cards__menu-button--active";

  buttons.forEach((btn, index) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove(activeClass));
      btn.classList.add(activeClass);

      const filterStatusId = MENU_FILTER_MAP[index];
      const filtered =
        filterStatusId === null
          ? datasets
          : datasets.filter((d) => (d.status?.id ?? d.status_id) === filterStatusId);

      renderDatasets(filtered);
    });
  });
}

function updateMenuCounters(datasets) {
  const buttons = document.querySelectorAll(".section-datasets-cards__menu-button");
  const counters = [...buttons].map((btn) =>
    btn.querySelector(".section-datasets-cards__menu-counter")
  );

  counters[0].textContent = datasets.length;

  for (let i = 1; i < buttons.length; i++) {
    const statusId = MENU_FILTER_MAP[i];
    counters[i].textContent = datasets.filter((d) => (d.status?.id ?? d.status_id) === statusId).length;
  }
}

async function fetchDatasets() {
  try {
    const res = await fetch("http://localhost:8000/api/getDatasets");

    if (!res.ok) {
      throw new Error(`HTTP error: ${res.status}`);
    }

    const data = await res.json();

    if (!Array.isArray(data)) {
      throw new Error("Invalid data format");
    }

    return data;
  } catch (err) {
    Notify.error("Ошибка получения датасетов из API");
    console.error(err);
    return [];
  }
}

async function init() {
  const datasets = await fetchDatasets();
  renderDatasets(datasets);
  initFilters(datasets);
  updateMenuCounters(datasets);
  const btn = document.querySelector('.section-datasets-header__button--upload-new-dataset');
  btn.addEventListener('click', uploadNewDataset);
}

document.addEventListener("DOMContentLoaded", init);