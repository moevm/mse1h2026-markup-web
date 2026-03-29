import { STATUS_TEMPLATE_MAP, MENU_FILTER_MAP } from "/js/datasetsEnums.js";

const DATASETS_MOCK = [
  {
    id: 1,
    name: "Городской трафик",
    status_id: 3,
    total_size: 1500,
    inwork_size: 1200,
    path: "/img/datasetsPage/dataset-preview-trafic.png",
    lastactivity: "2 часа назад",
    average_percent_success: 80,
    current_model_architecture: "yolo11n",
  },
  {
    id: 5,
    name: "Городской трафик",
    status_id: 3,
    total_size: 1500,
    inwork_size: 1200,
    path: "/img/datasetsPage/dataset-preview-trafic.png",
    lastactivity: "2 часа назад",
    average_percent_success: 80,
    current_model_architecture: "yolo11n",
  },
  {
    id: 2,
    name: "Сканы МРТ",
    status_id: 2,
    total_size: 800,
    inwork_size: 450,
    path: "/img/datasetsPage/dataset-preview-mrt.png",
    lastactivity: "вчера",
    average_percent_success: 56,
    current_model_architecture: "yolo11n",
  },
  {
    id: 3,
    name: "Механика",
    status_id: 1,
    total_size: 2100,
    inwork_size: 2100,
    path: "/img/datasetsPage/dataset-preview-engineer.png",
    lastactivity: "3 дня назад",
    average_percent_success: 100,
    current_model_architecture: "yolo11n",
  },
  {
    id: 4,
    name: "Пешеходы",
    status_id: 0,
    total_size: 950,
    inwork_size: 0,
    path: "/img/datasetsPage/dataset-preview-trafic.png",
    lastactivity: "5 дней назад",
    average_percent_success: null,
    current_model_architecture: "yolo11n",
  },
];

function formatTotal(n) {
  return n.toLocaleString("ru-RU");
}

function renderDatasetCard(dataset) {
  const templateId = STATUS_TEMPLATE_MAP[dataset.status_id];
  const template = document.getElementById(templateId);
  if (!template) {
    console.warn(`Template not found: ${templateId}`);
    return null;
  }

  const clone = template.content.cloneNode(true);
  const q = (field) => clone.querySelector(`[data-field="${field}"]`);

  q("dataset-card-preview-image").src = dataset.path;
  q("count-images-text").textContent = formatTotal(dataset.total_size);
  q("dataset-name").textContent = dataset.name;
  q("dataset-last-activity").textContent = `Обновлено ${dataset.lastactivity}`;

  const percent = dataset.average_percent_success ?? 0;
  q("dataset-percent").textContent = `${percent}%`;
  q("dataset-count-fraction").textContent =
    `${formatTotal(dataset.inwork_size)} / ${formatTotal(dataset.total_size)} изображений размечено`;

  const range = q("dataset-range");
  if (range) range.value = percent;

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
          : datasets.filter((d) => d.status_id === filterStatusId);

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
    counters[i].textContent = datasets.filter((d) => d.status_id === statusId).length;
  }
}

async function fetchDatasets() {
  return new Promise((resolve) =>
    setTimeout(() => resolve(DATASETS_MOCK), 300)
  );
}

async function init() {
  const datasets = await fetchDatasets();
  renderDatasets(datasets);
  initFilters(datasets);
  updateMenuCounters(datasets);
}

document.addEventListener("DOMContentLoaded", init);