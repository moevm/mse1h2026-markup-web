import { createPopup } from "./popup.js";

async function selectFolder() {
  const res = await fetch('/utils/select-folder');
  const { path } = await res.json();
  if (!path) return;
  return path;
}

export async function uploadNewDataset() {
  const path = await selectFolder();
  if (!path) return;

  createPopup(`
    <h2 class="popup__title">Загрузка нового датасета</h2>
    <p class="popup__sub">Выбранный путь: <span class="popup__path">${path}</span></p>
    <div class="popup__field">
      <label class="popup__label">Имя датасета:</label>
      <input class="popup__input" id="datasetNameInput" placeholder="Введите имя датасета" />
    </div>
    <div class="popup__actions">
      <button class="popup__btn popup__btn--addDataset">Продолжить</button>
    </div>
  `);

  document.querySelector('.popup__btn--addDataset').addEventListener('click', async () => {
    const name = document.getElementById('datasetNameInput').value.trim();
    if (!name) {
      document.getElementById('datasetNameInput').focus();
      return;
    }

    const res = await fetch('http://localhost:8000/api/addDataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_name: name, path }),
    });

    if (!res.ok) {
      return;
    }

    window.location.href = `/datasets`;
  });
}