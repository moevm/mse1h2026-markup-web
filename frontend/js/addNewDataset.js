import { createPopup } from "./popup.js";
import { Notify } from "./utils/notify.js";

async function selectFolder() {
  try {
    const res = await fetch('/utils/select-folder');
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);

    const data = await res.json();
    if (!data.path) {
      Notify.error("Путь не выбран");
      return null;
    }

    return data.path;
  } catch (err) {
    Notify.error("Не удалось выбрать папку. Проверьте сервер.");
    return null;
  }
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

    try {
      const res = await fetch('http://localhost:8000/api/addDataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_name: name, path }),
      });

      if (!res.ok) {
        Notify.error(`Ошибка API: ${res.status}`);
        return;
      }

      Notify.success("Датасет успешно добавлен");
      window.location.href = `/datasets`;
    } catch (err) {
      Notify.error("Не удалось подключиться к серверу. Проверьте его работу.");
    }
  });
}