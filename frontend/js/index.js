import { initAnimationText } from "./homePage/animationTextHomePage.js";
import { createPopup } from "./popup.js";

function initLogicChooseDatasetButton() {
  const btn = document.querySelector('.section-datasets-choose__button--choice');
  btn.addEventListener('click', () => {
    window.location.href = '/datasets';
  })
}

async function selectFolder() {
  const res = await fetch('/utils/select-folder');
  const { path } = await res.json();
  if (!path) return;
  return path;
}

async function uploadNewDataset() {
  const path  = await selectFolder();
  if (!path) return;
  createPopup(`<h2>Заголовок</h2><p>${path}</p>`);
}


document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('loadBtn').addEventListener('click', uploadNewDataset);
  initAnimationText();
  initLogicChooseDatasetButton();
})