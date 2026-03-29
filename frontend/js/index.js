import { initAnimationText } from "./homePage/animationTextHomePage.js";
import { uploadNewDataset } from "./addNewDataset.js";

function initLogicChooseDatasetButton() {
  const btn = document.querySelector('.section-datasets-choose__button--choice');
  btn.addEventListener('click', () => {
    window.location.href = '/datasets';
  })
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('loadBtn').addEventListener('click', uploadNewDataset);
  initAnimationText();
  initLogicChooseDatasetButton();
})