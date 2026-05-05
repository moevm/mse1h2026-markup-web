import { Notify } from "../utils/notify.js";

export const datasetsManager = {
  currentDataset: null,

  getDataset() {
    if (!this.currentDataset) {
      this.currentDataset = sessionStorage.getItem("currentDataset");
    }

    if (!this.currentDataset) {
      Notify.error("Нет выбранного датасета");
      return null;
    }

    return this.currentDataset;
  },

  setDataset(dataset) {
    if (dataset === null || dataset === undefined) {
      Notify.error("Нельзя установить пустой датасет");
      return;
    }

    this.currentDataset = dataset;
    sessionStorage.setItem("currentDataset", dataset);
  },
};