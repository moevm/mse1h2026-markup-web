export class NotifyCore {
  static container = null;

  static init() {
    if (!this.container) {
      this.container = document.createElement("div");
      this.container.style.position = "fixed";
      this.container.style.top = "20px";
      this.container.style.right = "20px";
      this.container.style.zIndex = "9999";
      document.body.appendChild(this.container);
    }
  }

  static show({ title, text, type = "info", duration = 4000 }) {
    this.init();

    const el = document.createElement("div");
    el.style.background = "#333";
    el.style.color = "#fff";
    el.style.padding = "12px 16px";
    el.style.marginTop = "10px";
    el.style.borderRadius = "8px";
    el.style.minWidth = "250px";
    el.style.fontFamily = "sans-serif";
    el.style.boxShadow = "0 4px 10px rgba(0,0,0,0.2)";
    el.style.cursor = "pointer";
    el.style.opacity = "0";
    el.style.transition = "opacity 0.3s";

    if (type === "success") el.style.background = "#2ecc71";
    if (type === "error") el.style.background = "#e74c3c";
    if (type === "warning") el.style.background = "#f39c12";

    el.innerHTML = `
      ${title ? `<div style="font-weight: bold; margin-bottom: 4px;">${title}</div>` : ""}
      <div>${text || ""}</div>
    `;

    this.container.appendChild(el);

    // fade in
    requestAnimationFrame(() => {
      el.style.opacity = "1";
    });

    const remove = () => {
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 300);
    };

    // авто-закрытие
    setTimeout(remove, duration);

    // закрытие по клику
    el.addEventListener("click", remove);
  }
}

export const Notify = {
  success(text, title = "Готово!") {
    NotifyCore.show({ title, text, type: "success" });
  },

  error(err) {
    const text =
      typeof err === "string"
        ? err
        : err?.response?.data?.message || err?.message || "Ошибка";

    NotifyCore.show({
      title: "Ошибка",
      text,
      type: "error",
    });
  },

  warning(text, title = "Внимание") {
    NotifyCore.show({ title, text, type: "warning" });
  },

  info(text, title = "Информация") {
    NotifyCore.show({ title, text, type: "info" });
  },
};