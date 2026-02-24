import { loadPageTime, routes } from "/js/routing/routerConfig.js";

export function initRouter() {

  const app = document.getElementById('app');
  const navLinks = document.querySelectorAll('nav a');

  function getPath() {
    const hash = location.hash.replace(/^#/, '') || '/';
    return hash.startsWith('/') ? hash : '/' + hash;
  }

  async function render() {
    const path = getPath();
    const file = routes[path];

    navLinks.forEach(a => {
      const href = a.getAttribute('href').replace(/^#/, '') || '/';
      a.classList.toggle('active', href === path);
    });

    app.classList.add('out');
    await new Promise(r => setTimeout(r, loadPageTime));

    if (!file) {
      app.innerHTML = `
        <div class="not-found">
          <span>404</span>
          <h2>Page not found</h2>
          <p>Маршрут <code>${path}</code> не зарегистрирован.</p>
          <a href="#/" class="btn">← На главную</a>
        </div>`;
    } else {
      const res = await fetch(file);
      app.innerHTML = await res.text();
    }
    app.classList.remove('out');
    app.classList.add('in');
    app.addEventListener('animationend', () => app.classList.remove('in'), { once: true });
  }
  window.addEventListener('hashchange', render);
  render();
}