import { loadPageTime, routes } from "/js/routing/routerConfig.js";

const injected = { styles: [], scripts: [] };

function removeInjected() {
  injected.styles.forEach(el => el.remove());
  injected.scripts.forEach(el => el.remove());
  injected.styles = [];
  injected.scripts = [];
}

function injectStyles(styles = []) {
  styles.forEach(href => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    document.head.appendChild(link);
    injected.styles.push(link);
  });
}

function injectScripts(scripts = []) {
  scripts.forEach(item => {
    const script = document.createElement('script');
    
    if (typeof item === 'string') {
      script.src = item;
      script.defer = true;
    } else {
      script.src = item.src;
      if (item.module) {
        script.type = 'module';
      } else {
        script.defer = true;
      }
    }

    document.body.appendChild(script);
    injected.scripts.push(script);
  });
}

export function initRouter() {
  const app = document.getElementById('app');
  const navLinks = document.querySelectorAll('nav a');

  function getPath() {
    const hash = location.hash.replace(/^#/, '') || '/';
    return hash.startsWith('/') ? hash : '/' + hash;
  }

  async function render() {
    const path = getPath();
    const route = routes[path];

    navLinks.forEach(a => {
      const href = a.getAttribute('href').replace(/^#/, '') || '/';
      a.classList.toggle('active', href === path);
    });

    app.classList.add('out');
    await new Promise(r => setTimeout(r, loadPageTime));

    removeInjected();

    if (!route) {
      app.innerHTML = `
        <div class="not-found">
          <span>404</span>
          <h2>Page not found</h2>
          <p>Маршрут <code>${path}</code> не зарегистрирован.</p>
          <a href="#/" class="btn">← На главную</a>
        </div>`;
    } else {
      const res = await fetch(route.file);
      app.innerHTML = await res.text();

      injectStyles(route.styles);
      injectScripts(route.scripts);
    }

    app.classList.remove('out');
    app.classList.add('in');
    app.addEventListener('animationend', () => app.classList.remove('in'), { once: true });
  }

  window.addEventListener('hashchange', render);
  render();
}