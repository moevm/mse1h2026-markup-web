const currentDataset = sessionStorage.getItem('currentDataset') || localStorage.getItem('currentDataset');
if (!currentDataset) {
  window.location.href = '/';
}

const dataset = JSON.parse(currentDataset);

async function loadStats() {
  try {
    const res = await fetch(`http://localhost:8000/api/datasets/${dataset.id}/stats`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Failed to load stats:', err);
    return null;
  }
}

function renderCounters(stats) {
  document.getElementById('stats-dataset-name').textContent = `Статистика по датасету: ${stats.dataset_name}`;

  const total = stats.total_images;
  document.getElementById('stat-total').textContent = total.toLocaleString('ru-RU');
  document.getElementById('stat-labeled').textContent = stats.labeled.toLocaleString('ru-RU');
  document.getElementById('stat-remaining').textContent = stats.unlabeled.toLocaleString('ru-RU');
  document.getElementById('stat-confidence').textContent =
    stats.avg_confidence !== null ? `${stats.avg_confidence}%` : '—';

  document.getElementById('range-labeled').value =
    total > 0 ? (stats.labeled / total * 100) : 0;
  document.getElementById('range-remaining').value =
    total > 0 ? (stats.unlabeled / total * 100) : 0;
  document.getElementById('range-confidence').value = stats.avg_confidence ?? 0;

  const totalBoxes = stats.total_boxes;
  document.getElementById('stat-total-boxes').textContent =
    totalBoxes >= 1_000_000
      ? `${(totalBoxes / 1_000_000).toFixed(1)}M`
      : totalBoxes.toLocaleString('ru-RU');
}

function renderClassList(distribution) {
  const list = document.getElementById('classes-list');
  list.innerHTML = '';

  if (distribution.length === 0) {
    list.innerHTML = '<li class="section-stats-graphs__list-item">Нет данных</li>';
    return;
  }

  const modifiers = ['--top', '--center', '--bottom'];
  distribution.slice(0, 3).forEach((cls, i) => {
    const li = document.createElement('li');
    li.className = `section-stats-graphs__list-item section-stats-graphs__list-item${modifiers[i] ?? ''}`;
    li.innerHTML = `
      <span class="section-stats-graphs__list-class-name">${cls.name}</span>
      <span class="section-stats-graphs__list-class-percent">${cls.percent}%</span>
    `;
    list.appendChild(li);
  });
}

let donutChart = null;

function renderDonutChart(distribution) {
  const ctx = document.getElementById('donutGraphCanvas').getContext('2d');
  if (donutChart) donutChart.destroy();

  const COLORS = ['#6567F1', '#F15A5A', '#F1A041', '#3DCF8A', '#C47FFF'];
  const top5 = distribution.slice(0, 5);

  const labels = top5.length > 0 ? top5.map(c => c.name) : ['Нет данных'];
  const values = top5.length > 0 ? top5.map(c => c.count) : [1];
  const colors = top5.length > 0 ? top5.map((_, i) => COLORS[i % COLORS.length]) : ['#334155'];

  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderWidth: 0, hoverOffset: 4 }],
    },
    options: {
      cutout: '75%',
      plugins: { legend: { display: false }, tooltip: { enabled: true } },
    },
  });
}

function formatRelativeTime(isoString) {
  const diff = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (diff < 60) return `${diff} сек. назад`;
  if (diff < 3600) return `${Math.floor(diff / 60)} мин. назад`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} ч. назад`;
  return `${Math.floor(diff / 86400)} дн. назад`;
}

async function init() {
  const stats = await loadStats();
  if (!stats) {
    document.getElementById('stats-dataset-name').textContent = 'Ошибка загрузки данных';
    return;
  }
  renderCounters(stats);
  renderClassList(stats.class_distribution);
  renderDonutChart(stats.class_distribution);
}

init();
