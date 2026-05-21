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
  document.getElementById('stats-dataset-name').textContent = stats.dataset_name;

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

let lineChart = null;
let donutChart = null;

function renderLineChart(metricsHistory) {
  const ctx = document.getElementById('averageMarkupPercentGraphCanvas');
  if (lineChart) lineChart.destroy();

  const hasData = metricsHistory.length > 0;
  const labels = hasData ? metricsHistory.map(m => `v${m.version}`) : ['Нет данных'];
  const data = hasData ? metricsHistory.map(m => m.f1 != null ? Math.round(m.f1 * 100) : null) : [null];

  lineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data,
        borderColor: '#6567F1',
        borderWidth: 3,
        tension: 0.4,
        fill: true,
        backgroundColor: (ctx) => {
          const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 300);
          gradient.addColorStop(0, 'rgba(101, 103, 241, 0.3)');
          gradient.addColorStop(1, 'rgba(101, 103, 241, 0)');
          return gradient;
        },
        pointRadius: 4,
        spanGaps: true,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
          ticks: { color: '#6b7280' },
        },
        y: {
          display: true,
          grid: { display: false },
          min: 0,
          max: 100,
          ticks: { color: '#6b7280' },
        },
      },
    },
  });
}

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

async function init() {
  const stats = await loadStats();
  if (!stats) {
    document.getElementById('stats-dataset-name').textContent = 'Ошибка загрузки данных';
    return;
  }
  renderCounters(stats);
  renderClassList(stats.class_distribution);
  renderLineChart(stats.metrics_history);
  renderDonutChart(stats.class_distribution);
}

init();
