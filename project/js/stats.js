function addGraph() {
  const ctx = document.getElementById('myChart');

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['DS1', 'DS2', 'DS3', 'DS4', 'DS5'],
      datasets: [{
        data: [40, 30, 65, 25, 95],
        borderColor: '#6567F1',
        borderWidth: 13,
        tension: 0.4,
        fill: true,
        backgroundColor: (ctx) => {
          const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 300);
          gradient.addColorStop(0, 'rgba(101, 103, 241, 0.3)');
          gradient.addColorStop(1, 'rgba(101, 103, 241, 0)');
          return gradient;
        },
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(255,255,255,0.05)',
            drawBorder: false,
          },
          ticks: { color: '#6b7280' },
        },
        y: {
          display: true,
          grid: { display: false },
        }
      }
    }
  });
}

function donnut() {
  const ctx = document.getElementById('doughnutChart').getContext('2d');

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [65, 25, 10],
        backgroundColor: ['#6567F1', '#F15A5A', '#F1A041'],
        borderWidth: 0,
        hoverOffset: 4,
      }]
    },
    options: {
      cutout: '75%',
      plugins: {
        legend: { display: true },
        tooltip: { enabled: true },
      }
    }
  });
}

// addGraph();

// donnut();


