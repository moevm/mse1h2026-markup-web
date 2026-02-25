export const routes = {
  '/':        {
    file: 'pagesContent/home.html',
    styles: [
      '/css/homePage/sectionWelcome.css',
      '/css/homePage/sectionDatasetsChoose.css',
      '/css/homePage/sectionRecentDatasets.css'
    ],
    scripts: []  
  },
  '/work':{
    file: 'pagesContent/work.html',
    styles: [],
    scripts: []  
  },
  '/datasets':{
    file: 'pagesContent/datasets.html',
    styles: [
      '/css/datasetsPage//datasetsPage.css',
      '/css/datasetsPage/sectionDatasetsHeader.css',
      '/css/datasetsPage/sectionDatasetsCards.css'
    ],
    scripts: []  
  },
  '/stats':{
    file: 'pagesContent/stats.html',
    styles: [
      '/css/statsPage/sectionStatsContainers.css',
      '/css/statsPage/sectionStatsGraphs.css'
    ],
    scripts: [
      '/js/stats.js',
    ]  
  },
};

export const loadPageTime = 200;