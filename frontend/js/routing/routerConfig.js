export const loadPageTime = 200;

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
    styles: [
      '/css/workPage/sectionWorkSpace.css',
      '/css/workPage/sectionWorkSpaceAside.css'
    ],
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
      '/css/statsPage/sectionStatsGraphs.css',
      '/css/statsPage/sectionStatsLastActivity.css'
    ],
    scripts: [
      '/js/stats.js',
    ]  
  },
};
