簡介：

網頁版的高雄公車通

    利用PTX交通部開放資料提供的高雄公⾞動態資訊與路線資料，參考高雄公車通 App 的功能，將其以 Javascript 仿製出來，可以說是網頁板的高雄公車通。高雄老司機可查詢公⾞站牌、路線的即時的動態資訊、收藏站牌與路線，路線動態資訊由即時向開放資料的API取得，而收藏功能使用 local storage 實作。高雄公車通也已將服務放在 GitHub Pages上。


GitHub網址：https://github.com/TimgleCT/TimgleCT.github.io/tree/master/BusApp
高雄老司機網址：https://timglect.github.io/BusApp/

本人負責部分：負責路線查詢的部分。向開放資料 API 取得路線、站牌、即時動態資料，並使其能夠隨著使用者的輸入，動態篩選出使用者欲查的路線、定時更新頁面以顯示最新資訊。