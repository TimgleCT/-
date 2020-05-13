
//一些全域變數，紀錄我的最愛欄位是否有值
var max = 10 //最大欄位數，可自己設定
var stopsdata
var myStops = [max]

console.log("初始化")
//抓取站牌資料的方選
function getStop() {
    var stopname = document.querySelector('.stop').value;
    var xhr = new XMLHttpRequest();
    xhr.open("get", "https://ptx.transportdata.tw/MOTC/v2/Bus/Stop/City/Kaohsiung?$format=json", true);
    xhr.send();

    xhr.onload = function () {
        var data = JSON.parse(xhr.responseText);
        stopsdata = data
        var stops = "<tr> <td>站名</td>  <td>儲存</td> </tr>"
        for (var i = 0; i < 10250; i++) {
            if (data[i].StopName.Zh_tw == stopname) {
                zhname = data[i].StopName.Zh_tw
                stops += "<tr> <td>" + data[i].StopName.Zh_tw + "</td> <td><input type='button' class='stopID' onclick='checkStop(" + i + ")' value='新增至常用站牌'></td> </tr>"
                break
            }
          
        }
        document.querySelector('.result').innerHTML = stops;
    }
    console.log("抓站牌")
}

//寫入localstorage的方選
function checkStop(objectID) {
    var i = 0
    while (i < max) {
        if (localStorage.getItem(i) !== null) {
            i++
        }
        else {
            storeStop( i, objectID)
            break
        }
    }
}
function storeStop( i, objectID) {
    
    //執行寫入
    localStorage.setItem(i, stopsdata[objectID].StopName.Zh_tw)
    console.log(stopsdata[objectID].StopName.Zh_tw)
    

}



//刪除個別站牌的方選
function delStop(localstorageID ) {
    localStorage.removeItem(localstorageID)
    var str =""
    var j = 0;
    while (j < max) {
        if (localStorage.getItem(j) !== null) {
         //   str += "<tr><td><div class='stop-collect'>" + localStorage.getItem(j) + "</div></td> <td><input type='button' onclick='delStop(" + j + ")' value='刪除自常用站牌'></td> </tr>"
         str +=  "<tbody><tr><td><a href='busstop.html'><div class='stop-collect'><h3>" + localStorage.getItem(j) + "</h3><p>301A,301B,紅53D</p></div></a></td><td><button type='button' class='del' onclick='delStop(" + j + ")'>刪除</button></td></tr></tbody>"
        }
        j++
    }
   
    document.getElementById("seeMyStops").innerHTML = str;
    

}

//清除全部localstorage的方選
function del_click() {
    //執行清除
    localStorage.clear();
    console.log("執行全部刪除")
    //傳回刪除後的結果
    var str = "<tr><td>常用的站牌</td><td>刪除</td></tr>"
    document.getElementById("myStop").innerHTML = str
    
}

function seeMyStops(){
    var str =""
    var j = 0;
    while (j < max) {
        if (localStorage.getItem(j) !== null) {
         //   str += "<tr><td><div class='stop-collect'>" + localStorage.getItem(j) + "</div></td> <td><input type='button' onclick='delStop(" + j + ")' value='刪除自常用站牌'></td> </tr>"
         str +=  "<tbody><tr><td><a href='busstop.html'><div class='stop-collect'><h3>" + localStorage.getItem(j) + "</h3><p>301A,301B,紅53D</p></div></a></td><td><button type='button' class='del' onclick='delStop(" + j + ")'>刪除</button></td></tr></tbody>"
        }
        j++
    }
   
    document.getElementById("seeMyStops").innerHTML = str;
    
}
