window.onload = get_routlistAPI();
var data;
function get_routlistAPI(){
    var get = new XMLHttpRequest();
    var header = GetAuthorizationHeader();
    get.open('get',"https://ptx.transportdata.tw/MOTC/v2/Bus/Route/City/Kaohsiung?$format=json",true);
    get.setRequestHeader('Authorization', header['Authorization']);
    get.setRequestHeader('X-Date', header['X-Date']);
   
    get.send();
    get.onload = function(){
        data = JSON.parse(get.responseText);
        console.log(data);
        var name;
        var first_step;
        var last_step;
        var result = "<table class='route-list'><tbody>";
        for(var i= 0;i<data.length;i++){
            name = data[i].RouteName.Zh_tw;
            first_step = data[i].DepartureStopNameZh;
            last_step = data[i].DestinationStopNameZh;
            result += "<tr><td id = 'name'><a href='routedetail.html?"+name+"'><h3>"+name+"</h3><p>"+first_step+"-"+last_step+"</p></a></td><td><button type='button' onclick='checkRoute(" + i + ")' class='btnheart'><img src='image/unlike.png' alt='unlike'></button></td></tr>";
            //result += "<h3 onclick=\"location.href='show_result.html?"+name+"'\">"+name+"</h3><li>"+first_step+"-"+last_step+"</li>";
        }
        result += "</tbody></table>"
        document.querySelector('.rf-main').innerHTML = result;
    }
}

function step_match(){
    var name;
    var first_step;
    var last_step;
    var result = "<table class='route-list'><tbody>";
    var search = document.getElementById('route').textContent;
    var j = 1;
        for(var i= 0;i<data.length;i++){
            name = data[i].RouteName.Zh_tw;
            first_step = data[i].DepartureStopNameZh;
            last_step = data[i].DestinationStopNameZh;
            var match = name.match(search);
            console.log(name);
            console.log(search);
            console.log(match);
            if(match){
                result += "<tr><td id = 'name'><a href='routedetail.html?"+name+"'><h3>"+name+"</h3><p>"+first_step+"-"+last_step+"</p></a></td><td><button type='button' onclick='checkRoute(" + i + ")' class='btnheart'><img src='image/unlike.png' alt='unlike'></button></td></tr>";
                //result += "<h3 id=\""+j+"\" onclick=\"location.href='show_result.html?"+name+"'\">"+name+"</h3><li>"+first_step+"-"+last_step+"</li>";
                j++;
            }
        }
        result += "</tbody></table>"
        document.querySelector('.rf-main').innerHTML = result; 
    
}
function GetAuthorizationHeader() {
    var AppID = '8651028858fd4e40af317a75674d8cb5';
    var AppKey = 'Fk3eCwv0QM34ihzk1ORkfw3MFUQ';

    var GMTString = new Date().toGMTString();
    var ShaObj = new jsSHA('SHA-1', 'TEXT');
    ShaObj.setHMACKey(AppKey, 'TEXT');
    ShaObj.update('x-date: ' + GMTString);
    var HMAC = ShaObj.getHMAC('B64');
    var Authorization = 'hmac username=\"' + AppID + '\", algorithm=\"hmac-sha1\", headers=\"x-date\", signature=\"' + HMAC + '\"';

    return { 'Authorization': Authorization, 'X-Date': GMTString }; //如果要將js運行在伺服器，可額外加入 'Accept-Encoding': 'gzip'，要求壓縮以減少網路傳輸資料量
    
}

