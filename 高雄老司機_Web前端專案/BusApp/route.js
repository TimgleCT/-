var stop_go = new Array();
var stop_back = new Array();
var result_go = "";
var result_back = "";

var urls = encodeURIComponent(location.href);
var url = decodeURIComponent(urls);
console.log(url);
var temp = url.split("?");
var vars = temp[1];
var kw = decodeURIComponent(vars);
console.log(kw);


function get_RouteOfStop_go(){
    var get = new XMLHttpRequest();
    var header = GetAuthorizationHeader();
    get.open('get',"https://ptx.transportdata.tw/MOTC/v2/Bus/StopOfRoute/City/Kaohsiung?$top=30&$format=JSON",true);
    get.setRequestHeader('Authorization', header['Authorization']);
    get.setRequestHeader('X-Date', header['X-Date']);
   
    get.send();
    get.onload = function(){
        var data = JSON.parse(get.responseText);

        for(var i = 0;data.length>i;i++){
            if(data[i].Direction==0){
                for(var j = 0;data[i].Stops.length>j;j++){
                    if(data[i].Stops[j].StopName.Zh_tw==kw){
                        stop_go[j] = data[i].RouteName.Zh_tw;
                        result_go += '<tr><td><div class="arrived-name"><h3>'+stop_go[j]+'號</h3><div></td></tr>';
                        console.log(j);
                        console.log(stop_go[j]);
                    }
                }
            }
        }
/*        for(var k = 0;stop_go.length>k;k++){
            result_go += '<li>'+stop_go[k]+'號</li>';
            console.log(k);
            console.log(stop_go[k]);
    }*/

        document.querySelector('.go_list').innerHTML = result_go;
    }    
}

function get_RouteOfStop_back(){
    var get = new XMLHttpRequest();
    var header = GetAuthorizationHeader();
    get.open('get',"https://ptx.transportdata.tw/MOTC/v2/Bus/StopOfRoute/City/Kaohsiung?$top=30&$format=JSON",true);
    get.setRequestHeader('Authorization', header['Authorization']);
    get.setRequestHeader('X-Date', header['X-Date']);
   
    get.send();
    get.onload = function(){
        var data = JSON.parse(get.responseText);

        for(var i = 0;data.length>i;i++){
            if(data[i].Direction==1){
                for(var j = 0;data[i].Stops.length>j;j++){
                    if(data[i].Stops[j].StopName.Zh_tw==kw){
                        stop_back[j] = data[i].RouteName.Zh_tw;
                        result_back += '<tr><td><div class="arrived-name"><h3>'+stop_back[j]+'號</h3><div></td></tr>';
                        console.log(j);
                        console.log(stop_back[j]);
                    }
                }
            }
        }
/*        for(var k = 0;stop_go.length>k;k++){
            result_go += '<li>'+stop_go[k]+'號</li>';
            console.log(k);
            console.log(stop_go[k]);
    }*/

        document.querySelector('.back_list').innerHTML = result_back;
    }    
}

function GetAuthorizationHeader() {
    var AppID = 'a0bfe9570c984e3f9acd2def6d3a3fb4';
    var AppKey = '1dL-eDdoRu7my1fLnidLBZRP0Xo';

    var GMTString = new Date().toGMTString();
    var ShaObj = new jsSHA('SHA-1', 'TEXT');
    ShaObj.setHMACKey(AppKey, 'TEXT');
    ShaObj.update('x-date: ' + GMTString);
    var HMAC = ShaObj.getHMAC('B64');
    var Authorization = 'hmac username=\"' + AppID + '\", algorithm=\"hmac-sha1\", headers=\"x-date\", signature=\"' + HMAC + '\"';

    return { 'Authorization': Authorization, 'X-Date': GMTString /*,'Accept-Encoding': 'gzip'*/}; //锟絧锟紾锟絥锟絅js锟紹锟斤拷b锟斤拷锟紸锟斤拷锟紸锟絠锟紹锟絶锟絒锟絁 'Accept-Encoding': 'gzip'锟紸锟絥锟紻锟斤拷锟結锟紿锟斤拷趾锟斤拷强锟斤拷贫q
}

get_RouteOfStop_go();
get_RouteOfStop_back();