setTimeout("window.onload=answer()",10);
var go_step = new Array();
var back_step = new Array();
var go_time = new Array();
var back_time = new Array();
var r = 0;
var k = 0;
var route;
var data;


function getRouteStep(){
    //go_step.length = 0;
    //back_step.length = 0;
    var get = new XMLHttpRequest();
    var header = GetAuthorizationHeader();
    get.open('get',"https://ptx.transportdata.tw/MOTC/v2/Bus/StopOfRoute/City/Kaohsiung?$format=JSON",true);
    get.setRequestHeader('Authorization', header['Authorization']);
    get.setRequestHeader('X-Date', header['X-Date']);
   
    get.send();
    get.onload = function(){
        var data = JSON.parse(get.responseText);
        for(var i = 0;data.length>i;i++){
            if(data[i].RouteName.Zh_tw==route){
                if(data[i].Direction==0){
                    for(var j = 0 ;data[i].Stops.length>j;j++){
                        go_step[j] = data[i].Stops[j].StopName.Zh_tw;
                    }
                }
                if(data[i].Direction==1){
                    for(var j = 0 ;data[i].Stops.length>j;j++){
                        back_step[j] = data[i].Stops[j].StopName.Zh_tw;
                     }
                }
            }
        }
        console.log(go_step);
        console.log(back_step);
        getAPI();
    }
}

function getAPI(){
    var get = new XMLHttpRequest();
    var header = GetAuthorizationHeader();
    get.open('get',"https://ptx.transportdata.tw/MOTC/v2/Bus/EstimatedTimeOfArrival/City/Kaohsiung?$format=JSON",true);
    get.setRequestHeader('Authorization', header['Authorization']);
    get.setRequestHeader('X-Date', header['X-Date']);
   
    get.send();
    get.onload = function(){
        //console.log(get.responseText);
        //var data = get.responseText;
        data = JSON.parse(get.responseText);
        //console.log(data);
        show_ans();
        r = 0;
        k = 0;
        setTimeout('getAPI()',30000);
    }
    
}

function get_goBustime(StopName){
    for(var i = 0;data.length>i;i++){
        //console.log(data[i].RouteName.Zh_tw);
            if(data[i].Direction==0&&data[i].RouteName.Zh_tw==route&&data[i].StopName.Zh_tw==StopName){
                    console.log(i);
                    var m = 0;
                    var time = data[i].EstimateTime;
                    var min = time/60;
                    var sec = time%60;
                    var NextBus = new Date(data[i].NextBusTime);
                    var NextBusHour = NextBus.getHours();
                    var NextBusMin = NextBus.getMinutes();
                   // console.log(NextBus);


                    console.log(data[i]);
                    console.log(data[i].StopName.Zh_tw);
                    console.log(StopName);
                    if(data[i].NextBusTime){
                         if(data[i].PlateNumb==-1||data[i].PlateNumb==""){
                            //console.log('2');
                        
                            go_time[r] = NextBusHour+':'+NextBusMin;
                            //result +='<h3>'+data[i].StopName.Zh_tw+'</h3><li>下一班公車預計到達時間：'+NextBusHour+':'+NextBusMin+'</li>';
                         }else{
                            //console.log('3');
                            go_time[r] = min+'分';
                            //result +='<h3>'+data[i].StopName.Zh_tw+'</h3><li>預估到站時間:'+min+'分'+sec+'秒，下一班公車預計到達時間：'+NextBusHour+':'+NextBusMin+'</li>';
                        }
                    }else{
                        go_time[r] = '末班車已過';
                        }
                   r++;
            
            
        }
    }
}

function get_backBustime(StopName){
        for(var i = 0;data.length>i;i++){
        //console.log(data[i].RouteName.Zh_tw);
            if(data[i].Direction==1&&data[i].RouteName.Zh_tw==route&&data[i].StopName.Zh_tw==StopName){
                    //console.log(i);
                    var m = 0;
                    var time = data[i].EstimateTime;
                    var min = time/60;
                    var sec = time%60;
                    var NextBus = new Date(data[i].NextBusTime);
                    var NextBusHour = NextBus.getHours();
                    var NextBusMin = NextBus.getMinutes();
                   // console.log(NextBus)
                    //console.log(data[i]);
                    //console.log(data[i].StopName.Zh_tw);
                    //console.log(StopName);
                    if(data[i].NextBusTime){
                        if(data[i].PlateNumb==-1||data[i].PlateNumb==""){
                            back_time[k] = NextBusHour+':'+NextBusMin;
                            //result +='<h3>'+data[i].StopName.Zh_tw+'</h3><li>下一班公車預計到達時間：'+NextBusHour+':'+NextBusMin+'</li>';
                        }else{
                            back_time[k] = min+'分';
                             //result +='<h3>'+data[i].StopName.Zh_tw+'</h3><li>預估到站時間:'+min+'分'+sec+'秒，下一班公車預計到達時間：'+NextBusHour+':'+NextBusMin+'</li>';
                        }
                    }
                    else{
                        back_time[k] = '末班車已過';
                    }
                k++;
          }          
     }
}

function show_ans(){
    var back_result="<h2>返程：</h2>";
    var count1 = 0;
    var count2 = 0;
    for(var j = 0;go_step.length>j;j++){
               count1++;
               //console.log(count1);
               //console.log('輸入：'+go_step[j]);
                //getBustime(go_step[j],0);
               get_goBustime(go_step[j]);
            }
    for(var j = 0;back_step.length>j;j++){
                count2++;
                //console.log(count2);
                //console.log('輸入：'+back_step[j]);
                get_backBustime(back_step[j]);
            }
    //console.log(go_time);
    //console.log(back_time);
    var go_result="<table class='route-stop'><tbody>";
    for(var j = 0;go_step.length>j;j++){
               go_result += "<tr><td><div class='time'>"+go_time[j]+"</div></td><td><div class='stopname'><h4>"+go_step[j]+"</h4></div></td></tr>";
               //go_result += '<h3>'+go_step[j]+'：</h3>'+go_time[j];
            }
    for(var j = 0;back_step.length>j;j++){
               back_result += "<tr><td><div class='time'>"+back_time[j]+"</div></td><td><div class='stopname'><h4>"+back_step[j]+"</h4></div></td></tr>";
               //back_result += '<h3>'+back_step[j]+'：</h3>'+back_time[j] ;
            }
    go_result += "</tbody></table>";
    back_result += "</tbody></table>"; 
    document.querySelector('.rd-main').innerHTML = go_result;
    //document.querySelector('.back_result').innerHTML = back_result;    
}

function answer(){
    var url = decodeURIComponent(location.href);
    var temp = url.split("?");
    var routeA = temp[1];    
    //var route1 = document.querySelector('.route').value;
    //var route2 = document.querySelector('.num').value;
    //route = route1+route2;
    route = routeA;
    console.log(route);
    document.querySelector('.url_back').innerHTML = "<a href='routedetail_back.html?"+route+"'><div class='rd-backroad'>返程</div></a>";
    document.querySelector('.name').textContent = route;
    getRouteStep();
    //console.log(go_step);
    //show_ans();
    //console.log(go_time);
    //console.log(back_time);
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