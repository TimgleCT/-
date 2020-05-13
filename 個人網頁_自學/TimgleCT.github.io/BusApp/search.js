var sName = new Array();
var result = '搜尋結果：';
var data;

function search_stop(){
    var get = new XMLHttpRequest();
    var header = GetAuthorizationHeader();
    get.open('get',"https://ptx.transportdata.tw/MOTC/v2/Bus/Stop/City/Kaohsiung?$top=30&$format=JSON",true);
    get.setRequestHeader('Authorization', header['Authorization']);
    get.setRequestHeader('X-Date', header['X-Date']);
   
    get.send();
    get.onload = function(){
         data = JSON.parse(get.responseText);
        var kw = document.querySelector('.keyword').value;
        var result = '搜尋結果：';

        for(var i = 0;data.length>i;i++){
            if(data[i].StopName.Zh_tw.match(kw)!=null){
                //console.log(data[i].StopName.Zh_tw);
                sName[i] = data[i].StopName.Zh_tw;
                console.log(sName[i]);
                result += "<tr><td ><div class='arrived-name' ><h3><a href='busstop.html?"+sName[i]+"'>"+sName[i]+"</h3><div></td><td><button type='button' onclick='checkStop(" + i + ")' class='btnheart'><img src='image/unlike.png' alt='unlike'></button></td></tr>"
            }
        }



        console.log(sName.length);
        console.log(result);
        document.querySelector('.route-arrived').innerHTML = result;

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

    return { 'Authorization': Authorization, 'X-Date': GMTString /*,'Accept-Encoding': 'gzip'*/}; //????��?��??��????�阶?��?��????��?��?js????��?��???�????��?��?��?�絭????��?��???�??????��???????��?��?????��?��i? 'Accept-Encoding': 'gzip'???????��?��????��?��???????��?��??��?��?��?��?�絬?????��?��????��?��????��?��???��??????��?��????
}

