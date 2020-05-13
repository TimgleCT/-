var max = 10
var myStops = [max]
function checkStop( objectID) {
    var i = 0
    var myStop = "" 
    while (i < max) {
        myStop = "myStop" + i
        if (localStorage.getItem(myStop) !== null) {
            i++
        }
        else {
            storeStop(myStop,objectID)
            break
        }
    }
}
function storeStop(myStop, objectID) {
    
    //執行寫入
    localStorage.setItem(myStop, data[objectID].StopName.Zh_tw)
    console.log("寫入")
    console.log(localStorage.getItem(myStop))
    
}

function delStop( ID ) {
    StopID = "myStop" + ID 
    localStorage.removeItem(StopID)
    var str =""
    var j = 0
    var myStop = ""
    while (j < max) {
        myStop = "myStop" + j 
        if (localStorage.getItem(myStop) !== null) {
            str += "<tr><td ><a<tr><td><a href='busstop.html?"+localStorage.getItem(myStop)+"'><div class='route-collect'><h3>"+localStorage.getItem(myStop)+"</h3></div></a></td><td><button type='button' onclick='delStop(" + j + ")' class='del'>刪除</button></td></tr>"
        }
        j++
    }
   
    document.getElementById("seeMyStops").innerHTML = str;
}
function seeMyStops(){
    var str =""
    var j = 0
    var myStop = ""
    while (j < max) {
        myStop = "myStop" + j 
        if (localStorage.getItem(myStop) !== null) {
            str += "<tr><td><a href='busstop.html?"+localStorage.getItem(myStop)+"'><div class='route-collect'><h3>"+localStorage.getItem(myStop)+"</h3></div></a></td><td><button type='button'  onclick='delStop(" + j + ")' class='del'>刪除</button></td></tr>"
        }
        j++
    }
   
    document.getElementById("seeMyStops").innerHTML = str;
}