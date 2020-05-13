var max = 10
var myRoutes = [max]
function checkRoute( objectID) {
    var i = 0
    var myRoute = "" 
    while (i < max) {
        myRoute = "myRoute" + i
        if (localStorage.getItem(myRoute) !== null) {
            i++
        }
        else {
            storeRoute(myRoute,objectID)
            break
        }
    }
}
function storeRoute(myRoute, objectID) {
    
    //執行寫入
    localStorage.setItem(myRoute, data[objectID].RouteName.Zh_tw)
    console.log("寫入")
    console.log(localStorage.getItem(myRoute))
    
}

function delRoute( ID ) {
    RouteID = "myRoute" + ID 
    localStorage.removeItem(RouteID)
    var str =""
    var j = 0
    var myRoute = ""
    while (j < max) {
        myRoute = "myRoute" + j 
        if (localStorage.getItem(myRoute) !== null) {
            str += "<tr><td><a href='routedetail.html?"+localStorage.getItem(myRoute)+"'><div class='route-collect'><h3>"+localStorage.getItem(myRoute)+"</h3></div></a></td><td><button type='button' onclick='delRoute(" + j + ")' class='del'>刪除</button></td></tr>"
        }
        j++
    }
   
    document.getElementById("seeMyRoutes").innerHTML = str;
}
function seeMyRoutes(){
    var str =""
    var j = 0
    var myRoute = ""
    while (j < max) {
        myRoute = "myRoute" + j 
        if (localStorage.getItem(myRoute) !== null) {
            str += "<tr><td><a href='routedetail.html?"+localStorage.getItem(myRoute)+"'><div class='route-collect'><h3>"+localStorage.getItem(myRoute)+"</h3></div></a></td><td><button type='button'  onclick='delRoute(" + j + ")' class='del'>刪除</button></td></tr>"
        }
        j++
    }
   
    document.getElementById("seeMyRoutes").innerHTML = str;
}