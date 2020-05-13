var route = ""; 
function get_red(){
    var name = document.getElementById('btnred').value;
    route = name;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_org(){
    var name = document.getElementById('btnora').value;
    route = name;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_yellow(){
    var name = document.getElementById('btnye').value;
    route = name;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_main(){
    var name = document.getElementById('btng').value;
    route = name;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_fast(){
    var name = document.getElementById('btnk').value;
    route = name;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_joy(){
    var name = document.getElementById('btnjoy').value;
    route = name;
    document.getElementById('route').textContent = route;
    step_match();
}

function get_num1(){
    var num = document.getElementById('btn1').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num2(){
    var num = document.getElementById('btn2').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num3(){
    var num = document.getElementById('btn3').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num4(){
    var num = document.getElementById('btn4').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num5(){
    var num = document.getElementById('btn5').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num6(){
    var num = document.getElementById('btn6').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num7(){
    var num = document.getElementById('btn7').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num8(){
    var num = document.getElementById('btn8').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num9(){
    var num = document.getElementById('btn9').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function get_num0(){
    var num = document.getElementById('btn0').value;
    route = route+num;
    document.getElementById('route').textContent = route;
    step_match();
}
function clean(){
    route = "";
    document.getElementById('route').textContent = route;
    get_routlistAPI();
}
