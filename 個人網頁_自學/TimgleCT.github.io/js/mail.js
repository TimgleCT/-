function validation(){
    var name = $("input[name = 'Name' ]").val();
    var email = $("input[name = 'Email' ]").val();
    var subject = $("input[name = 'Subject' ]").val();
    var content = $("textarea").val();
    var emailRegExp = /^\w+((-\w+)|(\.\w+))*\@[A-Za-z0-9]+((\.|-)[A-Za-z0-9]+)*\.[A-Za-z]+$/;
    var forgotErrorMessage = [];
    var errorMessage = "幫我檢查一下是不是";
    var inputError = "";
    var validate = true;
    var format = true;

    if(name == ""){
        forgotErrorMessage.push("姓名");
        validate = false;
    }
    if(email == ""){
        forgotErrorMessage.push("Email");
        validate = false;
    }else if(email.search(emailRegExp) == -1){
        inputError = "信箱格式錯誤囉!";
        format = false;
    }
    if(subject == ""){
        forgotErrorMessage.push("主旨");
        validate = false;
    }
    if(content == ""){
        forgotErrorMessage.push("信件內容");
        validate = false;
    }

    if(validate == true && format == true){
        $('#errorMessage').css('display','none');
        $('#seccessMessage').css('display','block');
        mailSubmitHandler();
        return true;
    }else{
        if(validate == false){
            for(var i = 0; i < forgotErrorMessage.length; i++){
                if(i == 0){
                    errorMessage = errorMessage + forgotErrorMessage[i];
                }else{
                    errorMessage = errorMessage +'、'+ forgotErrorMessage[i];
                }
            }
            errorMessage = errorMessage + '還沒有填喔!';
        }
        if(format == false){
            if(validate == false){
                errorMessage = errorMessage + '還有';
            }
            errorMessage = errorMessage + inputError;
        }
        
        $('#errorMessage').css('display','block');
        $('#errorMessage').children("p").html(errorMessage);
        return false;
    }
}

function mailSubmitHandler(){
    var to = "ct25283115@gmail.com";
    var name = $("input[name = 'Name' ]").val();
    var email = $("input[name = 'Email' ]").val();
    var subject = $("input[name = 'Subject' ]").val();
    var content = $("textarea").val();

    var body = content + "%0A%0A%0A" + "From：" + name;
    // var body = content;

    var thisMail = $('#mailTo');
    thisMail.attr("href","mailto:"+ to +"?subject=" + subject + "&body=" +body);

    if(document.all){
        thisMail.click();
        console.log("e04");
    }else{
        var evt = document.createEvent("MouseEvents");
        evt.initEvent("click", true, true);
        document.getElementById('mailTo').dispatchEvent(evt);
        console.log("安安");  
    }
}