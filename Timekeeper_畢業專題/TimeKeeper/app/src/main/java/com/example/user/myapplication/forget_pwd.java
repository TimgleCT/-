package com.example.user.myapplication;

import android.content.Context;
import android.content.Intent;
import android.os.IBinder;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Properties;

import javax.mail.Authenticator;
import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;

public class forget_pwd extends AppCompatActivity {
    TextView signup, login;
    EditText get_account,get_name;
    Button send;
    Connect_To_Server db_select;
    JSONArray get_result;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.forget_pwd);

        get_account = (EditText)findViewById(R.id.user_mail);
        get_name = (EditText)findViewById(R.id.user_name);
        send = (Button)findViewById(R.id.login_btn);
        db_select = new Connect_To_Server();

        send.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final String account = get_account.getText().toString();
                String name = get_name.getText().toString();
                Thread check_account = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        db_select.connect("select_sql","SELECT user_id,u_password,u_name FROM `user` WHERE user_id = '"+ account +"'");
                    }
                });
                check_account.start();
                try {
                    Thread.sleep(400);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                String u_id = null,u_pwd = null,u_name =null;
                try{
                    get_result = new JSONArray(db_select.get_data);
                    int lenght = get_result.length();
                    for(int i = 0;i < lenght;i++){
                        JSONObject jsonObject = get_result.getJSONObject(i);
                        u_id = jsonObject.getString("user_id");
                        u_pwd = jsonObject.getString("u_password");
                        u_name = jsonObject.getString("u_name");
                    }
                }
                catch(JSONException e) {
                    e.printStackTrace();
                }


                if(u_id == null){
                    new AlertDialog.Builder(forget_pwd.this).setTitle("在試試看一次喔~").setMessage("信箱或姓名錯誤!!")
                            .setNegativeButton("OK",null)
                            .show();
                }else {
                    String db_u_id = u_id;
                    String db_u_pwd = u_pwd;
                    String db_u_name = u_name;
                    if(db_u_name.equals(name)){
                        main(db_u_id,db_u_name,db_u_pwd);
                    }else{
                        new AlertDialog.Builder(forget_pwd.this).setTitle("在試試看一次喔~").setMessage("信箱或姓名錯誤!!")
                                .setNegativeButton("OK",null)
                                .show();
                    }
                }
            }
        });


        signup = findViewById(R.id.signup_text);
        signup.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(forget_pwd.this, sign_up.class);
                startActivity(intent);
            }
        });

        login = findViewById(R.id.login_text);
        login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent1 = new Intent(forget_pwd.this, login.class);
                startActivity(intent1);
            }
        });
    }

    public void main(final String address, final String name, final String pwd) {
        Thread send_mail = new Thread(new Runnable() {
            @Override
            public void run() {
                String host = "smtp.gmail.com";
                int port = 587;
                final String username = "timekeepernukim@gmail.com";
                final String password = "g6ru0 ej03ru8";//your password

                Properties props = new Properties();
                props.put("mail.smtp.host", host);
                props.put("mail.smtp.auth", "true");
                props.put("mail.smtp.starttls.enable", "true");
                props.put("mail.smtp.port", port);
                Session session = Session.getInstance(props, new Authenticator() {
                    protected PasswordAuthentication getPasswordAuthentication() {
                        return new PasswordAuthentication(username, password);
                    }
                });
                try {
                    Message message = new MimeMessage(session);
                    message.setFrom(new InternetAddress("timekeepernukim@gmail.com"));
                    message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(address));
                    message.setSubject("TimeKeeper 會員密碼");
                    message.setText("Dear "+name+", \n\n 您好，您的密碼為："+pwd);

                    Transport transport = session.getTransport("smtp");
                    transport.connect(host, port, username, password);

                    Transport.send(message);
                    //Log.d("寄信","結束");
                    System.out.println("寄送email結束.");
                } catch (MessagingException e) {
                    //e.printStackTrace();
                    throw new RuntimeException(e);
                }
            }
        });
        send_mail.start();
        new AlertDialog.Builder(forget_pwd.this).setTitle("去信箱看看吧~").setMessage("確認信已送出~!!")
                .setNegativeButton("OK",null)
                .show();
    }

    //點擊空白處隱藏鍵盤
    public boolean dispatchTouchEvent(MotionEvent ev) {
        if (ev.getAction() == MotionEvent.ACTION_DOWN) {

            // 获得当前得到焦点的View，一般情况下就是EditText（特殊情况就是轨迹求或者实体案件会移动焦点）
            View v = getCurrentFocus();

            if (isShouldHideInput(v, ev)) {
                hideSoftInput(v.getWindowToken());
            }
        }
        return super.dispatchTouchEvent(ev);
    }

    private boolean isShouldHideInput(View v, MotionEvent event) {
        if (v != null && (v instanceof EditText)) {
            int[] l = {0, 0};
            v.getLocationInWindow(l);
            int left = l[0], top = l[1], bottom = top + v.getHeight(), right = left
                    + v.getWidth();
            if (event.getX() > left && event.getX() < right
                    && event.getY() > top && event.getY() < bottom) {
                // 点击EditText的事件，忽略它。
                return false;
            } else {
                return true;
            }
        }
        // 如果焦点不是EditText则忽略，这个发生在视图刚绘制完，第一个焦点不在EditView上，和用户用轨迹球选择其他的焦点
        return false;
    }

    private void hideSoftInput(IBinder token) {
        if (token != null) {
            InputMethodManager im = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
            im.hideSoftInputFromWindow(token, InputMethodManager.HIDE_NOT_ALWAYS);
        }
    }

    // 按返回鍵取消delete狀態
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            finish();
        }
        return false;
    }
}
