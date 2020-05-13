package com.example.user.myapplication;


import android.app.DatePickerDialog;
import android.content.ContentResolver;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.InputType;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.FileNotFoundException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import static com.example.user.myapplication.hideinput.hideSoftInput;
import static com.example.user.myapplication.hideinput.isShouldHideInput;
import static com.example.user.myapplication.mainpage.KEY;
import static com.example.user.myapplication.mainpage.logon;

public class setting_setup extends AppCompatActivity {

    View timekeeper_logo;

    // hamburger
    Button menu, sign_out, edit_btn, save_btn, date_btn,clear;
    PopupWindow popupWindow;
    TextView name, mail, pwd, gender, birth, job;


    EditText name_edit, birthday_edit, pwd_edit;

    LinearLayout show_layout, edit_layout, show_btn_layout, edit_btn_layout;
    RadioGroup gender_edit;
    RadioButton male,female;
    View sticker;
    Spinner spinner;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.setting_setup);

        name = findViewById(R.id.name);
        name_edit = findViewById(R.id.name_edit);
        mail = findViewById(R.id.mail);
        pwd = findViewById(R.id.pwd);
        pwd_edit = findViewById(R.id.pwd_edit);
        gender = findViewById(R.id.gender);
        gender_edit = findViewById(R.id.group);
        male = findViewById(R.id.radioButton);
        female = findViewById(R.id.radioButton2);
        birthday_edit = findViewById(R.id.birthday_edit);
        birth = findViewById(R.id.birthday);
        job = findViewById(R.id.career);
        edit_btn = findViewById(R.id.btn_edit);
        save_btn = findViewById(R.id.btn_save);
        name_edit = findViewById(R.id.name_edit);
        birthday_edit = findViewById(R.id.birthday_edit);
        date_btn = findViewById(R.id.date_btn);
        show_layout = findViewById(R.id.show_layout);
        edit_layout = findViewById(R.id.edit_layout);
        show_btn_layout = findViewById(R.id.show_btn_layout);
        edit_btn_layout = findViewById(R.id.edit_btn_latout);
        clear = findViewById(R.id.btn_clear);

        final Connect_To_Server edit_data  = new Connect_To_Server();

        //下拉選單
        spinner = findViewById(R.id.career_edit);
        ArrayAdapter<CharSequence> careerList = ArrayAdapter.createFromResource(setting_setup.this, R.array.career,
                android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(careerList);

        //生日的輸入框動作
        birthday_edit.setInputType(InputType.TYPE_NULL); //不显示系统输入键盘
        birthday_edit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showDatePickerDialog();
            }
        });

        //避免點擊2下才打開
        birthday_edit.setOnFocusChangeListener(new View.OnFocusChangeListener() {
            @Override
            public void onFocusChange(View v, boolean hasFocus) {
                if (hasFocus) showDatePickerDialog();
            }
        });

        date_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showDatePickerDialog();
            }
        });
        //編輯性別
        final String[] new_gender = {null};
        gender_edit.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                switch(checkedId){
                    case R.id.radioButton:
                        new_gender[0] = "MALE";
                        break;
                    case R.id.radioButton2:
                        new_gender[0] = "FEMALE";
                        break;
                }
            }
        });

        //切換edit或show模式
        edit_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                show_layout.setVisibility(View.GONE);
                edit_layout.setVisibility(View.VISIBLE);
                show_btn_layout.setVisibility(View.GONE);
                edit_btn_layout.setVisibility(View.VISIBLE);
                name.setVisibility(View.GONE);
                name_edit.setVisibility(View.VISIBLE);
            }
        });

        save_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //儲存編輯
                Thread edit = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        edit_data.connect("insert_sql","UPDATE `user` SET `u_name` = '"+name_edit.getText().toString()+"', `u_password` = '"+pwd_edit.getText().toString()+"', `u_gender` = '"+new_gender[0]+"', `u_birth` = '"+birthday_edit.getText().toString()+"' , `u_job` = '"+spinner.getSelectedItem().toString()+"'WHERE `user`.`user_id` = '"+getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null)+"';");
                    }
                });
                edit.start();
                try {
                    edit.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                refresh_user_data();
                show_layout.setVisibility(View.VISIBLE);
                edit_layout.setVisibility(View.GONE);
                show_btn_layout.setVisibility(View.VISIBLE);
                edit_btn_layout.setVisibility(View.GONE);
                name.setVisibility(View.VISIBLE);
                name_edit.setVisibility(View.GONE);
            }
        });
        clear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                name_edit.setText("");
                pwd_edit.setText("");
                birthday_edit.setText("");
                gender_edit.check(R.id.radioButton);
                spinner.setSelection(0);
            }
        });

        final String u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
        final Bitmap[] img = new Bitmap[1];
        sticker = findViewById(R.id.headshot);
        Thread get_photo = new Thread(new Runnable() {
            @Override
            public void run() {
                img[0] = get_u_sticker.get_sticker("http://140.127.218.207/uploads/" + u_id + ".png");
            }
        });
        get_photo.start();
        try {
            get_photo.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        if (img[0] != null) {
            BitmapDrawable drawable_sticker = new BitmapDrawable(img[0]);
            sticker.setBackground(drawable_sticker);
        }
        sticker.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new AlertDialog.Builder(setting_setup.this)
                        .setTitle("更新大頭貼")
                        .setMessage("選擇更新或是刪除大頭貼")
                        .setPositiveButton("編輯",
                                new DialogInterface.OnClickListener() {
                                    @Override
                                    public void onClick(DialogInterface dialog, int which) {
                                        Intent intent = new Intent();
                                        intent.setType("image/*");
                                        intent.setAction(Intent.ACTION_PICK);
                                        startActivityForResult(intent,0);
                                    }
                                })
                        .setNegativeButton("清除",
                                new DialogInterface.OnClickListener() {

                                    @Override
                                    public void onClick(DialogInterface dialog,
                                                        int which) {
                                        Thread clean_img = new Thread(new Runnable() {
                                            @Override
                                            public void run() {
                                                delete_img clean = new delete_img();
                                                clean.deleteFile(u_id);
                                            }
                                        });
                                        clean_img.start();
                                        try {
                                            clean_img.join();
                                        } catch (InterruptedException e) {
                                            e.printStackTrace();
                                        }
                                        sticker.setBackgroundResource(R.drawable.logo_small);
                                    }
                                }).show();

            }
        });

        refresh_user_data();


        //登出
        sign_out = findViewById(R.id.btn_singout);
        sign_out.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                SharedPreferences pref = v.getContext().getSharedPreferences(KEY, MODE_PRIVATE);
                pref.edit().clear().commit();
                logon = false;
                Intent intent = new Intent(v.getContext(), mainpage.class);
                v.getContext().startActivity(intent);
            }
        });

        //回首頁
        timekeeper_logo = findViewById(R.id.timekeeper_logo);
        timekeeper_logo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent2 = new Intent(setting_setup.this, mainpage.class);
                startActivity(intent2);
            }
        });

        // 選單彈跳
        menu = findViewById(R.id.menu);
        menu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (popupWindow == null) {
                    showPopupWindow show = new showPopupWindow(setting_setup.this, getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null));
                    try {
                        show.showPopupWindow(menu);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    //showPopupWindow();
                } else if (popupWindow.isShowing()) {
                    popupWindow.dismiss();
                } else {
                    popupWindow.showAsDropDown(menu, -200, -155);
                }
            }
        });

    }

    public String getPath(Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = managedQuery(uri, projection, null, null, null);
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        return cursor.getString(column_index);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        View view = LayoutInflater.from(this).inflate(R.layout.menu_window, null);//获取popupWindow子布局对象
        //當使用者按下確定後
        if (resultCode == RESULT_OK) {
            //取得圖檔的路徑位置
            final Uri uri = data.getData();
            //寫log
            Log.e("uri", uri.toString());
            //抽象資料的接口
            ContentResolver cr = this.getContentResolver();
            try {
                final String user_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
                //由抽象資料接口轉換圖檔路徑為Bitmap
                Bitmap bitmap = BitmapFactory.decodeStream(cr.openInputStream(uri));
                Log.e("uri", bitmap.toString());
                Thread upload_thread = new Thread(new Runnable() {
                    public void run() {
                        upload_img upload_sticker = new upload_img();
                        upload_sticker.uploadFile(user_id, getPath(uri));
                    }
                });
                upload_thread.start();
                upload_thread.join();
            } catch (FileNotFoundException e) {
                Log.e("Exception", e.getMessage(), e);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        refresh();
        super.onActivityResult(requestCode, resultCode, data);
    }
    public void refresh_user_data(){
        final String u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
        JSONArray get_result;
        final Connect_To_Server u_data = new Connect_To_Server();
        Thread search_account = new Thread(new Runnable() {
            @Override
            public void run() {
                u_data.connect("select_sql", "SELECT * FROM `user` WHERE user_id = '" + u_id + "'");
            }
        });
        search_account.start();
        try {
            search_account.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        String u_pwd = null, u_name = null, u_gender = null, u_birth = null, u_job = null;
        try {
            get_result = new JSONArray(u_data.get_data);
            int lenght = get_result.length();
            for (int i = 0; i < lenght; i++) {
                JSONObject jsonObject = get_result.getJSONObject(i);
                u_pwd = jsonObject.getString("u_password");
                u_name = jsonObject.getString("u_name");
                u_gender = jsonObject.getString("u_gender");
                u_birth = jsonObject.getString("u_birth");
                u_job = jsonObject.getString("u_job");
                int pwd_lenght = u_pwd.length();
                String not_pwd = "";
                for (int j = 0; j < pwd_lenght; j++) {
                    not_pwd = not_pwd + "*";
                }
                name.setText(u_name);
                mail.setText(u_id);
                pwd.setText(not_pwd);
                gender.setText(u_gender);
                birth.setText(u_birth);
                job.setText(u_job);
                name_edit.setText(u_name, TextView.BufferType.EDITABLE);
                pwd_edit.setText(u_pwd,TextView.BufferType.EDITABLE);
                birthday_edit.setText(u_birth,TextView.BufferType.EDITABLE);
                if(u_gender.equals("MALE")){
                    gender_edit.check(R.id.radioButton);
                }else{
                    gender_edit.check(R.id.radioButton2);
                }
                switch (u_job){
                    case "Worker":
                        spinner.setSelection(0);
                        break;
                    case "Student":
                        spinner.setSelection(1);
                        break;
                    case "Homemaker":
                        spinner.setSelection(2);
                        break;
                    case "Retirees":
                        spinner.setSelection(3);
                        break;
                    case "Others":
                        spinner.setSelection(4);
                        break;
                }


            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }
    public void refresh() {
        final String u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
        final Bitmap[] img = new Bitmap[1];
        sticker = findViewById(R.id.headshot);
        Thread get_photo = new Thread(new Runnable() {
            @Override
            public void run() {
                img[0] = get_u_sticker.get_sticker("http://140.127.218.207/uploads/" + u_id + ".png");
            }
        });
        get_photo.start();
        try {
            get_photo.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        if (img[0] != null) {
            BitmapDrawable drawable_sticker = new BitmapDrawable(img[0]);
            sticker.setBackground(drawable_sticker);
        }
    }

    @Override
    protected void onResume() {

        super.onResume();

    }

    //跳出DatePicker選擇生日日期
    private void showDatePickerDialog() {
        Calendar c = Calendar.getInstance();
        new DatePickerDialog(setting_setup.this, new DatePickerDialog.OnDateSetListener() {

            @Override
            public void onDateSet(DatePicker view, int year, int monthOfYear, int dayOfMonth) {
                // TODO Auto-generated method stub
                birthday_edit.setText(" " + year + " / " + (monthOfYear + 1) + " / " + dayOfMonth);
            }
        }, c.get(Calendar.YEAR), c.get(Calendar.MONTH), c.get(Calendar.DAY_OF_MONTH)).show();

    }

    //點擊空白處隱藏鍵盤
    public boolean dispatchTouchEvent(MotionEvent ev) {
        if (ev.getAction() == MotionEvent.ACTION_DOWN) {

            // 获得当前得到焦点的View，一般情况下就是EditText（特殊情况就是轨迹求或者实体案件会移动焦点）
            View v = getCurrentFocus();

            if (isShouldHideInput(v, ev)) {
                hideSoftInput(v.getWindowToken(),this);
            }
        }
        return super.dispatchTouchEvent(ev);
    }

}


