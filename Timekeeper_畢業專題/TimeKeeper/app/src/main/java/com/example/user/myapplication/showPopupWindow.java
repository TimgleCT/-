package com.example.user.myapplication;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.PopupWindow;
import android.widget.TextView;

public class showPopupWindow extends Activity{
    Context context;
    ImageButton menu_open;
    ImageView photo_sticker;
    PopupWindow popupWindow;
    FrameLayout menu_window;
    TextView set_up, friend, about, monster;
    Bitmap img;
    String user_id;

    public showPopupWindow(Context context,String user_id){
        this.context = context;
        this.user_id = user_id;
    }

    public void showPopupWindow(Button menu) throws InterruptedException {
        View view = LayoutInflater.from(context).inflate(R.layout.menu_window,null);//获取popupWindow子布局对象
        popupWindow = new PopupWindow(view, ViewGroup.LayoutParams.WRAP_CONTENT,ViewGroup.LayoutParams.WRAP_CONTENT,true);//初始化
        popupWindow.showAsDropDown(menu,-300,-155);//在ImageView控件下方弹出
        popupWindow.setOutsideTouchable(true);
        popupWindow.update();

        photo_sticker = view.findViewById(R.id.photo_sticker);

        Thread get_photo = new Thread(new Runnable() {
            @Override
            public void run() {
                get_u_sticker get_img = new get_u_sticker();
                img = get_img.get_sticker("http://140.127.218.207/uploads/"+user_id+".png");
            }
        });
        get_photo.start();
        try {
            get_photo.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        if(img != null){
            photo_sticker.setImageBitmap(img);
        }
        menu_open = view.findViewById(R.id.menu_btn_open);
        menu_open.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                popupWindow.dismiss();
            }
        });

        set_up = view.findViewById(R.id.set_up);
        friend = view.findViewById(R.id.friend);
        about = view.findViewById(R.id.about);
        monster = view.findViewById(R.id.monster);

        set_up.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent2 = new Intent(context, setting_setup.class);
                context.startActivity(intent2);
                popupWindow.dismiss();
            }
        });

        friend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent3 = new Intent(context, setting_friend.class);
                context.startActivity(intent3);
                popupWindow.dismiss();
            }
        });

        about.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent3 = new Intent(context, about.class);
                context.startActivity(intent3);
                popupWindow.dismiss();
            }
        });

        monster.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent3 = new Intent(context, monster.class);
                context.startActivity(intent3);
                popupWindow.dismiss();
            }
        });

    }
}
