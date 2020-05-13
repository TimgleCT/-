package com.example.user.myapplication;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.ScrollView;

public class monster extends Activity {

    ImageButton fab;
    Button close;
    ScrollView list;
    LinearLayout linearLayout;
    RelativeLayout kind_list;
    Button qus;
    ImageView blood;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.monster);

        blood = findViewById(R.id.blood);
        Animation scaleAnimation = AnimationUtils.loadAnimation(this, R.anim.sacle_left_to_right);
        blood.startAnimation(scaleAnimation);

        fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(monster.this, mainpage.class);
                startActivity(intent);
                finish();
            }
        });


        linearLayout = findViewById(R.id.linearLayout);
        list = findViewById(R.id.list);
        kind_list = findViewById(R.id.kind_list);
        qus = findViewById(R.id.qus);
        qus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                kind_list.setVisibility(View.VISIBLE);
                linearLayout.setFocusableInTouchMode(true);
                linearLayout.setFocusable(true);
            }
        });


        close = findViewById(R.id.close);
        close.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                kind_list.setVisibility(View.GONE);
                linearLayout.setFocusableInTouchMode(false);
                linearLayout.setFocusable(false);
            }
        });
    }

}
