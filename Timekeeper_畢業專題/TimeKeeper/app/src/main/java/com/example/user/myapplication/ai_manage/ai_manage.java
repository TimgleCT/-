package com.example.user.myapplication.ai_manage;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.support.v4.view.ViewPager;
import android.support.v7.app.AppCompatActivity;

import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.widget.ArrayAdapter;

import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.Spinner;
import android.widget.Switch;

import com.example.user.myapplication.R;
import com.example.user.myapplication.ai_alarm;

import java.util.ArrayList;
import java.util.List;
import com.example.user.myapplication.setting_setup;


public class ai_manage extends AppCompatActivity {
    private ViewPager mViewPager;
    private Switch switch1;
    private CardPagerAdapter mCardAdapter;
    private ShadowTransformer mCardShadowTransformer;
    String smartlist;
    List<String> slist;
    Spinner spinner_home, spinner_work;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.ai_alarm_manage);
        mViewPager = findViewById(R.id.mviewPager);


        //下拉選單
        spinner_home = findViewById(R.id.home_address);
        ArrayAdapter<CharSequence> careerList1 = ArrayAdapter.createFromResource(ai_manage.this, R.array.address,
                android.R.layout.simple_spinner_dropdown_item);
        spinner_home.setAdapter(careerList1);

        spinner_work = findViewById(R.id.work_address);
        ArrayAdapter<CharSequence> careerList2 = ArrayAdapter.createFromResource(ai_manage.this, R.array.address,
                android.R.layout.simple_spinner_dropdown_item);
        spinner_work.setAdapter(careerList2);


        mCardAdapter = new CardPagerAdapter();
        mCardAdapter.addCardItem(new CardItem("若您所工作的縣市\n" +
                "發布颱風天停班停課\n" +
                "系統將自動幫您關閉當日鬧鐘\n" +
                "讓您享有一個美好的早晨 : )\n", R.drawable.typhoon));

        mCardAdapter.addCardItem(new CardItem("若該日為國定假日為\n" +
                "(係指勞基法第37條之規定)\n" +
                "系統將自動幫您關閉當日鬧鐘\n" +
                "讓您享有一個美好的早晨 : )\n", R.drawable.holiday));

        Intent intent1 = getIntent();
        slist = new ArrayList<>();
        if (intent1 != null){
            smartlist = intent1.getStringExtra("sbar");
            Log.d("測試smartlist","安安"+smartlist);
            if (smartlist != null && !smartlist.equals("0  0")){
                String[] arrays = smartlist.trim().split("\\s+");
                for(String s : arrays){
                    slist.add(s);
                }
                Log.d("測試安安","@@"+slist);
                if(slist != null){
                    mCardShadowTransformer = new ShadowTransformer(mViewPager, mCardAdapter,slist.get(2),slist.get(3));
                    int s_home = careerList1.getPosition(slist.get(0));
                    spinner_home.setSelection(s_home);
                    int s_work = careerList2.getPosition(slist.get(1));
                    spinner_work.setSelection(s_work);
                }
            }else {
                mCardShadowTransformer = new ShadowTransformer(mViewPager, mCardAdapter,"0","0");
            }
        }
        mViewPager.setAdapter(mCardAdapter);
        mViewPager.setPageTransformer(false, mCardShadowTransformer);
        mViewPager.setOffscreenPageLimit(3);
        mCardShadowTransformer.enableScaling(true);


        Button save = findViewById(R.id.btn_save);
        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int[] swithb = mCardShadowTransformer.switchb;
                String home = spinner_home.getSelectedItem().toString();
                String work = spinner_work.getSelectedItem().toString();
                String sbar = " " + home + " " + work + " ";
                for (int i = 0; i < 2 ; i++){
                    sbar = sbar + swithb[i] + "  ";
                }

                Intent intent = new Intent(ai_manage.this, ai_alarm.class);
                intent.putExtra("switchb", sbar);
                setResult(4, intent);
                Log.d("ss",":"+sbar);
                finish();
            }
        });


    }
}
