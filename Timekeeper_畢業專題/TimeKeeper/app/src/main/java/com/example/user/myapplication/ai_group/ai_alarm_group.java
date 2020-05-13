package com.example.user.myapplication.ai_group;

import android.app.Activity;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.Window;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ListView;
import android.widget.RelativeLayout;

import com.example.user.myapplication.Connect_To_Server;
import com.example.user.myapplication.R;
import com.example.user.myapplication.ai_alarm;
import com.example.user.myapplication.get_u_sticker;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

import static com.example.user.myapplication.mainpage.KEY;

public class ai_alarm_group extends Activity {
    private RecyclerView mRecyclerView;
    private ai_Adapter mMyAdapter;
    private ArrayList<Bitmap> mbitmapArray;
    private List<String> mList;
    Connect_To_Server find_friends;
    JSONArray get_fresult;
    List<String> friendlist;
    List<String> idlist;
    String ilist;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.ai_alarm_group);

        //RecyclerView三部曲+LayoutManager
        mRecyclerView = findViewById(R.id.group_list);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        mRecyclerView.setLayoutManager(linearLayoutManager);
        initData();
        mMyAdapter = new ai_Adapter(mList, mbitmapArray, friendlist, idlist);
        mRecyclerView.setAdapter(mMyAdapter);

        //为RecyclerView添加HeaderView和FooterView
        setFooterView(mRecyclerView);

        Intent intent1 = getIntent();
        if (intent1 != null){
            ilist = intent1.getStringExtra("idlist");
            if (ilist != null && !ilist.equals("")){
                String[] arrays = ilist.trim().split("\\s+");
                for(String s : arrays){
                    idlist.add(s);
                }
            }
        }

        Button btn_save = findViewById(R.id.btn_save);
        btn_save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String checklist = "";
                for(int i = 0; i < idlist.size(); i++){
                    checklist = checklist + idlist.get(i) + "  ";
                }
                Log.d("id",":"+checklist);
                Intent intent = new Intent(ai_alarm_group.this, ai_alarm.class);
                intent.putExtra("idlist", checklist);
                setResult(3, intent); //requestCode需跟A.class的一樣
                finish();
            }
        });

    }

    //      資料產生
    private void initData() {
        mList = new ArrayList<>();
        mbitmapArray = new ArrayList<>();
        friendlist = new ArrayList<>();
        idlist = new ArrayList<>();
        ArrayList friendID_list = new ArrayList();
        find_friends = new Connect_To_Server();

        final String u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
        Thread search_friend = new Thread(new Runnable() {
            @Override
            public void run() {
                find_friends.connect("select_sql", "SELECT user.u_name,user.user_id FROM `user` WHERE user.user_id = any(SELECT user_friends.friend_id FROM `user_friends` WHERE user_friends.user_id =  '" + u_id + "')");
            }
        });
        search_friend.start();
        try {
            search_friend.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        try {
            get_fresult = new JSONArray(find_friends.get_data);
            int lenght = get_fresult.length();
            for (int i = 0; i < lenght; i++) {
                JSONObject jsonObject = get_fresult.getJSONObject(i);
                mList.add(jsonObject.getString("u_name"));
                friendID_list.add(jsonObject.getString("user_id"));
                friendlist.add(jsonObject.getString("user_id"));
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < friendID_list.size(); i++) {
            final String f_id = friendID_list.get(i).toString();
            final Bitmap[] img = new Bitmap[1];
            Thread get_photo = new Thread(new Runnable() {
                @Override
                public void run() {
                    img[0] = get_u_sticker.get_sticker("http://140.127.218.207/uploads/" + f_id + ".png");
                }
            });
            get_photo.start();
            try {
                get_photo.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (img[0] != null) {
                mbitmapArray.add(img[0]);
            } else {
                Resources res = getResources();
                Bitmap logo = BitmapFactory.decodeResource(res, R.drawable.ai_open);
                mbitmapArray.add(logo);
            }
        }


    }

    private void setFooterView(RecyclerView view){
        View footer = LayoutInflater.from(this).inflate(R.layout.footer, view, false);
        mMyAdapter.setFooterView(footer);
    }

}
