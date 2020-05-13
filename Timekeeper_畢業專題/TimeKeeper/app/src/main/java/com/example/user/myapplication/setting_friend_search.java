package com.example.user.myapplication;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.List;

import static com.example.user.myapplication.hideinput.hideSoftInput;
import static com.example.user.myapplication.hideinput.isShouldHideInput;
import static com.example.user.myapplication.mainpage.KEY;

public class setting_friend_search extends AppCompatActivity {

    ImageButton search_btn;
    LinearLayout friend_show,friend_req;
    View timekeeper_logo;
    EditText search_friend;
    TextView friend_text;
    Connect_To_Server find_friend,check_friend;
    ImageView friend_photo;

    // list
    RecyclerView req_list;
    MyAdapter mAdapter;


    // hamburger
    Button menu, friend_add,cancel;
    ImageButton menu_open;
    PopupWindow popupWindow;
    FrameLayout menu_window;

    TextView set_up, friend, check,friend_name;
    JSONArray get_result,get_fresult,get_iresult;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.setting_friend_search);
        friend_show = findViewById(R.id.friend_show);
        friend_req = findViewById(R.id.friend_request);
        friend_text = findViewById(R.id.friend_text);
        friend_add = findViewById(R.id.friend_add_btn);
        cancel = findViewById(R.id.cancel);
        search_friend = (EditText) findViewById(R.id.search_friend);
        search_btn = findViewById(R.id.search_btn);
        friend_name = findViewById(R.id.friend_name);
        friend_photo = findViewById(R.id.friend_photo);
        find_friend = new Connect_To_Server();
        final String u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
        search_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Thread search_account = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        find_friend.connect("select_sql", "SELECT user_id,u_password,u_name FROM `user` WHERE user_id = '" + search_friend.getText() + "'");
                    }
                });
                search_account.start();
                try {
                    search_account.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                String f_id = null,f_pwd = null,f_name =null;
                try{
                    get_result = new JSONArray(find_friend.get_data);
                    int lenght = get_result.length();
                    for(int i = 0;i < lenght;i++){
                        JSONObject jsonObject = get_result.getJSONObject(i);
                        f_id = jsonObject.getString("user_id");
                        f_name = jsonObject.getString("u_name");
                    }
                }
                catch(JSONException e) {
                    e.printStackTrace();
                }

                if(f_id == null){

                    new AlertDialog.Builder(setting_friend_search.this).setTitle("在試試看一次~").setMessage("沒有這位使用者喔~")
                            .setNegativeButton("OK", null)
                            .show();
                }else {
                    friend_name.setText(f_name);
                    if(friend_show.getVisibility() != View.VISIBLE){
                        friend_show.setVisibility(View.VISIBLE);
                        friend_req.setVisibility(View.GONE);
                        friend_text.setText(R.string.friend_ser);
                    }
                    final String finalF_id = f_id;
                    final Bitmap[] img = new Bitmap[1];
                    Thread get_photo = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            img[0] = get_u_sticker.get_sticker("http://140.127.218.207/uploads/"+finalF_id+".png");
                        }
                    });
                    get_photo.start();
                    try {
                        get_photo.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    if(img[0] != null){
                        friend_photo.setImageBitmap(img[0]);
                    }
                    friend_add.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            String friend_id = finalF_id;
                            //查詢是否已送過交友邀請
                            Thread search_friend_invitation = new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    find_friend.connect("select_sql","SELECT friend_id FROM `user_friends_invitation` WHERE user_id = '"+u_id+"'");
                                }
                            });
                            search_friend_invitation.start();
                            try {
                                search_friend_invitation.join();
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                            int exist = 0;
                            String flist_u_id = null;
                            try{
                                get_fresult = new JSONArray(find_friend.get_data);
                                int lenght = get_fresult.length();
                                for(int i = 0;i < lenght;i++){
                                    JSONObject jsonObject = get_fresult.getJSONObject(i);
                                    flist_u_id = jsonObject.getString("friend_id");
                                    if(flist_u_id.equals(friend_id)){
                                        new AlertDialog.Builder(setting_friend_search.this).setTitle("等待對方回覆邀請中").setMessage("已送過交友邀請")
                                                .setNegativeButton("OK", new DialogInterface.OnClickListener() {
                                                    @Override
                                                    public void onClick(DialogInterface dialog, int which) {
                                                        friend_show.setVisibility(View.GONE);
                                                        friend_req.setVisibility(View.VISIBLE);
                                                        friend_text.setText(R.string.friend_req);
                                                    }
                                                })
                                                .show();
                                        exist = 1;
                                        break;
                                    }
                                }
                            }
                            catch(JSONException e) {
                                e.printStackTrace();
                            }

                            if(exist == 0){
                                int status = 0;
                                find_friend.connect("insert_sql","INSERT INTO `user_friends_invitation` (`user_id`, `friend_id`, `status`) VALUES('" + u_id + "', '" + friend_id + "', '" + status + "')");
                                new AlertDialog.Builder(setting_friend_search.this).setTitle("好友邀請已送出").setMessage("對方接受邀請後你們就是朋友囉")
                                        .setNegativeButton("OK", new DialogInterface.OnClickListener() {
                                            @Override
                                            public void onClick(DialogInterface dialog, int which) {
                                                friend_show.setVisibility(View.GONE);
                                                friend_req.setVisibility(View.VISIBLE);
                                                friend_text.setText(R.string.friend_req);
                                            }
                                        })
                                        .show();
                            }
                        }
                    });
                    cancel.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            friend_show.setVisibility(View.GONE);
                            friend_req.setVisibility(View.VISIBLE);
                            friend_text.setText(R.string.friend_req);
                        }
                    });
                }
            }
        });

        //回首頁
        timekeeper_logo = findViewById(R.id.timekeeper_logo);
        timekeeper_logo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent2 = new Intent(setting_friend_search.this, mainpage.class);
                startActivity(intent2);
            }
        });


        // 選單彈跳
        menu = findViewById(R.id.menu);
        menu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (popupWindow == null) {
                    showPopupWindow show = new showPopupWindow(setting_friend_search.this,getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null));
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

        Thread search_friend_invitations = new Thread(new Runnable() {
            @Override
            public void run() {
                find_friend.connect("select_sql", "SELECT user.u_name,user.user_id FROM `user` WHERE user.user_id = any(SELECT user_friends_invitation.user_id FROM `user_friends_invitation` WHERE user_friends_invitation.friend_id =  '" + u_id + "')");
            }
        });
        search_friend_invitations.start();
        try {
            search_friend_invitations.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        ArrayList<String> invitation = new ArrayList<>();
        ArrayList<String> invit_id = new ArrayList<>();
        try{
            get_iresult = new JSONArray(find_friend.get_data);
            int lenght = get_iresult.length();
            for(int i = 0;i < lenght;i++){
                JSONObject jsonObject = get_iresult.getJSONObject(i);
                invitation.add(jsonObject.getString("u_name"));
                invit_id.add(jsonObject.getString("user_id"));
            }
        }
        catch(JSONException e) {
            e.printStackTrace();
        }


        // recyclerview
        req_list = findViewById(R.id.req_list);
        mAdapter = new MyAdapter(invitation,invit_id);
        final LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        layoutManager.setOrientation(LinearLayoutManager.VERTICAL);
        req_list.setLayoutManager(layoutManager);
        req_list.setAdapter(mAdapter);

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


    // RecyclerView
    public class MyAdapter extends RecyclerView.Adapter<MyAdapter.ViewHolder> {
        private List<String> mData1;
        private List<String> mData2;

        public class ViewHolder extends RecyclerView.ViewHolder {
            public TextView mTextView;
            public ImageView photo;
            public ImageButton Y_response,N_response;/////
            public ViewHolder(View v) {
                super(v);
                mTextView = v.findViewById(R.id.friend_name);
                Y_response = v.findViewById(R.id.request_yes);////
                N_response = v.findViewById(R.id.request_no);////
                photo = v.findViewById(R.id.friend_photo);
            }
        }

        public MyAdapter(List<String> data1,List<String> data2) {
            mData1 = data1;
            mData2 = data2;
        }

        @Override
        public MyAdapter.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.friend_request_item, parent, false);
            ViewHolder vh = new ViewHolder(v);
            return vh;
        }

        @Override
        public void onBindViewHolder(final ViewHolder holder, final int position) {
            final String data = mData2.get(position);
            final Bitmap[] img = new Bitmap[1];
            Thread get_photo = new Thread(new Runnable() {
                @Override
                public void run() {
                    img[0] = get_u_sticker.get_sticker("http://140.127.218.207/uploads/"+data+".png");
                }
            });
            get_photo.start();
            try {
                get_photo.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if(img[0] !=null){
                holder.photo.setImageBitmap(img[0]);
            }
            holder.mTextView.setText(mData1.get(position));
            holder.Y_response.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    final String Y_u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
                    check_friend = new Connect_To_Server();
                    Thread res_friend_invitations = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            check_friend.connect("insert_sql", "INSERT INTO `user_friends` (`user_id`, `friend_id`) VALUES ('"+Y_u_id+"', '"+data+"')");
                            check_friend.connect("insert_sql", "INSERT INTO `user_friends` (`user_id`, `friend_id`) VALUES ('"+data+"', '"+Y_u_id+"')");
                            check_friend.connect("insert_sql","DELETE FROM `user_friends_invitation` WHERE `user_friends_invitation`.`user_id` = '"+data+"' AND `user_friends_invitation`.`friend_id` = '"+Y_u_id+"'");
                        }
                    });
                    res_friend_invitations.start();
                    Toast.makeText(setting_friend_search.this, "你與 " + mData1.get(position) + "已經是好朋友囉!", Toast.LENGTH_SHORT).show();
                    mData1.remove(position);
                    mData2.remove(position);
                    notifyItemRemoved(position);
                }
            });
            holder.N_response.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    final String N_u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
                    check_friend = new Connect_To_Server();
                    Thread res_friend_invitations = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            check_friend.connect("insert_sql","DELETE FROM `user_friends_invitation` WHERE `user_friends_invitation`.`user_id` = '"+mData2.get(position)+"' AND `user_friends_invitation`.`friend_id` = '"+N_u_id+"'");
                        }
                    });
                    res_friend_invitations.start();
                    Toast.makeText(setting_friend_search.this, "已刪除 " + mData1.get(position) + "的好友邀請", Toast.LENGTH_SHORT).show();
                    mData1.remove(position);
                    mData2.remove(position);
                    notifyItemRemoved(position);
                }
            });
            holder.itemView.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Toast.makeText(setting_friend_search.this, "Item " + position + " is clicked.", Toast.LENGTH_SHORT).show();
                }
            });
            holder.itemView.setOnLongClickListener(new View.OnLongClickListener() {
                @Override
                public boolean onLongClick(View v) {
                    Toast.makeText(setting_friend_search.this, "Item " + position + " is long clicked.", Toast.LENGTH_SHORT).show();
                    return true;
                }
            });
        }

        @Override
        public int getItemCount() {
            return mData1.size();
        }
    }

}
