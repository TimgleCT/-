package com.example.user.myapplication;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.HapticFeedbackConstants;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.AlphaAnimation;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.GridView;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.PopupWindow;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

import static com.example.user.myapplication.mainpage.KEY;

public class setting_friend extends AppCompatActivity {
    Button add_friend_btn, friend_delete;
    View timekeeper_logo;

    private GridView gridView;
    private boolean isShowDelete = false;
    private GridViewAdapter mGridAdapter;
    private List<AreaEntity> areaEneities = new ArrayList<AreaEntity>();
    AlphaAnimation alphaAnimation1 = new AlphaAnimation(1.0f, 0.1f);
    AlphaAnimation alphaAnimation2 = new AlphaAnimation(0.1f, 1.0f);

    // hamburger
    Button menu;
    ImageButton menu_open;
    PopupWindow popupWindow;
    FrameLayout menu_window;
    TextView set_up, friend, check;
    Connect_To_Server find_friends,delete_friend;
    JSONArray get_fresult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.setting_friend);
        find_friends = new Connect_To_Server();
        final String u_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
        Thread search_friend = new Thread(new Runnable() {
            @Override
            public void run() {
                find_friends.connect("select_sql","SELECT user.u_name,user.user_id FROM `user` WHERE user.user_id = any(SELECT user_friends.friend_id FROM `user_friends` WHERE user_friends.user_id =  '"+u_id +"')");
            }
        });
        search_friend.start();
        try {
            search_friend.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        try{
            get_fresult = new JSONArray(find_friends.get_data);
            int lenght = get_fresult.length();
            //增加好友
            gridView = findViewById(R.id.friend_list);
            for(int i = 0;i < lenght;i++){
                JSONObject jsonObject = get_fresult.getJSONObject(i);
                AreaEntity areaEntity = new AreaEntity(jsonObject.getString("user_id")+ "", R.drawable.ai_open, "" +jsonObject.getString("u_name"));
                areaEneities.add(areaEntity);

                //好友長按事件
                mGridAdapter = new GridViewAdapter(areaEneities, setting_friend.this);
                gridView.setAdapter(mGridAdapter);
                gridView.setOnItemLongClickListener(new AdapterView.OnItemLongClickListener() {
                    @Override
                    public boolean onItemLongClick(AdapterView<?> parent, View view, int position, long id) {
                        if (isShowDelete) {
                            isShowDelete = false;
                        } else {
                            isShowDelete = true;
                        }
                        mGridAdapter.setIsShowDelete(isShowDelete);
                        view.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS);
                        return true;
                    }
                });
            }
        }
        catch(JSONException e) {
            e.printStackTrace();
        }


        //點選 ADD FRIEND 按鈕
        add_friend_btn = findViewById(R.id.add_friend_btn);
        add_friend_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent1 = new Intent(setting_friend.this, setting_friend_search.class);
                startActivity(intent1);
            }
        });

        //回首頁
        timekeeper_logo = findViewById(R.id.timekeeper_logo);
        timekeeper_logo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent2 = new Intent(setting_friend.this, mainpage.class);
                startActivity(intent2);
            }
        });

        // 選單彈跳
        menu = findViewById(R.id.menu);
        menu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(popupWindow==null){
                    showPopupWindow show = new showPopupWindow(setting_friend.this,getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null));
                    try {
                        show.showPopupWindow(menu);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    //showPopupWindow();
                }else if(popupWindow.isShowing()){
                    popupWindow.dismiss();
                }
                else{
                    popupWindow.showAsDropDown(menu,-200,-155);
                }
            }
        });
    }

    // 按返回鍵取消delete狀態
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            if (isShowDelete == true) {
                isShowDelete = false;
                mGridAdapter.setIsShowDelete(false);
            } else {
                finish();
            }
        }
        return false;
    }


    // 按空白處取消delete狀態
    public boolean dispatchTouchEvent(MotionEvent ev) {
        if (ev.getAction() == MotionEvent.ACTION_DOWN) {

            View v = gridView;

            if (isShouldHideInput(v, ev)) {
                if (isShowDelete == true) {
                    isShowDelete = false;
                    mGridAdapter.setIsShowDelete(false);
                }
            }
        }
        return super.dispatchTouchEvent(ev);
    }

    private boolean isShouldHideInput(View v, MotionEvent event) {
        if (v != null) {
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


    class GridViewAdapter extends BaseAdapter {
        private boolean isShowDelete; //根据这个布尔型变量来判断是否显示删除图标，true是显示，false是不显示
        private List<AreaEntity> areaEneity;
        private LayoutInflater mInflater;
        private Context mContext;

        public GridViewAdapter(List<AreaEntity> areaEneities, Context context) {
            super();
            this.areaEneity = areaEneities;
            this.mInflater = LayoutInflater.from(context);
            this.mContext = context;
        }

        public void setIsShowDelete(boolean isShowDelete) {
            this.isShowDelete = isShowDelete;
            notifyDataSetChanged();
        }

        @Override
        public int getCount() {
            return areaEneity.size();
        }

        @Override
        public Object getItem(int position) {
            return areaEneity.get(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(final int position, View convertView, ViewGroup parent) {

            ViewHolder holder;
            if (convertView == null) {
                convertView = mInflater.inflate(R.layout.setting_friend_item, null);
                holder = new ViewHolder();
                holder.img = convertView.findViewById(R.id.friend_photo);
                holder.name_tv = convertView.findViewById(R.id.friend_name);
                holder.delete_btn = convertView.findViewById(R.id.friend_delete);
                convertView.setTag(holder);
            } else {
                holder = (ViewHolder) convertView.getTag();
            }
            holder.name_tv.setText(areaEneity.get(position).getArea());
            holder.delete_btn.setVisibility(isShowDelete ? View.VISIBLE : View.GONE);

            final String my_id = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
            final String f_id = areaEneity.get(position).getId();
            final Bitmap[] img = new Bitmap[1];
            Thread get_photo = new Thread(new Runnable() {
                @Override
                public void run() {
                    img[0] = get_u_sticker.get_sticker("http://140.127.218.207/uploads/"+f_id+".png");
                }
            });
            get_photo.start();
            try {
                get_photo.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if(img[0] !=null){
                holder.img.setImageBitmap(img[0]);
            }
            holder.img.setAlpha(isShowDelete ? 0.5f : 1.0f);
            holder.delete_btn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    delete_friend = new Connect_To_Server();
                    Thread res_friend_invitations = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            delete_friend.connect("insert_sql","DELETE FROM `user_friends` WHERE `user_friends`.`user_id` = '"+my_id+"' AND `user_friends`.`friend_id` = '"+f_id+"'");
                            delete_friend.connect("insert_sql","DELETE FROM `user_friends` WHERE `user_friends`.`user_id` = '"+f_id+"' AND `user_friends`.`friend_id` = '"+my_id+"'");
                        }
                    });
                    res_friend_invitations.start();
                    Toast.makeText(setting_friend.this, "已刪除 " +areaEneity.get(position).getArea()+ "好友", Toast.LENGTH_SHORT).show();
                    areaEneity.remove(areaEneity.get(position));
                    notifyDataSetChanged();
                }
            });
            return convertView;
        }

        class ViewHolder {
            private Button delete_btn;
            private ImageView img;
            private TextView name_tv;
        }
    }

    class AreaEntity {
        private String id;
        private int pic;
        private String area;

        public AreaEntity(String id, int pic, String area) {
            this.id = id;
            this.pic = pic;
            this.area = area;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public Integer getPic() {
            return pic;
        }

        public void setPic(Integer pic) {
            this.pic = pic;
        }

        public String getArea() {
            return area;
        }

    }
}
