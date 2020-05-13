package com.example.user.myapplication;

import android.Manifest;
import android.app.Activity;
import android.app.AlarmManager;
import android.app.AlertDialog;
import android.app.AppOpsManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.support.v4.app.ActivityCompat;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.animation.Animation;
import android.view.animation.TranslateAnimation;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.example.user.myapplication.ai_group.ai_Adapter;
import com.example.user.myapplication.guide.guide_page;
import com.nikhilpanju.recyclerviewenhanced.OnActivityTouchListener;
import com.nikhilpanju.recyclerviewenhanced.RecyclerTouchListener;

import java.sql.RowId;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

import cdflynn.android.library.crossview.CrossView;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;

public class mainpage extends Activity implements RecyclerTouchListener.RecyclerTouchListenerHelper {
    public static final String KEY = "com.example.user.myapplication.app";
    public static boolean logon = false;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    ImageButton add_btn, normal_btn, ai_btn, counter_btn;
    LinearLayout normal_layout, ai_layout, counter_layout;
    FrameLayout frame_layout;
    CrossView crossView;
    List<Integer> itemlist = new ArrayList<>();
    List<Integer> requestcode = new ArrayList<>();
    AlertDialog dialog;
    DB_usage db;

    // footer
    public static final int TYPE_FOOTER = 1;  //说明是带有Footer的
    public static final int TYPE_NORMAL = 2;  //说明是不带有header和footer的

    View add_background;

    // hamburger
    Button menu;
    ImageButton menu_open, qus;
    LinearLayout qus_view;
    PopupWindow popupWindow;
    FrameLayout menu_window;
    TextView set_up, friend, check;
    TextView textView1, textView2, textView3;
    private ai_Adapter mMyAdapter;


    public class BuildDev {
        public static final int RECORD_AUDIO = 0;
    }

    //recyclerview
    RecyclerView mRecyclerView;
    MainAdapter mAdapter;
    String[] dialogItems;
    List<Integer> unclickableRows, unswipeableRows;
    private RecyclerTouchListener onTouchListener;
    private int openOptionsPosition;
    private OnActivityTouchListener touchListener;
    String[] alarmtype = new String[50];
    ImageView photo_sticker;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.mainpage);

        add_background = findViewById(R.id.add_background);

        add_btn = findViewById(R.id.add_btn);
        ai_btn = findViewById(R.id.ai_btn);
        ai_layout = findViewById(R.id.ai_layout);
        normal_btn = findViewById(R.id.normal_btn);
        normal_layout = findViewById(R.id.normal_layout);
//        counter_btn = findViewById(R.id.counter_btn);
//        counter_layout = findViewById(R.id.counter_layout);
        crossView = findViewById(R.id.cross_view);
        frame_layout = findViewById(R.id.framelayout);

//        qus = findViewById(R.id.qus);
//        qus_view = findViewById(R.id.qus_view);
//        qus.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Intent intent = new Intent(mainpage.this, guide_page.class);
//                startActivity(intent);
//                finish();
//            }
//        });

        textView1 = findViewById(R.id.textView1);
        textView1.setText("當鬧鐘響後\n" +
                "我會記錄你的使用行為\n" +
                "用於之後的AI訓練");

        textView2 = findViewById(R.id.textView2);
        textView2.setText("起床後請到「設定醒/睡著」\n" +
                "讓我知道你剛剛是醒著還是睡著喔！\n" +
                "這樣我才能學習呦");

        textView3 = findViewById(R.id.textView3);
        textView3.setText("當跳出頁面後就無法修改囉！！\n");

        String user = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null);
        String pwd = getSharedPreferences(KEY, MODE_PRIVATE).getString("u_pwd", null);
        Log.d("測試", "暫存" + user + "/" + pwd);
        if (user == null || pwd == null) {
            Intent intent = new Intent(this, login.class);
            startActivity(intent);
        }

        normal_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent1 = new Intent(mainpage.this, normal_alarm.class);
                startActivity(intent1);
                closeMenu();
            }
        });

        ai_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent1 = new Intent(mainpage.this, ai_alarm.class);
                startActivity(intent1);
                closeMenu();
            }
        });

//        counter_btn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Intent intent1 = new Intent(mainpage.this, counter_countdown.class);
//                startActivity(intent1);
//                closeMenu();
//            }
//        });

        Intent service = new Intent(this, Friend_Invite_Service.class);
        service.putExtra("my_id", user);
        startService(service);
        Log.d("背景", "測試");
        Intent group_service = new Intent(this, Group_service.class);
        group_service.putExtra("my_id", user);
        startService(group_service);

        //recyclerview
        unclickableRows = new ArrayList<>();
        unswipeableRows = new ArrayList<>();
        dialogItems = new String[25];
        for (int i = 0; i < 25; i++) {
            dialogItems[i] = String.valueOf(i + 1);
        }

        mRecyclerView = findViewById(R.id.recyclerView);
        mAdapter = new MainAdapter(this, getData());
        mRecyclerView.setAdapter(mAdapter);
        mRecyclerView.setLayoutManager(new LinearLayoutManager(this));

        setFooterView(mRecyclerView);
        mAdapter.getFooterView().setEnabled(false);
        mAdapter.getFooterView().setFocusable(false);
        mAdapter.getFooterView().setClickable(false);


        //點擊空白處，add_btn選項會消失
        add_background.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (normal_layout.getVisibility() == View.VISIBLE && ai_layout.getVisibility() == View.VISIBLE) {
                    closeMenu();
                }
            }
        });

        onTouchListener = new RecyclerTouchListener(this, mRecyclerView);
        onTouchListener
                .setIndependentViews(R.id.rowButton)
                .setViewsToFade(R.id.rowButton)
                .setClickable(new RecyclerTouchListener.OnRowClickListener() {
                    @Override
                    public void onRowClicked(int position) {
                        Intent intent;
                        if (alarmtype[position].equals("normal")) {
                            intent = new Intent(mainpage.this, normal_alarm.class);
                        } else {
                            intent = new Intent(mainpage.this, ai_alarm.class);
                        }
                        intent.putExtra("requestcode", requestcode.get(position));
                        startActivity(intent);

                    }

                    @Override
                    public void onIndependentViewClicked(int independentViewID, int position) {
//                        ToastUtil.makeToast(getApplicationContext(), "Button in row " + (position + 1) + " clicked!");
                    }
                })
                .setLongClickable(true, new RecyclerTouchListener.OnRowLongClickListener() {
                    @Override
                    public void onRowLongClicked(int position) {
//                        ToastUtil.makeToast(getApplicationContext(), "Row " + (position + 1) + " long clicked!");
                    }
                })
                .setSwipeOptionViews(R.id.delete)
                .setSwipeable(R.id.rowFG, R.id.rowBG, new RecyclerTouchListener.OnSwipeOptionsClickListener() {
                    @Override
                    public void onSwipeOptionClicked(int viewID, int position) {
                        String message = "";
                        if (viewID == R.id.delete) {
                            message += "Change";
                        }
                    }
                });
    }

    //設定add btn的顯示與隱藏
    private void openMenu() {
        mRecyclerView.setLayoutFrozen(true);
        normal_layout.setVisibility(View.VISIBLE);
        ai_layout.setVisibility(View.VISIBLE);
//        counter_layout.setVisibility(View.VISIBLE);
        crossView.cross();
        add_background.setVisibility(View.VISIBLE);
    }

    private void closeMenu() {
        mRecyclerView.setLayoutFrozen(false);
        normal_layout.setVisibility(View.GONE);
        ai_layout.setVisibility(View.GONE);
//        counter_layout.setVisibility(View.GONE);
        crossView.plus();
        add_background.setVisibility(View.GONE);
    }

    //recyclerview
    @Override
    protected void onResume() {
        super.onResume();
//        checkusage();
        mAdapter = new MainAdapter(this, getData());
        mRecyclerView.setAdapter(mAdapter);
        mRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        mRecyclerView.addOnItemTouchListener(onTouchListener);

        // 選單彈跳
        menu = findViewById(R.id.menu);
        menu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (popupWindow == null) {
                    showPopupWindow show = new showPopupWindow(mainpage.this, getSharedPreferences(KEY, MODE_PRIVATE).getString("u_id", null));
                    try {
                        show.showPopupWindow(menu);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    //showPopupWindow();
                } else if (popupWindow.isShowing()) {
                    popupWindow.dismiss();
                } else {
                    popupWindow.showAsDropDown(menu, 0, -155);
                }
            }
        });

        //設定增加的子按鈕顯示或隱藏
        add_btn.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {
                if (normal_layout.getVisibility() == View.VISIBLE && ai_layout.getVisibility() == View.VISIBLE) {
                    closeMenu();
                } else {
                    openMenu();
                }
            }
        });

        //Permission
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            if (Build.VERSION.SDK_INT >=  Build.VERSION_CODES.M){
                if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                        Manifest.permission.RECORD_AUDIO)) {
                    final AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage("ＡＩ需要開啟麥克風及九軸的權限，才能進行ＡＩ收集！請麻煩一定要開啟權限喔。");
                    builder.setPositiveButton("我知道了", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            Uri packageURI = Uri.parse("package:" + "com.example.user.timekeeper_testtest");
                            Intent appintent= new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, packageURI);
                            startActivity(appintent);
                        }
                    });
                    dialog = builder.show();
                    builder.show();
                }else {
                    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO},
                            BuildDev.RECORD_AUDIO);
                }
            }
        }
        int permission = ActivityCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            if (Build.VERSION.SDK_INT >=  Build.VERSION_CODES.M){
                if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                        Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    final AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage("提供存取權限，才能記錄鬧鐘以及選音樂喔！");
                    builder.setPositiveButton("我知道了", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            Uri packageURI = Uri.parse("package:" + "com.example.user.timekeeper_testtest");
                            Intent appintent= new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, packageURI);
                            startActivity(appintent);
                        }
                    });
                    dialog = builder.show();
                    builder.show();
                }else {
                    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                            REQUEST_EXTERNAL_STORAGE);
                }
            }else {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                        REQUEST_EXTERNAL_STORAGE);
            }
        }

    }

    @Override
    protected void onPause() {
        super.onPause();
        if (dialog != null && dialog.isShowing()){
            dialog.dismiss();
            dialog.cancel();
        }
        mRecyclerView.removeOnItemTouchListener(onTouchListener);
    }

    private List<mainpage_RowModel> getData() {
        List<mainpage_RowModel> list = new ArrayList<>(25);
        DB_normal_alarm db = new DB_normal_alarm(this);

        //cursor
        if (db != null) {
            Cursor cursor = db.select();
            if (cursor.getCount() > 0) {
                int i = 0;
                for (cursor.moveToFirst(); !cursor.isAfterLast(); cursor.moveToNext()) {
                    Calendar cal = Calendar.getInstance();
                    Long t = Long.parseLong(cursor.getString(6));
                    cal.setTimeInMillis(t);
                    int hour = cal.get(Calendar.HOUR_OF_DAY);
                    int min = cal.get(Calendar.MINUTE);
                    String time = "";
                    if (hour < 10 && min < 10) {
                        time = "0" + hour + ":0" + min;
                    } else if (hour < 10 && min >= 10) {
                        time = "0" + hour + ":" + min;
                    } else if (hour >= 10 && min < 10) {
                        time = hour + ":0" + min;
                    } else if (hour >= 10 && min >= 10) {
                        time = hour + ":" + min;
                    }
                    list.add(new mainpage_RowModel(cursor.getString(0), cursor.getString(1), cursor.getString(2), cursor.getInt(3), Boolean.parseBoolean(cursor.getString(4))
                            , cursor.getString(5), time, cursor.getString(7), cursor.getInt(8)));
                    requestcode.add(i, cursor.getInt(3));
                    alarmtype[i] = cursor.getString(7);
                    itemlist.add(i, i);
                    i++;
                }
            }
        }
        return list;
    }


    @Override
    public boolean dispatchTouchEvent(MotionEvent ev) {
        try {
            if (touchListener != null) {
                touchListener.getTouchCoordinates(ev);
            }
            return super.dispatchTouchEvent(ev);
        } catch (Exception e) {
            return false;
        }

    }

    @Override
    public void setOnActivityTouchListener(OnActivityTouchListener listener) {
        this.touchListener = listener;
    }

    private class MainAdapter extends RecyclerView.Adapter<MainAdapter.MainViewHolder> {
        LayoutInflater inflater;
        List<mainpage_RowModel> modelList;

        // footer
        public static final int TYPE_FOOTER = 1;  //说明是带有Footer的
        public static final int TYPE_NORMAL = 2;  //说明是不带有header和footer的
        private View mFooterView;

        //FooterView的get和set函数
        public View getFooterView() {
            return mFooterView;
        }

        public void setFooterView(View footerView) {
            mFooterView = footerView;
            mFooterView.setClickable(false);
            mFooterView.setFocusable(false);
            mFooterView.setEnabled(false);
            notifyItemInserted(getItemCount() - 1);
        }

        @Override
        public int getItemViewType(int position) {
            if (mFooterView == null) {
                return TYPE_NORMAL;
            }
            if (position == getItemCount() - 1) {
                //最后一个,应该加载Footer
                return TYPE_FOOTER;
            }
            return TYPE_NORMAL;
        }


        public MainAdapter(Context context, List<mainpage_RowModel> list) {
            inflater = LayoutInflater.from(context);
            modelList = new ArrayList<>(list);
        }

        @Override
        public MainViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            if (mFooterView != null && viewType == TYPE_FOOTER) {
                return new MainViewHolder(mFooterView);
            }
            View view = inflater.inflate(R.layout.mainpage_alarm_row, parent, false);
            return new MainViewHolder(view);
        }

        @Override
        public void onBindViewHolder(final MainViewHolder holder, final int position) {
//            holder.bindData(modelList.get(position));

            //增加footer
            if (getItemViewType(position) == TYPE_NORMAL) {
                if (holder instanceof MainViewHolder) {
                    //这里加载数据的时候要注意，是从position-1开始，因为position==0已经被header占用了
                    holder.bindData(modelList.get(position));

                    //點擊刪除layout後的動作
                    holder.delete.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View view) {
                            DB_normal_alarm db = new DB_normal_alarm(mainpage.this);
                            AlarmManager alarmManager = (AlarmManager) getSystemService(Context.ALARM_SERVICE);
                            Intent intent = new Intent(mainpage.this, normal_alarmalert.class);
                            PendingIntent pi = PendingIntent.getActivity(mainpage.this, requestcode.get(position), intent, PendingIntent.FLAG_CANCEL_CURRENT);
                            alarmManager.cancel(pi);
                            db.delete(requestcode.get(position));
                            removeData(itemlist.get(position));
                            if (db != null) {
                                Cursor cursor = db.select();
                                if (cursor.getCount() > 0) {
                                    int i = 0;
                                    for (cursor.moveToFirst(); !cursor.isAfterLast(); cursor.moveToNext()) {
                                        requestcode.add(i, cursor.getInt(3));
                                        i++;
                                    }
                                }
                            }
                            db.close();
                        }
                    });
                    return;
                }
                return;
            } else {
                return;
            }

        }

        //刪除鬧鐘
        public void removeData(int position) {
            modelList.remove(position);
            //删除动画
            notifyItemRemoved(position);
            notifyItemRemoved(position);
            for (int a = position+1; (itemlist.size()) > a; a++){
                itemlist.set(a, itemlist.get(a)-1);
            }
        }

        @Override
        public int getItemCount() {
//            return modelList.size();
            if (mFooterView == null) {
                return modelList.size();
            } else {
                return modelList.size() + 1;
            }
        }


        class MainViewHolder extends RecyclerView.ViewHolder {

            TextView mainText, repeatday, alarm_time;
            TextView tv;
            Button alarm_btn;
            LinearLayout alarm, rowFG;
            RelativeLayout delete;
            int state, requestcode;
            String type;

            public MainViewHolder(View itemView) {
                super(itemView);
                mainText = itemView.findViewById(R.id.mainText);
                repeatday = itemView.findViewById(R.id.repeatday);
                alarm_time = itemView.findViewById(R.id.alarm_time);
                delete = itemView.findViewById(R.id.delete);
                rowFG = itemView.findViewById(R.id.rowFG);

                //加入footer
                if (itemView == mFooterView) {
                    return;
                }

                // 開啟/關閉鬧鐘
                alarm = itemView.findViewById(R.id.alarm);
                alarm_btn = itemView.findViewById(R.id.rowButton);
                alarm_btn.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        DB_normal_alarm db = new DB_normal_alarm(mainpage.this);
                        if (type.equals("ai")) {
                            switch (state) {
                                case 1:
                                    alarm_btn.setBackground(getResources().getDrawable(R.drawable.ai_close));
                                    alarm.setBackground(getResources().getDrawable(R.drawable.mainpage_alarm_background_close));
                                    state = 0;
                                    db.updatestate(requestcode, state);

                                    AlarmManager alarmManager = (AlarmManager) getSystemService(Context.ALARM_SERVICE);
                                    Intent intent = new Intent(mainpage.this, ai_alarmalert.class);
                                    PendingIntent pi = PendingIntent.getActivity(mainpage.this, requestcode, intent, PendingIntent.FLAG_UPDATE_CURRENT);
                                    alarmManager.cancel(pi);
                                    break;

                                case 0:
                                    alarm_btn.setBackground(getResources().getDrawable(R.drawable.ai_open));
                                    alarm.setBackground(getResources().getDrawable(R.drawable.mainpage_alarm_background));
                                    state = 1;
                                    db.updatestate(requestcode, state);
                                    Intent service = new Intent(mainpage.this, BootService.class);
                                    service.putExtra("req",requestcode);
                                    startService(service);
                                    break;
                            }
                        } else {
                            switch (state) {
                                case 1:
                                    alarm_btn.setBackground(getResources().getDrawable(R.drawable.normal_close));
                                    alarm.setBackground(getResources().getDrawable(R.drawable.mainpage_alarm_background_close));
                                    state = 0;
                                    db.updatestate(requestcode, state);

                                    AlarmManager alarmManager = (AlarmManager) getSystemService(Context.ALARM_SERVICE);
                                    Intent intent = new Intent(mainpage.this, normal_alarmalert.class);
                                    PendingIntent pi = PendingIntent.getActivity(mainpage.this, requestcode, intent, PendingIntent.FLAG_UPDATE_CURRENT);
                                    alarmManager.cancel(pi);
                                    break;

                                case 0:
                                    alarm_btn.setBackground(getResources().getDrawable(R.drawable.normal));
                                    alarm.setBackground(getResources().getDrawable(R.drawable.mainpage_alarm_background));
                                    state = 1;
                                    db.updatestate(requestcode, state);
                                    Intent service = new Intent(mainpage.this, BootService.class);
                                    service.putExtra("req",requestcode);
                                    startService(service);
                                    break;
                            }
                        }
                    }
                });
            }

            public void bindData(mainpage_RowModel rowModel) {
                mainText.setText(rowModel.getNormal_edit_title());
                repeatday.setText(rowModel.getRepeat());
                alarm_time.setText(rowModel.getAlarm_time());
                type = rowModel.getType();
                state = rowModel.getState();
                requestcode = rowModel.getRequestcode();

                if (state == 0) {
                    alarm.setBackground(getResources().getDrawable(R.drawable.mainpage_alarm_background_close));
                    if (type.equals("ai")){
                        alarm_btn.setBackground(getResources().getDrawable(R.drawable.ai_close));
                    }else{
                        alarm_btn.setBackground(getResources().getDrawable(R.drawable.normal_close));
                    }
                } else {
                    alarm.setBackground(getResources().getDrawable(R.drawable.mainpage_alarm_background));
                    if (type.equals("normal")) {
                        alarm_btn.setBackground(getResources().getDrawable(R.drawable.normal));
                    } else if (type.equals("ai")) {
                        alarm_btn.setBackground(getResources().getDrawable(R.drawable.ai_open));
                    }
                }
            }
        }

    }

    private void setFooterView(RecyclerView view) {
        View footer = LayoutInflater.from(this).inflate(R.layout.footer, view, false);
        footer.setClickable(false);
        footer.setFocusable(false);
        footer.setEnabled(false);
        mAdapter.setFooterView(footer);
    }

    private boolean isAccessGranted() {
        try {
            PackageManager packageManager = getPackageManager();
            ApplicationInfo applicationInfo = packageManager.getApplicationInfo(getPackageName(), 0);
            AppOpsManager appOpsManager = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
            int mode = 0;
            if (android.os.Build.VERSION.SDK_INT > android.os.Build.VERSION_CODES.KITKAT) {
                mode = appOpsManager.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS,
                        applicationInfo.uid, applicationInfo.packageName);
            }
            return (mode == AppOpsManager.MODE_ALLOWED);

        } catch (PackageManager.NameNotFoundException e) {
            return false;
        }
    }

//    public void checkusage(){
//        db = new DB_usage(this);
//        if (db != null) {
//            Cursor cursor = db.select();
//            if (cursor.getCount() > 0) {
//                for (cursor.moveToFirst(); !cursor.isAfterLast(); cursor.moveToNext()) {
//                    if (cursor.getString(3) == null) {
//                        Intent pageintent = new Intent(this, check.class);
//                        startActivity(pageintent);
//                        db.close();
//                        finish();
//                    }
//                }
//            }
//        }
//        db.close();
//    }

}