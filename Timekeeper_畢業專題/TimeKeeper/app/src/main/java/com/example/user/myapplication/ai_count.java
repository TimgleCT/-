package com.example.user.myapplication;

import android.app.AlarmManager;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Handler;
import android.os.PowerManager;
import android.support.v4.app.NotificationCompat;
import android.support.v7.app.AlertDialog;
import android.text.format.DateFormat;
import android.util.Log;
import android.view.WindowManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

import static android.content.Context.MODE_PRIVATE;
import static com.example.user.myapplication.mainpage.KEY;

//extends AppCompatActivity
public class ai_count{
    public static int clock_count = 0;
    long stopuse;
    int sec;
    static String state;
    static double volume;
    private DB_usage dbUsage;
    private DB_soundaxis dbSoundaxis;
    private IntentFilter theFilter;
    private ScreenBroadReceiver mScreenReceiver;
    public SensorManager sensorManager;
    Get_Sound sound_recorder;
    Nine_axis_record axis_recorder;
    public Date Today;
    public Timestamp timestamp;
    public SimpleDateFormat record_time;
    public static ArrayList id;
    public static ArrayList name;
    public static ArrayList date_time;
    public static ArrayList x_axis;
    public static ArrayList y_axis;
    public static ArrayList z_axis;
    public static ArrayList sound_db;
    ArrayList up_id,up_date_alarm,up_date_time,up_x_axis,up_y_axis,up_z_axis,up_sound_db;
    ArrayList up_id_u,up_date_time_u,up_period_u,up_awake_u;
    String sql,sqltmp,sql_u,sql_u_tmp,u_id;
    Connect_To_Server connecting;
    Context record ;
    data_img_prediction produce_img;
    public static int requestcode;
    public static long timedate;
    long ttdd;
    List<String> idlist;

    public void record(Context context){
        this.record = context;
        sensorManager = (SensorManager)record.getSystemService(Context.SENSOR_SERVICE);
        theFilter = new IntentFilter();
        theFilter.addAction(Intent.ACTION_SCREEN_ON);
        theFilter.addAction(Intent.ACTION_SCREEN_OFF);
        theFilter.addAction(Intent.ACTION_USER_PRESENT);

        mScreenReceiver = new ScreenBroadReceiver();
        sound_recorder = new Get_Sound();
        axis_recorder = new Nine_axis_record();
        date_time =new ArrayList();
        id = new ArrayList();
        name = new ArrayList();
        x_axis = new ArrayList();
        y_axis = new ArrayList();
        z_axis = new ArrayList();
        sound_db = new ArrayList();
        up_id = new ArrayList();
        up_date_time = new ArrayList();
        up_date_alarm = new ArrayList();
        up_x_axis = new ArrayList();
        up_y_axis = new ArrayList();
        up_z_axis = new ArrayList();
        up_sound_db = new ArrayList();
        up_id_u = new ArrayList();
        up_date_time_u = new ArrayList();
        up_awake_u  = new ArrayList();
        up_period_u = new ArrayList();
        dbSoundaxis = new DB_soundaxis(record);
        dbUsage = new DB_usage(record);
        sql = new String();
        sqltmp = new String();
        sql_u = new String();
        sql_u_tmp = new String();
        u_id = new String();
        connecting = new Connect_To_Server();
        produce_img = new data_img_prediction();

        u_id = record.getSharedPreferences(KEY,MODE_PRIVATE).getString("u_id",null);

        Calendar cc = Calendar.getInstance();
        cc.setTimeInMillis(System.currentTimeMillis());
        ttdd = cc.getTimeInMillis();
        getScreen(ttdd);
        if (timedate == 0){
            timedate = ttdd;
        }
//        start_time = System.currentTimeMillis();
//        stop_record_time = start_time+ 60*1000;//結束時間設為一分鐘後
        Log.d("state", "s"+state);
        if(state == "true"){
            start_listen_nine_axis();
            sound_recorder.startRecord();//分貝開始記錄
            startListenAudio();//存取資料在陣列裡的執行續
            Log.d("紀錄", "開始");
        }
        CheckState_SubmitRecord();
    }

    public void usagetime(long stopuse){
        Calendar calendar1 = Calendar.getInstance();
        calendar1.setTimeInMillis(ttdd);
        Calendar calendar2 = Calendar.getInstance();
        calendar2.setTimeInMillis(stopuse);
        Long tt = calendar2.getTimeInMillis() - calendar1.getTimeInMillis();
        sec = (int) (tt/1000);
        Log.d("totla",":"+tt/1000);
    }

    public void getScreen(Long t){
        PowerManager pm = (PowerManager)record.getSystemService(Context.POWER_SERVICE);
        boolean isScreenOn = pm.isInteractive();
        Calendar fivemin = Calendar.getInstance();
        fivemin.setTimeInMillis(System.currentTimeMillis());
        Long five = fivemin.getTimeInMillis();
        Calendar fivmin2 = Calendar.getInstance();
        fivmin2.setTimeInMillis(t+60000);
        Long five2 = fivmin2.getTimeInMillis();
        if (five < five2){
            if (isScreenOn == true){
                state = "true";
            }else if (isScreenOn == false){
                state = "false";
            }
        }else {
            state = "false";
        }
    }

    public void start_listen_nine_axis(){
        Thread start = new Thread((new Runnable() {
            @Override
            public void run() {
                axis_recorder.start_record(sensorManager, true);
            }
        }));
        start.start();
    }
    public void CheckRecordtime_SubmitRecord(){
        Thread thread_check_time = new Thread(new Runnable() {
            @Override
            public void run() {
                Cursor update_cursor = dbSoundaxis.select_update();
                for(int i = 0; i<update_cursor.getCount();i++){
                    update_cursor.moveToPosition(i);
                    up_id.add(update_cursor.getString(update_cursor.getColumnIndex("_id")));
                    up_date_alarm.add(update_cursor.getString(update_cursor.getColumnIndex("date_alarm")));
                    up_date_time.add((update_cursor.getString(update_cursor.getColumnIndex("date_time"))));
                    up_x_axis.add((update_cursor.getString(update_cursor.getColumnIndex("x_axis"))));
                    up_y_axis.add((update_cursor.getString(update_cursor.getColumnIndex("y_axis"))));
                    up_z_axis.add((update_cursor.getString(update_cursor.getColumnIndex("z_axis"))));
                    up_sound_db.add((update_cursor.getString(update_cursor.getColumnIndex("sound_db"))));
                }
                sql = "INSERT INTO `sound_axis_record` (`User_id`, `Date_alarm`, `Date_time`, `X_axis`, `Y_axis`, `Z_axis`, `Sound_db`) VALUES";
                for(int i = 0;i<up_id.size();i++){
                    String id = up_id.get(i).toString();
                    String date_alarm = (String) up_date_alarm.get(i);
                    Timestamp time = Timestamp.valueOf(up_date_time.get(i).toString());
                    double x_axis = Double.parseDouble(up_x_axis.get(i).toString());
                    double y_axis = Double.parseDouble(up_y_axis.get(i).toString());
                    double z_axis = Double.parseDouble(up_z_axis.get(i).toString());
                    double sound_db = Double.parseDouble(up_sound_db.get(i).toString());
                    if(i == up_id.size()-1){
                        sqltmp = sqltmp+"('"+id+"','"+date_alarm+"', '"+time+"', '"+x_axis+"', '"+y_axis+"', '"+z_axis+"', '"+sound_db+"');";
                    }else {
                        sqltmp = sqltmp+"('"+id+"','"+date_alarm+"', '"+time+"', '"+x_axis+"', '"+y_axis+"', '"+z_axis+"', '"+sound_db+"'),";
                    }
                }
                sql = sql+sqltmp;
                Log.d("sql語法",sql);
                connecting.connect("insert_sql",sql);
                try {
                    Thread.sleep(400);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                Log.d("測試","狀態"+connecting.internet_connect);
                if(connecting.internet_connect){
                    dbSoundaxis.update_state_change();
                    dbSoundaxis.deleteAll();
                }
            }
        });
        thread_check_time.start();
        try {
            thread_check_time.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Thread produce = new Thread(new Runnable() {
            @Override
            public void run() {
                produce_img.produce_img(String.valueOf(ttdd));
            }
        });
        produce.start();
        try {
            produce.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Log.d("回應：","安安"+produce_img.AI_response);
        int res = Integer.parseInt(produce_img.AI_response.trim());
        if(res == 1){
            Thread update_awake_1 = new Thread(new Runnable() {
                @Override
                public void run() {
                    connecting.connect("insert_sql","UPDATE `screen_record` SET `r_ifawake` = '1' WHERE `screen_record`.`Date` = '"+String.valueOf(ttdd)+"';");
                }
            });
            update_awake_1.start();
            sendNotification("1",String.valueOf(ttdd));
        }else if (res == 0){
            Thread update_awake_0 = new Thread(new Runnable() {
                @Override
                public void run() {
                    connecting.connect("insert_sql","UPDATE `screen_record` SET `r_ifawake` = '0' WHERE `screen_record`.`Date` = '"+String.valueOf(ttdd)+"';");
                }
            });
            update_awake_0.start();
            sendNotification("0",String.valueOf(ttdd));

            int count = asleeptimes(String.valueOf(timedate));
            Log.d("inttimes",""+count);
            if(count < 1){
                count++;
                Log.d("安安",""+count);
                updateasleep(count,String.valueOf(timedate));
            }else if(count == 1){
                //送通知
                String[] grouplist = getfriend(requestcode);
                Log.d("group列表",""+grouplist);
                if(grouplist!=null){
                    String group_sql ="INSERT INTO `alarm_group_Notification` (`user_id`, `friend_id`) VALUES";
                    for(int i = 0;i<grouplist.length;i++){
                        if(i == grouplist.length-1){
                            group_sql = group_sql+"('"+u_id+"', '"+grouplist[i]+"');";
                        }else {
                            group_sql = group_sql+"('"+u_id+"', '"+grouplist[i]+"'),";
                        }
                    }
                    final String finalGroup_sql = group_sql;
                    Thread sendgroupsql = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            connecting.connect("insert_sql", finalGroup_sql);
                        }
                    });
                    sendgroupsql.start();
                    Log.d("測試","睡著三次了"+finalGroup_sql);
                }
            }

        }
    }

    public class SetNewAlarm extends TimerTask
    {
        public void run()
        {
            //新增一個鬧鐘
            Timer timer = new Timer(true);
            if(!Sleep_NotificationReceiver.user_response){

                Log.d("30秒到","設一個新鬧鐘!!");
                AlarmManager am = (AlarmManager) record.getSystemService(Context.ALARM_SERVICE);
                long triggertime = System.currentTimeMillis()+30000;
                Intent intent = new Intent(record, ai_alarmalert.class);
                intent.putExtra("requestcode",requestcode);
                intent.putExtra("timedate",timedate);
                PendingIntent op = PendingIntent.getActivity(record, 1, intent ,PendingIntent.FLAG_UPDATE_CURRENT);

                am.setExact(AlarmManager.RTC, triggertime, op);

            }
            timer.cancel();
        }
    };

    protected void sendNotification(String awake,String alarmtime){

        int notification_num = 2;


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {

            String channel_id = "TimeKeeper 智慧時間管家";
            NotificationChannel channelFriend = new NotificationChannel(
                    channel_id,
                    "AI判斷",
                    NotificationManager.IMPORTANCE_HIGH);
            channelFriend.setDescription("TimeKeeper 時間管家：判斷你是否起床");
            channelFriend.enableLights(true);
            channelFriend.enableVibration(true);


            NotificationManager manager = (NotificationManager)
                    record.getSystemService(Context.NOTIFICATION_SERVICE);
            manager.createNotificationChannel(channelFriend);
            if (awake.equals("0")){
                final Intent sleep_Intent = new Intent(record, Sleep_NotificationReceiver.class); // 通知的的Intent
                sleep_Intent.putExtra("alarmtime",alarmtime); // 傳入通知的識別號碼
                sleep_Intent.putExtra("cancel_notify_id", notification_num);
                int flags = PendingIntent.FLAG_ONE_SHOT; // ONE_SHOT：PendingIntent只使用一次；CANCEL_CURRENT：PendingIntent執行前會先結束掉之前的；NO_CREATE：沿用先前的PendingIntent，不建立新的PendingIntent；UPDATE_CURRENT：更新先前PendingIntent所帶的額外資料，並繼續沿用
                final PendingIntent pendingCancelIntent = PendingIntent.getBroadcast(record, 0, sleep_Intent, flags); // 取得PendingIntent

                final Notification.BigTextStyle bigTextStyle = new Notification.BigTextStyle(); // 建立BigTextStyle
                bigTextStyle.setBigContentTitle("貪睡提醒!!!!"); // 當BigTextStyle顯示時，用BigTextStyle的setBigContentTitle覆蓋setContentTitle的設定
                bigTextStyle.bigText("TimeKeeper認為您還在賴床，於稍等將再為您設一個鬧鐘！如果已經醒來的話，請按我已起床。"); // 設定BigTextStyle的文字內容

                Notification.Builder builder =
                        new Notification.Builder(record)
                                .setSmallIcon(R.drawable.ai_alarm_btn)
                                .setContentTitle("貪睡提醒!!!!")
                                .setContentText("TimeKeeper認為您還在賴床，於稍等將再為您設一個鬧鐘！如果已經醒來的話，請按我已起床。")
                                .addAction(R.drawable.ai_open,"我已起床",pendingCancelIntent)
                                .setStyle(bigTextStyle)
                                .setChannelId(channel_id);
                manager.notify(notification_num, builder.build());
                notification_num++;

            }else if(awake.equals("1")){

                final Intent awake_Intent = new Intent(record, Awake_NotificationReceiver.class); // 取消通知的的Intent
                awake_Intent.putExtra("alarmtime",alarmtime); // 傳入通知的識別號碼
                awake_Intent.putExtra("cancel_notify_id", notification_num);
                int flags = PendingIntent.FLAG_ONE_SHOT; // ONE_SHOT：PendingIntent只使用一次；CANCEL_CURRENT：PendingIntent執行前會先結束掉之前的；NO_CREATE：沿用先前的PendingIntent，不建立新的PendingIntent；UPDATE_CURRENT：更新先前PendingIntent所帶的額外資料，並繼續沿用
                final PendingIntent pendingCancelIntent = PendingIntent.getBroadcast(record, 0, awake_Intent, flags); // 取得PendingIntent


                final Notification.BigTextStyle bigTextStyle = new Notification.BigTextStyle(); // 建立BigTextStyle
                bigTextStyle.setBigContentTitle("恭喜起床!!"); // 當BigTextStyle顯示時，用BigTextStyle的setBigContentTitle覆蓋setContentTitle的設定
                bigTextStyle.bigText("TimeKeeper認為您剛剛已經醒來，如果其實貪睡的話，請按我賴床了。"); // 設定BigTextStyle的文字內容

                Notification.Builder builder =
                        new Notification.Builder(record)
                                .setSmallIcon(R.drawable.ai_alarm_btn)
                                .setContentTitle("恭喜起床!!")
                                .setContentText("TimeKeeper認為您剛剛已經醒來，如果其實貪睡的話，請按我賴床了。")
                                .addAction(R.drawable.ai_open,"我賴床了",pendingCancelIntent)
                                .setStyle(bigTextStyle)
                                .setChannelId(channel_id);
                manager.notify(notification_num, builder.build());
                notification_num++;
            }

        }else{
            NotificationCompat.Builder builder =
                    new NotificationCompat.Builder(record);
            // 準備設定通知效果用的變數
            int defaults = 0;
            // 加入震動效果
            defaults |= Notification.DEFAULT_VIBRATE;
            // 加入音效效果
            defaults |= Notification.DEFAULT_SOUND;
            // 加入閃燈效果
            defaults |= Notification.DEFAULT_LIGHTS;

            // 設定通知效果
            builder.setDefaults(defaults);
            // 設定小圖示、大圖示、狀態列文字、時間、內容標題、內容訊息和內容額外資訊
            if (awake.equals("0")){
                final Intent sleep_Intent = new Intent(record, Sleep_NotificationReceiver.class); // 通知的的Intent
                sleep_Intent.putExtra("alarmtime",alarmtime); // 傳入通知的識別號碼
                sleep_Intent.putExtra("cancel_notify_id", notification_num);
                int flags = PendingIntent.FLAG_ONE_SHOT; // ONE_SHOT：PendingIntent只使用一次；CANCEL_CURRENT：PendingIntent執行前會先結束掉之前的；NO_CREATE：沿用先前的PendingIntent，不建立新的PendingIntent；UPDATE_CURRENT：更新先前PendingIntent所帶的額外資料，並繼續沿用
                final PendingIntent pendingCancelIntent = PendingIntent.getBroadcast(record, 0, sleep_Intent, flags); // 取得PendingIntent

                final android.support.v4.app.NotificationCompat.BigTextStyle bigTextStyle = new android.support.v4.app.NotificationCompat.BigTextStyle(); // 建立BigTextStyle
                bigTextStyle.setBigContentTitle("貪睡提醒!!!!"); // 當BigTextStyle顯示時，用BigTextStyle的setBigContentTitle覆蓋setContentTitle的設定
                bigTextStyle.bigText("TimeKeeper認為您還在賴床，於稍等將再為您設一個鬧鐘！如果已經醒來的話，請按我已起床。"); // 設定BigTextStyle的文字內容

                builder.setSmallIcon(R.drawable.ai_alarm_btn)
                        .setWhen(System.currentTimeMillis())
                        .setContentTitle("貪睡提醒!!!!")
                        .setContentText("TimeKeeper認為您還在賴床，於稍等將再為您設一個鬧鐘！如果已經醒來的話，請按我已起床。")
                        .setContentInfo("3")
                        .addAction(R.drawable.ai_open,"我已起床",pendingCancelIntent)
                        .setStyle(bigTextStyle)
                        .setAutoCancel(true);
            }else if(awake.equals("1")){
                final Intent awake_Intent = new Intent(record, Awake_NotificationReceiver.class); // 取消通知的的Intent
                awake_Intent.putExtra("alarmtime",alarmtime); // 傳入通知的識別號碼
                awake_Intent.putExtra("cancel_notify_id", notification_num);
                int flags = PendingIntent.FLAG_ONE_SHOT; // ONE_SHOT：PendingIntent只使用一次；CANCEL_CURRENT：PendingIntent執行前會先結束掉之前的；NO_CREATE：沿用先前的PendingIntent，不建立新的PendingIntent；UPDATE_CURRENT：更新先前PendingIntent所帶的額外資料，並繼續沿用
                final PendingIntent pendingCancelIntent = PendingIntent.getBroadcast(record, 0, awake_Intent, flags); // 取得PendingIntent

                final android.support.v4.app.NotificationCompat.BigTextStyle bigTextStyle = new android.support.v4.app.NotificationCompat.BigTextStyle(); // 建立BigTextStyle
                bigTextStyle.setBigContentTitle("恭喜起床!!"); // 當BigTextStyle顯示時，用BigTextStyle的setBigContentTitle覆蓋setContentTitle的設定
                bigTextStyle.bigText("TimeKeeper認為您剛剛已經醒來，如果其實貪睡的話，請按我賴床了。"); // 設定BigTextStyle的文字內容


                builder.setSmallIcon(R.drawable.ai_alarm_btn)
                        .setWhen(System.currentTimeMillis())
                        .setContentTitle("恭喜起床!!")
                        .setContentText("TimeKeeper認為您剛剛已經醒來，如果其實貪睡的話，請按我賴床了。")
                        .setContentInfo("3")
                        .setStyle(bigTextStyle)
                        .addAction(R.drawable.ai_open,"我賴床了",pendingCancelIntent)
                        .setAutoCancel(true);
            }
            // 取得NotificationManager物件
            NotificationManager manager = (NotificationManager)
                    record.getSystemService(Context.NOTIFICATION_SERVICE);

            Notification notification = builder.build();
            manager.notify(notification_num, notification);
            notification_num++;
        }
        if(awake.equals("0")){
            Timer timer = new Timer(true);
            Calendar cal = Calendar.getInstance(Locale.ENGLISH);
            cal.setTimeInMillis(System.currentTimeMillis()+30000);
            Date date = cal.getTime();
            timer.schedule(new SetNewAlarm(), date);
            Log.d("時間：","應該是30S後"+date);
        }


    }



    public void CheckState_SubmitRecord(){
        Thread thread_check_state = new Thread(new Runnable() {
            @Override
            public void run() {
                while(state == "true"){
                    getScreen(ttdd);
                    Log.d("執行續", "檢查螢幕狀態");
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                if(state == "false"){
                    Log.d("state", "enter if");
                    Calendar calendar2 = Calendar.getInstance();
                    calendar2.setTimeInMillis(System.currentTimeMillis());
                    stopuse = calendar2.getTimeInMillis();
                    usagetime(stopuse);
                    dbUsage.insert(u_id,String.valueOf(ttdd),(int)sec);


                    Log.d("測試", "帳號:"+u_id);
                    Log.d("紀錄", "結束");
                    Log.d("TAG", "螢幕使用狀態上傳");
                }
                Cursor update_cursor = dbUsage.select_update();
                for(int i = 0; i<update_cursor.getCount();i++){
                    update_cursor.moveToPosition(i);
                    up_id_u.add(update_cursor.getString(update_cursor.getColumnIndex("id")));
                    up_date_time_u.add(update_cursor.getString(update_cursor.getColumnIndex("date")));
                    up_period_u.add((update_cursor.getString(update_cursor.getColumnIndex("period"))));
                }
                //sql_u = "INSERT INTO `screen_record` (`Date`, `User_id`, `Period`, `r_ifawake`) VALUES";
                for(int i = 0;i<up_id_u.size();i++){
                    String id = up_id_u.get(i).toString();
                    String time = up_date_time_u.get(i).toString();
                    int period  = Integer.parseInt(up_period_u.get(i).toString());
                    sql_u = "INSERT INTO `screen_record` (`Date`, `User_id`, `Period`) VALUES";
                    if(i == up_id_u.size()-1){
                        sql_u_tmp = sql_u_tmp+"('"+time+"','"+id+"', '"+period+"');";
                    }else {
                        sql_u_tmp = sql_u_tmp+"('"+time+"','"+id+"', '"+period+"'),";
                    }

                }
                sql_u = sql_u+sql_u_tmp;
                Log.d("sql語法",sql_u);
                connecting.connect("insert_sql",sql_u);
                try {
                    Thread.sleep(400);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                Log.d("測試","狀態"+connecting.internet_connect);
                if(connecting.internet_connect){
                    dbUsage.update_state_change();
                }
            }
        });
        thread_check_state.start();
    }


    public void startListenAudio() {
        final int Base = 1;
        Thread thread_sound = new Thread(new Runnable() {
            int count = 0;
            @Override
            public void run() {
                while (count < 121) {
                    try {
                        if (sound_recorder.mMediaRecorder != null) {
                            volume = sound_recorder.mMediaRecorder.getMaxAmplitude() / Base;  //獲取聲壓值
                            Today = new Date();
                            timestamp = new Timestamp(Today.getTime());
                            id.add(u_id);
                            name.add("管理員");
                            date_time.add(timestamp);
                            x_axis.add(String.format("%.02f", axis_recorder.x));
                            y_axis.add(String.format("%.02f", axis_recorder.y));
                            z_axis.add(String.format("%.02f", axis_recorder.z));
                            if (volume > 1) {
                                volume = 20 * Math.log10(volume);//將聲壓值轉為分貝值
                            }
                            sound_db.add(String.format("%.02f",volume));
                            Log.d("執行續",(count+1)+" data");
                            Log.d("執行續",volume+"分貝");
                            Log.d("執行續",axis_recorder.x+"X軸");
                            Log.d("執行續",axis_recorder.y+"Y軸");
                            Log.d("執行續",axis_recorder.z+"Z軸");
                            count++;
                        }
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                clock_count--;
                if(clock_count == 0){
                    axis_recorder.start_record(sensorManager,false);//九軸停止紀錄(但現在我註解掉了)
                    sound_recorder.stopRecord();//停止錄音
                }
//                axis_recorder.start_record(sensorManager,false);//九軸停止紀錄(但現在我註解掉了)
//                sound_recorder.stopRecord();//停止錄音
                for(int i = 0;i<id.size();i++){
                    dbSoundaxis.insert(id.get(i).toString(), (Timestamp) date_time.get(i), Double.parseDouble(x_axis.get(i).toString()),
                            Double.parseDouble(y_axis.get(i).toString()),Double.parseDouble(z_axis.get(i).toString()),
                            Double.parseDouble(sound_db.get(i).toString()), String.valueOf(ttdd));
                    Log.d("資料庫",id.get(i).toString()+"/"+(Timestamp) date_time.get(i)+"/"+Double.parseDouble(x_axis.get(i).toString())+"/"+Double.parseDouble(y_axis.get(i).toString())+"/"+Double.parseDouble(z_axis.get(i).toString())+"/"+Double.parseDouble(sound_db.get(i).toString()));
                }
                Log.d("紀錄", "結束");
                Log.d("TAG", "九軸&分貝資料上傳");
                CheckRecordtime_SubmitRecord();
            }
        });
        thread_sound.start();
    }


    private class ScreenBroadReceiver extends BroadcastReceiver {
        private String action = null;
        @Override
        public void onReceive(Context context, Intent intent) {
            action = intent.getAction();
            if (Intent.ACTION_SCREEN_ON.equals(action)|| Intent.ACTION_SCREEN_OFF.equals(action)){
                state = "true";
            }else if (Intent.ACTION_USER_PRESENT.equals(action)){
                state = "false";
            }
        }
    }

    public int asleeptimes(final String date){
        Thread get_sleep = new Thread(new Runnable() {
            @Override
            public void run() {
                connecting.connect("select_sql","SELECT asleep FROM `screen_record` WHERE  Date = '"+date+"'");
            }
        });
        get_sleep.start();
        try {
            get_sleep.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        int atimes = 0;
        try {
            JSONArray times = new JSONArray(connecting.get_data);
            int lenght = times.length();
            for(int i = 0;i < lenght;i++){
                JSONObject jsonObject = times.getJSONObject(i);
                atimes = jsonObject.getInt("asleep");
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }

        return atimes;
    }

    public void updateasleep(final int a, final String date){
        Thread updatesleep = new Thread(new Runnable() {
            @Override
            public void run() {
                Log.d("updatedate",""+date);
                Log.d("updatedate",""+a);
                connecting.connect("insert_sql","UPDATE `screen_record` SET `asleep` = '"+a+ "' WHERE `screen_record`.`Date` = '"+date+"' ;");
            }
        });
        updatesleep.start();
    }

    public String[] getfriend(int req){
        DB_normal_alarm db = new DB_normal_alarm(record);
        Cursor cursor = db.selectbycode(req);
        String[] ff = new String[0];
        if (cursor != null && cursor.moveToFirst()){
            String f = cursor.getString(9);
            Log.d("好晚啦啦啦啊啦","累"+f);

            if (f != null && !f.equals("")){
                ff = f.split("  ");
                for (int a = 0; a < ff.length; a++){
                    Log.d("呵呵呵呵呵呵呵呵呵呵呵呵呵幹",""+ff[a]);
                }
//                String[] arrays = f.trim().split("\\s+");
//                for(String s : arrays){
//                    Log.d("shite幹幹幹",""+s);
//                }
            }

        }
        return ff;
    }
}

