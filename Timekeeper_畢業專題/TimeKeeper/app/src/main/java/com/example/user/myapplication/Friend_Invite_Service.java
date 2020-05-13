package com.example.user.myapplication;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.IBinder;
import android.support.v4.app.NotificationCompat;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import static com.example.user.myapplication.mainpage.KEY;

public class Friend_Invite_Service extends Service{
    public Friend_Invite_Service() {
    }
    Connect_To_Server search_friend_invitation = new Connect_To_Server();
    JSONArray get_result ;
    Boolean run = true ;
    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        Log.i("brad", "Service:onBind()");
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();

    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        final String my_id = intent.getStringExtra("my_id");
        Thread check_invitation_list = new Thread(new Runnable() {
            @Override
            public void run() {
                while(run){
                    Log.d("查看好友邀請","!!!!");
                    search_friend_invitation.connect("select_sql","SELECT user.u_name ,user.user_id FROM `user` WHERE user.user_id = any(SELECT user_friends_invitation.user_id FROM `user_friends_invitation` WHERE user_friends_invitation.friend_id ='"+my_id+"' AND user_friends_invitation.status =0)");
                    try {
                        Thread.sleep(400);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    String send_id = null,send_name = null;
                    try{
                        get_result = new JSONArray(search_friend_invitation.get_data);
                        int lenght = get_result.length();
                        for(int i = 0;i < lenght;i++){
                            JSONObject jsonObject = get_result.getJSONObject(i);
                            send_id = jsonObject.getString("user_id");
                            send_name = jsonObject.getString("u_name");
                        }
                        if(lenght!=0){
                            sendNotification(send_name);
                            search_friend_invitation.connect("insert_sql","UPDATE `user_friends_invitation` SET `status` = '1' WHERE `user_friends_invitation`.`user_id` = '"+ send_id +"' AND `user_friends_invitation`.`friend_id` = '"+my_id+"'");
                        }
                    }
                    catch(JSONException e) {
                        e.printStackTrace();
                    }
                    try {
                        Thread.sleep(30000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        check_invitation_list.start();

        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

    }
    protected void sendNotification(String f_name){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            int notification_num = 1;
            String channel_id = "好友邀請通知";
            NotificationChannel channelFriend = new NotificationChannel(
                    channel_id,
                    "好友邀請",
                    NotificationManager.IMPORTANCE_HIGH);
            channelFriend.setDescription("TimeKeeper 時間管家：你會不定時收到別人的好友邀請!");
            channelFriend.enableLights(true);
            channelFriend.enableVibration(true);


            NotificationManager manager = (NotificationManager)
                    getSystemService(Context.NOTIFICATION_SERVICE);
            manager.createNotificationChannel(channelFriend);
            Notification.Builder builder =
                    new Notification.Builder(this, "default")
                            .setSmallIcon(R.drawable.ai_alarm_btn)
                            .setContentTitle("收到一封好友邀請!")
                            .setContentText(f_name+"想要和你成為好友呦")
                            .setChannelId(channel_id);
            Intent intent =new Intent (this,setting_friend_search.class);
            PendingIntent pi = PendingIntent.getActivities(this, 0, new Intent[]{intent}, PendingIntent.FLAG_CANCEL_CURRENT);
            builder.setContentIntent(pi);
            manager.notify(notification_num, builder.build());
            notification_num++;

        }else{
            int notification_num = 1;
            Bitmap largeIcon = BitmapFactory.decodeResource(
                    getResources(), R.drawable.ai_alarm_btn);
            NotificationCompat.Builder builder =
                    new NotificationCompat.Builder(this);
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
            builder.setSmallIcon(R.drawable.ai_alarm_btn)
                    .setWhen(System.currentTimeMillis())
                    .setContentTitle("收到一封好友邀請!")
                    .setContentText(f_name+"想要和你成為好友呦")
                    .setContentInfo("3")
                    .setAutoCancel(true);
            // 取得NotificationManager物件
            NotificationManager manager = (NotificationManager)
                    getSystemService(Context.NOTIFICATION_SERVICE);
            Intent intent =new Intent (this,setting_friend_search.class);
            PendingIntent pi = PendingIntent.getActivities(this, 0, new Intent[]{intent}, PendingIntent.FLAG_CANCEL_CURRENT);
            builder.setContentIntent(pi);

            Notification notification = builder.build();
            manager.notify(notification_num, notification);
            notification_num++;
        }

    }
}
