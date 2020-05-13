package com.example.user.myapplication;

import android.app.NotificationManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

public class Sleep_NotificationReceiver extends BroadcastReceiver {
    public static boolean user_response = false;
    Connect_To_Server connecting = new Connect_To_Server();
    @Override
    public void onReceive(final Context context, final Intent intent) {
        user_response = true;
        final int notifyID = intent.getIntExtra("cancel_notify_id", 0);
        final NotificationManager notificationManager = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE); // 取得系統的通知服務
        notificationManager.cancel(notifyID);

        final String alarmtime = intent.getStringExtra("alarmtime");
        Log.d("通知傳值","成功"+alarmtime);
        Thread update_awake_1 = new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    connecting.connect("insert_sql","UPDATE `screen_record` SET `r_ifawake` = '1' WHERE `screen_record`.`Date` = '"+alarmtime+"';");
                                }
                            });
                            update_awake_1.start();
    }
}
