package com.example.user.myapplication;

import android.app.Activity;
import android.app.AlarmManager;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.example.user.myapplication.ai_manage.Holiday;

import java.io.IOException;
import java.util.Calendar;

public class ai_alarmalert extends Activity {
    int requestcode;
    MediaPlayer mp;
    String musicpath;
    Button btn_ok;
    Handler h = new Handler();
    AlertDialog dialog;
    private MyReceiver receiver;
    int state = 1;
    long timedate;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        DB_normal_alarm db = new DB_normal_alarm(this);
        Intent intent = getIntent();
        requestcode = intent.getIntExtra("requestcode", 0);
        timedate = intent.getLongExtra("timedate",0);
        Log.d("request",":"+requestcode);
        Log.d("timedate",""+timedate);
        AudioManager audioManager =(AudioManager)this.getSystemService(Context.AUDIO_SERVICE);
        // Set the volume of played media to your choice.
        audioManager.setStreamVolume (AudioManager.STREAM_MUSIC,10,0);
        if (requestcode == 0){
        }

        Cursor cursor = db.selectbycode(requestcode);
        String[] g_div = new String[0];
        if (cursor != null && cursor.moveToFirst()){
            musicpath = cursor.getString(1);
            state = cursor.getInt(8);
            String group = cursor.getString(10);
            if(group != null){
                Log.d("group",":"+group);
                g_div = group.split(" ");
                Holiday holiday = new Holiday();
                if(!holiday.isholiday() || holiday.iftyphoon(g_div[1], g_div[0])){
                    finish();
                }
            }
        }

        receiver = new MyReceiver();
        IntentFilter homeFilter = new IntentFilter(Intent.ACTION_CLOSE_SYSTEM_DIALOGS);

        registerReceiver(receiver, homeFilter);

        Window win = getWindow();
        win.addFlags(WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD | WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED);
        win.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON | WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);

        detectrepeat(requestcode, cursor);
        db.updatestate(requestcode, state);
        db.close();

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d("destory","!!");
        if (receiver != null){
            unregisterReceiver(receiver);
        }
        if (mp!=null){
            if (mp.isPlaying()){
                mp.stop();
            }
            mp.release();
        }
        if (dialog != null && dialog.isShowing()){
            Log.d("destory",":set");
            dialog.dismiss();
            dialog.cancel();
        }
        Intent service = new Intent(this, BootService.class);
        service.putExtra("req",requestcode);
        startService(service);
    }

    public void send(){
        Intent intent3 = new Intent(this, AlertService.class);
        this.startService(intent3);
    }

//    public void alarmDialog(){
//        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
//        LayoutInflater inflater = this.getLayoutInflater();
//
//        builder.setView(inflater.inflate(R.layout.dialog_ai, null));
//        btn_ok = findViewById(R.id.btn_ok);

//        builder.setPositiveButton("LATER", new DialogInterface.OnClickListener() {
//            @Override
//            public void onClick(DialogInterface dialog, int which) {
//                alarm();
//                h.removeCallbacksAndMessages(null);
//                finish();
//            }
//        });
//        builder.setNegativeButton("OK", new DialogInterface.OnClickListener() {
//            @Override
//            public void onClick(DialogInterface dialog, int which) {
//                mp.stop();
//                h.removeCallbacksAndMessages(null);
//                Calendar cd = Calendar.getInstance();
//                cd.setTimeInMillis(System.currentTimeMillis());
//                long time = cd.getTimeInMillis();
//                Log.d("alert", "time"+time);
//                Log.d("嗨嗨嗨嗨嗨嗨嗨", "時間時間~~~~~"+time);
//                ai_count.clock_count++;//1017
//                ai_count.requestcode = requestcode;
//                if (timedate != 0){
//                }else{
//                    ai_count.timedate = timedate;
//                }
//                Log.d("tag", "get"+time);
//                send();
//                finish();
//            }
//        });
//
//        dialog = builder.show();
//        builder.create();
//        dialog.setCancelable(false);
//        dialog.setCanceledOnTouchOutside(false);
//    }

    public void alarmDialog(){
        Dialog dialog = new Dialog(this,R.style.mDialog_ai);
        dialog.setContentView(R.layout.dialog_ai);

        dialog.show();
        dialog.setCancelable(false);
        dialog.setCanceledOnTouchOutside(false);

        TextView ok = dialog.findViewById(R.id.ok);
        ok.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mp.stop();
                h.removeCallbacksAndMessages(null);
                Calendar cd = Calendar.getInstance();
                cd.setTimeInMillis(System.currentTimeMillis());
                long time = cd.getTimeInMillis();
                Log.d("alert", "time"+time);
                Log.d("嗨嗨嗨嗨嗨嗨嗨", "時間時間~~~~~"+time);
                ai_count.clock_count++;//1017
                ai_count.requestcode = requestcode;
                if (timedate != 0){
                }else{
                    ai_count.timedate = timedate;
                }
                Log.d("tag", "get"+time);
                send();
                finish();
            }
        });
    }

    private void alarm(){
        AlarmManager am = (AlarmManager) getSystemService(Context.ALARM_SERVICE);

        Log.d("code",":"+requestcode);
        long triggertime = System.currentTimeMillis()+300000;
        Intent intent = new Intent(this, ai_alarmalert.class);
        intent.putExtra("requestcode", requestcode);
        PendingIntent op = PendingIntent.getActivity(this, 1, intent ,PendingIntent.FLAG_UPDATE_CURRENT);

        am.setExact(AlarmManager.RTC, triggertime, op);
    }

    public void oneminute(){
        Runnable stopPlaybackRun = new Runnable() {
            public void run(){
                try {
                    alarm();
                    mp.stop();
                    System.exit(0);
                }catch(Exception e) {
                    e.printStackTrace();
                }
            }
        };
        h.postDelayed(stopPlaybackRun, 60 * 1000);
    }

    public void detectrepeat(int requestcode, Cursor cursor) {
        ring(musicpath);
        state = 0;
    }

    public void ring(String musicpath){
        mp = new MediaPlayer();
        if (musicpath == null){
            mp = MediaPlayer.create(this, R.raw.test);
            mp.setLooping(true);
            mp.setVolume(1.0f, 1.0f);
            mp.start();
            oneminute();
            alarmDialog();
        }else {
            try {
                mp.setDataSource(musicpath);
                mp.prepare();
                mp.setVolume(1.0f, 1.0f);
                mp.start();
                oneminute();
                alarmDialog();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private class MyReceiver extends BroadcastReceiver {

        private final String SYSTEM_DIALOG_REASON_KEY = "reason";
        private final String SYSTEM_DIALOG_REASON_HOME_KEY = "homekey";
        private final String SYSTEM_DIALOG_REASON_RECENT_APPS = "recentapps";

        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();

            if (action.equals(Intent.ACTION_CLOSE_SYSTEM_DIALOGS)) {
                String reason = intent.getStringExtra(SYSTEM_DIALOG_REASON_KEY);

                if (reason == null)
                    return;

                // Home键
                if (reason.equals(SYSTEM_DIALOG_REASON_HOME_KEY)) {
                    Toast.makeText(getApplicationContext(), "關閉鬧鐘", Toast.LENGTH_SHORT).show();
                    mp.stop();
                    h.removeCallbacksAndMessages(null);
                    Calendar cd = Calendar.getInstance();
                    cd.setTimeInMillis(System.currentTimeMillis());
                    long time = cd.getTimeInMillis();
                    Log.d("alert", "time"+time);
                    send();
                    finish();
                }

                // 最近任务列表键
                if (reason.equals(SYSTEM_DIALOG_REASON_RECENT_APPS)) {

                }
            }
        }
    }

}
