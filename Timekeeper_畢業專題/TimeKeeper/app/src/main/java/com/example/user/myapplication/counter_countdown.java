package com.example.user.myapplication;

import android.content.Intent;
import android.os.CountDownTimer;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.util.concurrent.TimeUnit;

public class counter_countdown extends AppCompatActivity implements View.OnClickListener {
    static long time,time2;
    static private long timeCountInMilliSeconds;
    private long bartime;
    static long timeleft;

    private enum TimerStatus {
        INITIAL,
        STARTED,
        STOPPED
    }

    private TimerStatus timerStatus = TimerStatus.INITIAL;

    private ProgressBar progressBarCircle;
    private TextView textViewTime;
    private ImageView imageViewReset;
    private ImageView imageViewStartStop;
    private ImageView imageViewClose;
    private CountDownTimer countDownTimer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.counter_countdown);

        // method call to initialize the views

        initViews();
//        textViewTime.setText(hmsTimeFormatter(timeCountInMilliSeconds));
        // method call to initialize the listeners
        initListeners();
        if (time == 0){
            time = getIntent().getLongExtra("time", 0);
            timeCountInMilliSeconds = time * 1000;
        }


        Log.d("十時十時十時",";"+time);
        if (time == 0){
            Intent counter = new Intent(this, counter.class);
            startActivity(counter);
            initial();
        }else {
            if (timeleft != 0){
                time = timeleft;
            }
            setProgressBarValues();
            imageViewStartStop.setImageResource(R.drawable.counter_stop);
            timerStatus = TimerStatus.STARTED;
            startCountDownTimer();
        }


    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

    }

    private void initViews() {
        progressBarCircle = findViewById(R.id.progressBarCircle);
        textViewTime = findViewById(R.id.textViewTime);
        imageViewReset = findViewById(R.id.imageViewReset);
        imageViewStartStop = findViewById(R.id.imageViewStartStop);
        imageViewClose = findViewById(R.id.imageViewClose);
    }

    private void initListeners() {
        imageViewReset.setOnClickListener(this);
        imageViewStartStop.setOnClickListener(this);
        imageViewClose.setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.imageViewReset:
                reset();
                break;
            case R.id.imageViewStartStop:
                startStop();
                break;
            case R.id.imageViewClose:
                goback();
                break;
        }
    }

    private void initial(){
        if (time != 0){
            Log.d("????????","??????????");
            if (timerStatus == TimerStatus.INITIAL){
                setProgressBarValues();
                imageViewStartStop.setImageResource(R.drawable.counter_stop);
                timerStatus = TimerStatus.STARTED;
                startCountDownTimer();
            }
        }
    }

    private void startStop() {
        if (timerStatus == TimerStatus.STOPPED) {
            // call to initialize the timer values
            setTimerValues();
            // call to initialize the progress bar values
            setProgressBarValues();
            // changing play icon to stop icon
            imageViewStartStop.setImageResource(R.drawable.counter_stop);
            // changing the timer status to started
            timerStatus = TimerStatus.STARTED;
            // call to start the count down timer
            startCountDownTimer();
        } else {
            // changing stop icon to start icon
            imageViewStartStop.setImageResource(R.drawable.counter_start);
            // changing the timer status to stopped
            timerStatus = TimerStatus.STOPPED;
            stopCountDownTimer();
        }
    }

    private void reset() {
//        // call to initialize the timer values
//        setTimerValues();
//        // call to initialize the progress bar values
//        setProgressBarValues();
//        // changing play icon to stop icon
//        imageViewStartStop.setImageResource(R.drawable.counter_stop);
//        // changing the timer status to started
//        timerStatus = TimerStatus.STARTED;
//        // call to start the count down timer
//        stopCountDownTimer();
//        startCountDownTimer();
        stopCountDownTimer();
        timerStatus = TimerStatus.INITIAL;
        initial();
    }

    private void goback(){
        stopCountDownTimer();
        time = 0;
        Intent stop = new Intent(this, counter.class);
        stop.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        startActivity(stop);
    }

    private void setTimerValues() {

        if (time != 0){
            Log.d("string",";"+textViewTime.getText().toString());
            String[] split_time = textViewTime.getText().toString().split(":");
            Log.d("nnills",";@@@"+split_time[0]);
            long timemill = TimeUnit.HOURS.toSeconds(Long.parseLong(split_time[0]));
            Log.d("hour",":"+timemill);
            timemill += TimeUnit.MINUTES.toSeconds(Long.parseLong(split_time[1]));
            timemill += Long.parseLong(split_time[2]);
            time = timemill;
        }
//        if (time != 0) {
//            // fetching value from edit text and type cast to integer
//            time = Long.parseLong(textViewTime.getText().toString().trim());
//        } else {
//            // toast message to fill edit text
//            Toast.makeText(getApplicationContext(), getString(R.string.message_minutes), Toast.LENGTH_LONG).show();
//        }
        // assigning values after converting to milliseconds
        timeCountInMilliSeconds = time * 1000;
    }

    /**
     * method to start count down timer
     */
    private void startCountDownTimer() {

        countDownTimer = new CountDownTimer(timeCountInMilliSeconds, 1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                timeleft = millisUntilFinished;
                textViewTime.setText(hmsTimeFormatter(millisUntilFinished));
                progressBarCircle.setProgress((int) (millisUntilFinished / 1000));
            }

            @Override
            // 倒數計時完成
            public void onFinish() {
                textViewTime.setText(hmsTimeFormatter(timeCountInMilliSeconds));
                // call to initialize the progress bar values
                setProgressBarValues();
                // hiding the reset icon
//                imageViewReset.setVisibility(View.GONE);
                // changing stop icon to start icon
                imageViewStartStop.setImageResource(R.drawable.counter_start);
                // changing the timer status to stopped
                timerStatus = TimerStatus.STOPPED;

                Toast.makeText(counter_countdown.this, " Time's up! ", Toast.LENGTH_SHORT).show();
            }

        }.start();
        countDownTimer.start();
    }



    /**
     * method to stop count down timer
     */
    private void stopCountDownTimer() {
        countDownTimer.cancel();
    }

    /**
     * method to set circular progress bar values
     */
    private void setProgressBarValues() {
        progressBarCircle.setMax((int) timeCountInMilliSeconds / 1000);
        progressBarCircle.setProgress((int) (timeCountInMilliSeconds / 1000));

    }


    /**
     * method to convert millisecond to time format
     *
     * @param milliSeconds
     * @return HH:mm:ss time formatted string
     */
    private String hmsTimeFormatter(long milliSeconds) {

        String hms = String.format("%02d:%02d:%02d",
                TimeUnit.MILLISECONDS.toHours(milliSeconds),
                TimeUnit.MILLISECONDS.toMinutes(milliSeconds) - TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(milliSeconds)),
                TimeUnit.MILLISECONDS.toSeconds(milliSeconds) - TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(milliSeconds)));

        return hms;
    }

//    private long StringToLong(TextView textView){
//        SimpleDateFormat format = new SimpleDateFormat("HH:mm:ss");
//        String string = textView.getText().toString();
//        Date date = format.parseObject(string);
//        time2 = date.getTime();
//        return time2;
//    }

}
