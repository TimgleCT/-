package com.example.user.myapplication;

import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.ColorDrawable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.NumberPicker;
import android.widget.RelativeLayout;

import java.lang.reflect.Field;

public class counter extends AppCompatActivity{

    NumberPicker hour, min, sec;
    int m_hour, m_min, m_sec;
    Button start;
    long time;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.counter);

        // 小時
        hour = findViewById(R.id.hour);
        hour.setMinValue(0);
        hour.setValue(0);
        hour.setMaxValue(23);
        hour.setWrapSelectorWheel(true);

        hour.setDescendantFocusability(NumberPicker.FOCUS_BLOCK_DESCENDANTS);
//        setNumberPickerTextColor(hour, Color.BLUE);
        hour.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
                Log.d("old",":"+oldVal);
                Log.d("new",":"+newVal);

                m_hour = newVal*60*60;
            }
        });
        setNumberPickerDividerColor(hour);

        // 分鐘
        min = findViewById(R.id.min);
        min.setMinValue(0);
        min.setMaxValue(59);
        min.setValue(0);
        min.setWrapSelectorWheel(true);
        min.setDescendantFocusability(NumberPicker.FOCUS_BLOCK_DESCENDANTS);
        min.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
                m_min = newVal*60;
            }
        });
        setNumberPickerDividerColor(min);

        // 秒
        sec = findViewById(R.id.sec);
        sec.setValue(0);
        sec.setMinValue(0);
        sec.setMaxValue(59);
        sec.setWrapSelectorWheel(true);
        sec.setDescendantFocusability(NumberPicker.FOCUS_BLOCK_DESCENDANTS);
        sec.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
                m_sec = newVal;
            }
        });
        setNumberPickerDividerColor(sec);

        start = findViewById(R.id.start_btn);
        start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setTime();
                Intent intent = new Intent(counter.this, counter_countdown.class);
                intent.putExtra("time", time);
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intent);
                Log.d("時間時間時間",""+time);
            }
        });

    }
    public long setTime(){
        time = m_hour + m_min + m_sec;
        return time;
    }

    private void setNumberPickerDividerColor(NumberPicker numberPicker){
        Field[] pickerFields = NumberPicker.class.getDeclaredFields();
        for (Field pf : pickerFields) {
            if (pf.getName().equals("mSelectionDivider")) {
                pf.setAccessible(true);
                try {
                    //设置分割线的颜色值
                    pf.set(numberPicker, new ColorDrawable(getResources().getColor(R.color.gray)));
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
                break;
            }
        }
    }

    public static void setNumberPickerTextColor(NumberPicker numberPicker, int color){

        final int count = numberPicker.getChildCount();
        for(int i = 0; i < count; i++){
            View child = numberPicker.getChildAt(i);
            if(child instanceof EditText){
                try{
                    Field selectorWheelPaintField = numberPicker.getClass().getDeclaredField("mSelectorWheelPaint");
                    selectorWheelPaintField.setAccessible(true);
                    ((Paint)selectorWheelPaintField.get(numberPicker)).setColor(color);
                    numberPicker.invalidate();
                    ((EditText)child).setTextColor(color);
                    Log.d("ll","!!!!!!!!!!!");

                }catch(Exception e){
                    e.printStackTrace();
                }

            }
        }

        RelativeLayout llFirst = (RelativeLayout) numberPicker.getChildAt(0);
        RelativeLayout mSpinners = (RelativeLayout) llFirst.getChildAt(0);

//        for (int i = 0; i < mSpinners.getChildCount(); i++) {
//
//            NumberPicker picker = (NumberPicker) mSpinners.getChildAt(i);
//
//            Field[] pickerFields = NumberPicker.class.getDeclaredFields();
//            for (Field pf : pickerFields) {
//                pf.setAccessible(true);
//                String pfString = pf.getName();
//                try {
//                    if (pfString.equals("mSelectionDivider")) {
////                        pf.set(picker, new ColorDrawable(Color.parseColor("#ffffff")));//設置分割線顏色
////                        break;
//                    } else if (pfString.equals("mInputText")) {
//                        EditText mInputText = (EditText) pf.get(picker);//獲得該屬性對應的對象
//                        mInputText.setTextColor(Color.BLUE);
//                        break;
//                    } else if (pfString.equals("mSelectorWheelPaint")) {
//                        Paint mPaint = (Paint) pf.get(picker);//獲得該屬性對應的對象
//                        mPaint.setColor(Color.GREEN);
//                        break;
//                    } else if (pfString.equals("mVirtualButtonPressedDrawable")) {
//                        pf.set(picker, new ColorDrawable(Color.parseColor("#ffffff")));
//                        break;
//                    }
//                } catch (IllegalArgumentException e) {
//                    e.printStackTrace();
//                } catch (Resources.NotFoundException e) {
//                    e.printStackTrace();
//                } catch (IllegalAccessException e) {
//                    e.printStackTrace();
//                }
//            }
//        }
    }
}
