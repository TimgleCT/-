package com.example.user.myapplication.ai_manage;

import android.graphics.Color;
import android.graphics.PorterDuff;
import android.support.v4.view.ViewPager;
import android.support.v7.widget.CardView;
import android.text.Layout;
import android.util.Log;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.Switch;

import com.example.user.myapplication.R;


public class ShadowTransformer implements ViewPager.OnPageChangeListener, ViewPager.PageTransformer {

    private ViewPager mViewPager;
    private CardAdapter mAdapter;
    private float mLastOffset;
    private boolean mScalingEnabled;
    int[] switchb = new int[2];
    String status_1 ;
    String status_2 ;


    public ShadowTransformer(ViewPager viewPager, CardAdapter adapter,String status_1,String status_2) {
        mViewPager = viewPager;
        viewPager.addOnPageChangeListener(this);
        mAdapter = adapter;
        this.status_2 = status_2;
        this.status_1 = status_1;

    }

    public void enableScaling(boolean enable) {
        if (mScalingEnabled && !enable) {
            // shrink main card
            CardView currentCard = mAdapter.getCardViewAt(mViewPager.getCurrentItem());
            if (currentCard != null) {
                currentCard.animate().scaleY(1);
                currentCard.animate().scaleX(1);
            }
        }else if(!mScalingEnabled && enable){
            // grow main card
            CardView currentCard = mAdapter.getCardViewAt(mViewPager.getCurrentItem());
            if (currentCard != null) {
                currentCard.animate().scaleY(1.1f);
                currentCard.animate().scaleX(1.1f);
            }
        }

        mScalingEnabled = enable;
    }

    @Override
    public void transformPage(View page, float position) {

    }

    @Override
    public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {
        int realCurrentPosition;
        int nextPosition;
        float baseElevation = mAdapter.getBaseElevation();
        float realOffset;
        boolean goingLeft = mLastOffset > positionOffset;

        // If we're going backwards, onPageScrolled receives the last position
        // instead of the current one
        if (goingLeft) {
            realCurrentPosition = position + 1;
            nextPosition = position;
            realOffset = 1 - positionOffset;
        } else {
            nextPosition = position + 1;
            realCurrentPosition = position;
            realOffset = positionOffset;
        }

        // Avoid crash on overscroll
        if (nextPosition > mAdapter.getCount() - 1
                || realCurrentPosition > mAdapter.getCount() - 1) {
            return;
        }

        CardView currentCard = mAdapter.getCardViewAt(realCurrentPosition);

        //修改switch的顏色
        final Switch switch1 = mAdapter.getCardViewAt(0).findViewById(R.id.switch1);
        if (status_1.equals("1")){
            switch1.setChecked(true);
            switchb[0] = 1;
            switch1.getThumbDrawable().setColorFilter(Color.rgb( 33, 130, 185), PorterDuff.Mode.MULTIPLY);
            switch1.getTrackDrawable().setColorFilter(Color.argb( 100,255, 255, 255), PorterDuff.Mode.MULTIPLY);
        }else {
            switchb[0] = 0;
            switch1.getThumbDrawable().setColorFilter(Color.WHITE, PorterDuff.Mode.MULTIPLY);
            switch1.getTrackDrawable().setColorFilter(Color.BLACK, PorterDuff.Mode.MULTIPLY);
        }
        switch1.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                if (b) {
                    switch1.getThumbDrawable().setColorFilter(Color.rgb( 33, 130, 185), PorterDuff.Mode.MULTIPLY);
                    switch1.getTrackDrawable().setColorFilter(Color.argb( 100,255, 255, 255), PorterDuff.Mode.MULTIPLY);
                    switchb[0] = 1;
                    status_1 = "1";
                }else {
                    switch1.getThumbDrawable().setColorFilter(Color.WHITE, PorterDuff.Mode.MULTIPLY);
                    switch1.getTrackDrawable().setColorFilter(Color.BLACK, PorterDuff.Mode.MULTIPLY);
                    switchb[0] = 0;
                    status_1 = "0";
                }
                sbar();
            }
        });

        final Switch switch2 = mAdapter.getCardViewAt(1).findViewById(R.id.switch1);
        Log.d("傳值","狀態"+status_2);
        if (status_2.equals("1")){
            switch2.setChecked(true);
            switchb[1] = 1;
            switch2.getThumbDrawable().setColorFilter(Color.rgb( 255, 64, 129), PorterDuff.Mode.MULTIPLY);
            switch2.getTrackDrawable().setColorFilter(Color.argb( 100,255, 255, 255), PorterDuff.Mode.MULTIPLY);
        }else {
            switchb[1] = 0;
            switch2.getThumbDrawable().setColorFilter(Color.WHITE, PorterDuff.Mode.MULTIPLY);
            switch2.getTrackDrawable().setColorFilter(Color.BLACK, PorterDuff.Mode.MULTIPLY);
        }
        //sbar();
        switch2.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                if (b) {
                    switch2.getThumbDrawable().setColorFilter(Color.rgb( 255, 64, 129), PorterDuff.Mode.MULTIPLY);
                    switch2.getTrackDrawable().setColorFilter(Color.argb( 100,255, 255, 255), PorterDuff.Mode.MULTIPLY);
                    switchb[1] = 1;
                    status_2 = "1";
                }else {
                    switch2.getThumbDrawable().setColorFilter(Color.WHITE, PorterDuff.Mode.MULTIPLY);
                    switch2.getTrackDrawable().setColorFilter(Color.BLACK, PorterDuff.Mode.MULTIPLY);
                    switchb[1] = 0;
                    status_2 = "0";
                }
                sbar();
            }
        });


        // This might be null if a fragment is being used
        // and the views weren't created yet
        if (currentCard != null) {
            if (mScalingEnabled) {
                currentCard.setScaleX((float) (1 + 0.1 * (1 - realOffset)));
                currentCard.setScaleY((float) (1 + 0.1 * (1 - realOffset)));
            }
            currentCard.setCardElevation((baseElevation + baseElevation
                    * (CardAdapter.MAX_ELEVATION_FACTOR - 1) * (1 - realOffset)));

        }

        CardView nextCard = mAdapter.getCardViewAt(nextPosition);

        // We might be scrolling fast enough so that the next (or previous) card
        // was already destroyed or a fragment might not have been created yet
        if (nextCard != null) {
            if (mScalingEnabled) {
                nextCard.setScaleX((float) (1 + 0.1 * (realOffset)));
                nextCard.setScaleY((float) (1 + 0.1 * (realOffset)));
            }
            nextCard.setCardElevation((baseElevation + baseElevation
                    * (CardAdapter.MAX_ELEVATION_FACTOR - 1) * (realOffset)));
        }

        mLastOffset = positionOffset;
    }

    @Override
    public void onPageSelected(int position) {

    }

    @Override
    public void onPageScrollStateChanged(int state) {

    }

    public int[] sbar(){
        return switchb;
    }

    public void setcheck(String a, String b){
        this.status_1 = a;
        this.status_2 = b;
//        final Switch switch1 = mAdapter.getCardViewAt(0).findViewById(R.id.switch1);
//        final Switch switch2 = mAdapter.getCardViewAt(1).findViewById(R.id.switch1);
//        if (a.equals("1")){
//            switch1.setChecked(true);
//            switchb[0] = 1;
//        }else {
//            switchb[0] = 0;
//        }
//        if (b.equals("1")){
//            switch2.setChecked(true);
//            switchb[1] = 1;
//        }else {
//            switchb[1] = 0;
//        }
//        sbar();
    }
}
