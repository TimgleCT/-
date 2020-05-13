package com.example.user.myapplication.ai_alarm_manage;

import android.graphics.Color;
import android.graphics.PorterDuff;
import android.support.v4.view.ViewPager;
import android.support.v7.widget.CardView;
import android.util.Log;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;

import com.example.user.myapplication.R;


public class ShadowTransformer implements ViewPager.OnPageChangeListener, ViewPager.PageTransformer {

    private ViewPager mViewPager;
    private CardAdapter mAdapter;
    private float mLastOffset;
    private boolean mScalingEnabled;

    public ShadowTransformer(ViewPager viewPager, CardAdapter adapter) {
        mViewPager = viewPager;
        viewPager.addOnPageChangeListener(this);
        mAdapter = adapter;
    }

    public void enableScaling(boolean enable) {
        if (mScalingEnabled && !enable) {
            // shrink main card
            CardView currentCard = mAdapter.getCardViewAt(mViewPager.getCurrentItem());
            if (currentCard != null) {
                currentCard.animate().scaleY(1);
                currentCard.animate().scaleX(1);
            }
        } else if (!mScalingEnabled && enable) {
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
        // This might be null if a fragment is being used
        // and the views weren't created

        if (currentCard != null) {
            if (mScalingEnabled) {
                currentCard.setScaleX((float) (1 + 0.1 * (1 - realOffset)));
                currentCard.setScaleY((float) (1 + 0.1 * (1 - realOffset)));
            }
            currentCard.setCardElevation((baseElevation + baseElevation
                    * (CardAdapter.MAX_ELEVATION_FACTOR - 1) * (1 - realOffset)));

            CardView setcard1 = mAdapter.getCardViewAt(1);
            ImageView imageView = (ImageView) setcard1.findViewById(R.id.cView);
            imageView.setImageResource(R.drawable.holiday);
            TextView textView = (TextView) setcard1.findViewById(R.id.cText);
            textView.setText("若該日為國定假日為\n" +
                    "(係指勞基法第37條之規定)\n" +
                    "系統將自動幫您關閉當日鬧鐘\n" +
                    "讓您享有一個美好的早晨 : )");

            CardView setcard2 = mAdapter.getCardViewAt(0);
            ImageView imageView2 = setcard2.findViewById(R.id.cView);
            imageView2.setImageResource(R.drawable.typhoon);
            TextView textView1 = setcard2.findViewById(R.id.cText);
            textView1.setText("若您所工作的縣市\n" +
                    "發布颱風天停班停課\n" +
                    "系統將自動幫您關閉當日鬧鐘\n" +
                    "讓您享有一個美好的早晨 : )");
            final Switch switch1 = setcard2.findViewById(R.id.switch1);
            switch1.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
                @Override
                public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                    if (b) {
                        switch1.getThumbDrawable().setColorFilter(Color.rgb( 85, 145, 198), PorterDuff.Mode.MULTIPLY);
                        switch1.getTrackDrawable().setColorFilter(Color.argb( 100,255, 255, 255), PorterDuff.Mode.MULTIPLY);
                    }else {
                        switch1.getThumbDrawable().setColorFilter(Color.WHITE, PorterDuff.Mode.MULTIPLY);
                        switch1.getTrackDrawable().setColorFilter(Color.BLACK, PorterDuff.Mode.MULTIPLY);
                    }
                }
            });

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

}
