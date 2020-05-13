package com.example.user.myapplication.ai_alarm_manage;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.annotation.Nullable;
import android.support.v4.view.ViewPager;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.example.user.myapplication.R;

public class ai_manage_holiday extends Fragment {
    private View view;
    private ViewPager mViewPager;
    private CardFragmentPagerAdapter mFragmentCardAdapter;
    private ShadowTransformer mFragmentCardShadowTransformer;

    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        view = inflater.inflate(R.layout.fragment_ai_manage_holiday, container,false);

        mViewPager = view.findViewById(R.id.viewPager);

        mFragmentCardAdapter = new CardFragmentPagerAdapter(getChildFragmentManager(),
                dpToPixels(2, getActivity()));

        mFragmentCardShadowTransformer = new ShadowTransformer(mViewPager, mFragmentCardAdapter);
        mViewPager.setAdapter(mFragmentCardAdapter);
        mViewPager.setPageTransformer(false, mFragmentCardShadowTransformer);
        mViewPager.setOffscreenPageLimit(3);
        mFragmentCardShadowTransformer.enableScaling(true);

        return view;
    }

    public static float dpToPixels(int dp, Context context) {
        return dp * (context.getResources().getDisplayMetrics().density);
    }
}
