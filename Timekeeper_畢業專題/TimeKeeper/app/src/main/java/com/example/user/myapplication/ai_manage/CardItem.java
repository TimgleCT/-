package com.example.user.myapplication.ai_manage;


public class CardItem {

    private String mTextResource;
    private int mTitleResource;
    private int mImageView;

    public CardItem(String text, int image) {
        mTextResource = text;
        mImageView = image;
    }

    public String getText() {
        return mTextResource;
    }

    public int getTitle() {
        return mTitleResource;
    }

    public int getImage(){ return mImageView;}
}
