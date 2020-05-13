package com.example.user.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class get_u_sticker {
    public static Bitmap get_sticker(String src){
        try{
            URL url = new URL(src);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.connect();

            InputStream input = conn.getInputStream();
                Bitmap mbitmap = BitmapFactory.decodeStream(input);
                return mbitmap;
        }catch (IOException e) {
            e.printStackTrace();
            return  null;
        }
        //return  null;
    }
}
