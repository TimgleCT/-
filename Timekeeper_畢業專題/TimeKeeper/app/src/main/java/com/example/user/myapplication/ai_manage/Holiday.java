package com.example.user.myapplication.ai_manage;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.Calendar;

import okhttp3.Call;
import okhttp3.Response;

public class Holiday {
    Boolean isbreak = false;
    Boolean istyphoon = false;
    String re_info;

    public boolean isholiday(){
        HttpUtil.sendRequestWithOkhttp("http://data.ntpc.gov.tw/api/v1/rest/datastore/382000000A-000077-002", new okhttp3.Callback() {
            @Override
            public void onFailure(Call call, IOException e) {

            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        String jobject = "";
                        try {
                            jobject = response.body().string();

                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        JSONObject jsonObject = null;
                        try {
                            jsonObject = new JSONObject(jobject);
                            JSONObject result = jsonObject.getJSONObject("result");
                            JSONArray array = result.getJSONArray("records");
                            for (int i = 0; i < array.length(); i++) {
                                JSONObject obj = array.getJSONObject(i);
                                Calendar calendar = Calendar.getInstance();
                                calendar.setTimeInMillis(System.currentTimeMillis());
                                int year = calendar.get(Calendar.YEAR);
                                int month = calendar.get(Calendar.MONTH) + 1;
                                int day = calendar.get(Calendar.DAY_OF_MONTH);
                                String today = year +"/"+month+"/"+day;
                                Log.d("today",":"+today);
                                if (obj.getString("date").equals(today) && !obj.getString("holidayCategory").equals("星期六、星期日")){
                                    if (obj.getString("isHoliday").equals("是")){
                                        isbreak = true;
                                        break;
                                    }
                                }
                            }
                            Log.d("break",":"+isbreak);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                    }
                }).start();
            }
        });

        return isbreak;
    }

    public String typhooninfo(){
        HttpUtil.sendRequestWithOkhttp("https://alerts.ncdr.nat.gov.tw/JSONAtomFeed.ashx?AlertType=33", new okhttp3.Callback() {
            @Override
            public void onFailure(Call call, IOException e) {

            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        String jobject = "";
                        try {
                            jobject = response.body().string();

                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        JSONObject jsonObject = null;
                        try {
                            jsonObject = new JSONObject(jobject);
                            JSONObject entry = jsonObject.getJSONObject("entry");
                            String updated = entry.getString("updated");
                            String[] u_date = updated.split("T");
                            String[] s_city = new String[0];
                            String[] c_date = new String[0];
                            String[] d_date = new String[0];
                            JSONObject summary = entry.getJSONObject("summary");
                            for (int i = 0; i < summary.length(); i++) {
                                String text = summary.getString("#text");
                                String[] text2 = text.split(":");
                                String[] s_context = text2[1].split("。");
                                s_city = text2[0].split("]");

                                if (strStr(s_context[0], "停止上班、停止上課") != -1){
                                    c_date = s_context[0].split("停止上班、停止上課");
                                } else if (strStr(s_context[0], "已達停止上班及上課標準") != -1){
                                    c_date = s_context[0].split("已達停止上班及上課標準");
                                } else {
                                    c_date = s_context[0].split("");
                                }
                                if (strStr(c_date[0], "晚") != -1){
                                    d_date = c_date[0].split("晚");
                                    c_date[0] = d_date[0];
                                }else if (strStr(c_date[0], "中") != -1){
                                    d_date = c_date[0].split("中");
                                    c_date[0] = d_date[0];
                                }else if (strStr(c_date[0], "早") != -1){
                                    d_date = c_date[0].split("早");
                                    c_date[0] = d_date[0];
                                }
                            }
                            re_info = u_date[0]+ " " +s_city[1]+ " " +c_date[0]+ " ";
                            Log.d("info",":"+re_info);

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                    }
                }).start();
            }
        });
        return re_info;
    }

    public int  strStr(String haystack,String needle) {
        int l1=haystack.length(),l2=needle.length();
        if(l1<l2) {
            return -1;
        }else if(l2==0) {
            return 0;
        }
        int threshold=l1-l2;
        for(int i=0; i<=threshold;i++) {
            if(haystack.substring(i,i+l2).equals(needle)) {
                return i;
            }
        }
        return -1;
    }

    public boolean iftyphoon(String work, String home){
        String info = typhooninfo();
        Log.d("info",":"+info);
        if (info != null){
            String[] in = info.split(" ");
            Calendar calendar = Calendar.getInstance();
            calendar.setTimeInMillis(System.currentTimeMillis());
            int year = calendar.get(Calendar.YEAR);
            int month = calendar.get(Calendar.MONTH) + 1;
            int day = calendar.get(Calendar.DAY_OF_MONTH);
            String today = year +"-"+month+"-"+day;
            String yesterday = year + "-" + month + "-" + (day-1);
            if (!in[2].equals("今天") && !in[2].equals("明天")){
                String[] m = in[0].split("-");
                String md = m[1]+"/"+m[2];
                if (in[2].equals(md)){
                    in[2] = "今天";
                }else {
                    in[2] = "明天";
                }
            }

            if (in[0].equals(today) && in[2].equals("今天")){
                if (work.equals(in[1]) || home.equals(in[1])){
                    return true;
                }
            }else if (in[0].equals(yesterday) && in[2].equals("明天")){
                if (work.equals(in[1]) || home.equals(in[1])){
                    return true;
                }
            }

        }

        return istyphoon;
    }
}
