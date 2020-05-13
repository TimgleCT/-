package com.example.user.myapplication;

import android.util.Log;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class data_img_prediction {
    private int serverResponseCode = 0;
    //private ProgressDialog dialog = null;
    private String upLoadServerUri;
    public String AI_response;

    public int produce_img(final String alarm_time) {

        upLoadServerUri = "http://140.127.218.207:80/produce_data_img.php";
        HttpURLConnection conn = null;
        DataOutputStream dos = null;
        String lineEnd = "\r\n";
        String twoHyphens = "--";
        String boundary = "*****";
        //dialog = new ProgressDialog(context.getApplicationContext());

        try {

            // open a URL connection to the Servlet
            URL url = new URL(upLoadServerUri);

            // Open a HTTP  connection to  the URL
            conn = (HttpURLConnection) url.openConnection();
            conn.setDoInput(true); // Allow Inputs
            conn.setDoOutput(true); // Allow Outputs
            conn.setUseCaches(false); // Don't use a Cached Copy
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Connection", "Keep-Alive");
            conn.setRequestProperty("ENCTYPE", "multipart/form-data");
            conn.setRequestProperty("Content-Type", "multipart/form-data;boundary=" + boundary);

            dos = new DataOutputStream(conn.getOutputStream());

            dos.writeBytes(twoHyphens + boundary + lineEnd);


            //adding parameter

            //Send parameter #name
            dos.writeBytes("Content-Disposition: form-data; name='alarm_time'" + lineEnd);
            dos.writeBytes(lineEnd);
            dos.writeBytes(alarm_time + lineEnd);
            dos.writeBytes(twoHyphens + boundary + lineEnd);


            // Responses from the server (code and message)
            serverResponseCode = conn.getResponseCode();
            String serverResponseMessage = conn.getResponseMessage();

            Log.d("uploadFile", "HTTP Response is : "
                    + serverResponseMessage + ": " + serverResponseCode);
            if(serverResponseCode == 200) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"));
                String line;
                StringBuffer sb = new StringBuffer();

                while((line = reader.readLine()) != null) {
                    sb.append(line + "\n");
                }
                reader.close();
                Log.d("成功", "成功製圖");
                Log.d("成功", "主機回傳結果："+sb.toString());
                AI_response = sb.toString();
            }else{
                Log.d("失敗","失敗："+serverResponseCode);
            }
            //close the streams //
            dos.flush();
            dos.close();

        } catch (MalformedURLException ex) {

            //dialog.dismiss();
            ex.printStackTrace();
            Log.d("錯誤","MalformedURLException Exception : check script url.");
            Log.e("Upload file to server", "error: " + ex.getMessage(), ex);
        } catch (Exception e) {
            //dialog.dismiss();
            e.printStackTrace();
            Log.d("錯誤","看錯誤碼");
            Log.e("server Exception", "Exception : "  + e.getMessage(), e);
        }
        //dialog.dismiss();
        return serverResponseCode;

        // End else block
    }
}