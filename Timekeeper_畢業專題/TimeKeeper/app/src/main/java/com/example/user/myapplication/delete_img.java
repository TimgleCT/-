package com.example.user.myapplication;

import android.util.Log;

import java.io.DataOutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class delete_img {
    private int serverResponseCode = 0;
    //private ProgressDialog dialog = null;
    private String upLoadServerUri;

    public int deleteFile(final String user_id) {

        upLoadServerUri = "http://140.127.218.207:80/delete_img.php";
        HttpURLConnection conn = null;
        DataOutputStream dos = null;
        String lineEnd = "\r\n";
        String twoHyphens = "--";
        String boundary = "*****";
        int bytesRead, bytesAvailable, bufferSize;
        byte[] buffer;
        int maxBufferSize = 1 * 1024 * 1024;
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
            final String description =user_id+".png";
            //Send parameter #name
            dos.writeBytes("Content-Disposition: form-data; name='description'" + lineEnd);
            dos.writeBytes(lineEnd);
            dos.writeBytes(description + lineEnd);
            dos.writeBytes(twoHyphens + boundary + lineEnd);


            // Responses from the server (code and message)
            serverResponseCode = conn.getResponseCode();
            String serverResponseMessage = conn.getResponseMessage();

            Log.i("uploadFile", "HTTP Response is : "
                    + serverResponseMessage + ": " + serverResponseCode);
            if(serverResponseCode == 200) {
                Log.d("成功", "照片刪除");
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