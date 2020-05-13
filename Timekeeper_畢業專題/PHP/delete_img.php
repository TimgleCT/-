<?php
    $description =$_POST['description'];
    $file_path = "uploads/";
    $file_path = $file_path . $description;
    if(file_exists($file_path)){
        unlink($file_path);//將檔案刪除
    }
?>