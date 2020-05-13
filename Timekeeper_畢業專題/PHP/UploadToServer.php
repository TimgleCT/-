<?php
    $description =$_POST['description'];
   $file_path = "uploads/";
   $file_path = $file_path . $description;
   //$file_path = $file_path . basename( $_FILES['uploaded_file']['name']);
   if(move_uploaded_file($_FILES['uploaded_file']['tmp_name'], $file_path)) {
       echo "success";
	exec("/usr/bin/python /var/www/html/change_u_sticker.py");
   } else{
       echo "fail";
   }
?>