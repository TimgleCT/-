<?php
   $alarm_time = $_POST['alarm_time'];
   if(isset($_POST['alarm_time'])) {
        $alarm_time = $_POST['alarm_time'];
	$ExecPath = "/usr/bin/python /var/www/html/timekeeper_single_data_img.py ".$alarm_time;
	$ExecPath2 = "/usr/bin/python /var/www/html/timekeeper_matric_predict.py ".$alarm_time." 2>&1";
	exec($ExecPath);
	$output = exec($ExecPath2);
	print_r($output);

   } else{
       echo "fail";
   }
?>