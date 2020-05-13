<?php
header("content-type:text/html;charset=utf8");
$link = mysqli_connect('localhost','root','keeper','TimeKeeper')or die("無法連線資料庫");
mysqli_set_charset($link,"utf8");
//echo "靠";
?>