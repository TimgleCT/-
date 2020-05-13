<?php
    include("include.php");
    if(isset($_POST['select_sql'])){
        $sql = $_POST['select_sql'];
        $get_all_result = mysqli_query($link,$sql);
		while($row = mysqli_fetch_assoc($get_all_result)){
			$output[] = $row;
        	    	//echo $row["user_id"]."/";
			//echo $row["u_password"]."/";
			//echo $row["u_name"];            
		}
		print(json_encode($output));        
    }else if(isset($_POST['insert_sql'])){
        $sql = $_POST['insert_sql'];
        $insert_data = mysqli_query($link,$sql);
        echo "我的SQL:".$sql; 
        	echo "上傳完成";
    }
    
    mysqli_close($link);
?>