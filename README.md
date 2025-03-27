# 2025-03-15 检测试题（备份）

- 功能：多模态融合感知节点

    - 1.使用OpenCV将图像转换为cv::Mat，订阅ros2话题图像话题，调用yolov5，假设接口为detect_objects();将检测结果发布（边界框和类别）
    - 2.订阅sensor_msgs/msg/PointCloud2类型的点云话题，调用pcl库，进行体素滤波和离群点去除，对处理后的结果进行欧式聚类，生成3D边界框，<span style="color:red;">将视觉检测结果和点云聚类结果融合（如Iou匹配），发布融合后的结果</span>


- 结果：<span style="background: linear-gradient(to right,yellow, Aqua,DeepPink); -webkit-background-clip: text; color: transparent;">
    没有安装cuda，仅使用了cpu进行推理，完成了检测和聚类，但聚类结果没有和检测结果融合</span>（没有相机的内参和相机相对于激光雷达的外参）


## 环境
- Ubuntu 20.04，没有安装cuda，本机部署，没有使用docker，依赖`yolov5/requirements.txt`   

- 一些其他的依赖指令：
    - ONNX Runtime  

        sudo apt-get install libonnxruntime-dev （无法定位软件包下面手动下载）
        https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz  
        tar xvf onnxruntime-linux-x64-1.16.3.tgz
        

    - 设置了环境变量
        ```bash
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/erlang/study_project/kaoshi_3.15/onnxruntime-linux-x64-1.16.3/lib" >> ~/.bashrc  
        source ~/.bashrc
        ```
    
    - other.......


## 运行实现

使用yolov转为onnx模型   
 `python3 export.py --weights weights/yolov5s.pt --include onnx --simplify`

播放了bag包  

可根据需要更改话题检测参数等，启动launch即可  

使用rviz自带类型，选择"MarkerArray"来显示边界框  

(不是特别准确，可以调用autoware消息类型)

![Alt text](/img/img.png)
