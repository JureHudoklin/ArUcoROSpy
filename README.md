# ArUcoDetect
Detect ArUco Marker and visualize its pose using IntelRealsense or Azure Kinect camera and python

##Detecting 36 ArUco Markers at once
[![36arucomarkers](https://user-images.githubusercontent.com/69029439/145345616-5071700a-db01-4aee-aae0-50d06c755227.png)](https://www.youtube.com/watch?v=2GhBSx3AGbs)

### [What is ArUco Marker?](https://www.pyimagesearch.com/2020/11/02/apriltag-with-python/)

### Architecture

![MarkerDetectionSchematic](https://user-images.githubusercontent.com/69029439/145345568-1a678c56-5e92-44ff-8517-d279841cc42a.jpg)
 

### Prerequisites (Installations and Dependencies)

- Intel Realsense
    
    [**ROS Releases**](https://github.com/IntelRealSense/realsense-ros)
    
    [**Realsense SDK - for realsense-viewer**](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
    
    
- Azure Kinect
    
    [**Azure Kinect ROS driver**](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#Installation)
    
      
[opencv-contrib-python] cv2 does not have an attribute "aruco"
        
[Python 3 cv2 Has No Attribute Aruco](https://winkdoubleguns.com/2021/02/13/python-3-cv2-has-no-attribute-aruco/)
        
[cv_bridge](https://wiki.ros.org/cv_bridge)
        
        

### Dependencies info

- [Intel Realsense ROS](http://wiki.ros.org/realsense_camera)
    
- [Azure Kinect ROS Topic](https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/docs/usage.md)
    
- [Opencv ArUco](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa)
- 
- [cv_bridge](https://wiki.ros.org/cv_bridge)

        
### Running the Code

## 1. Run the camera code in ROS
    
- Intel Realsense ROS
    
    **Run the built-in Intel Realsense program to see the camera's current state**
    
    ```bash
    realsense-viewer
    ```
    
    **Run the launch files**
    
    ```bash
    roslaunch realsense2_camera rs_camera.launch
    ```
    
- Azure Kinect ROS
    
    **Run the built-in Azure Kinect program to see the camera's current state**
    
    ```bash
    k4aviewer
    ```
    
    **Run the launch files**
    
    ```bash
    roslaunch azure_kinect_ros_driver kinect_rgbd.launch
    ```

## 2. Run ArucoDetect.py

    ```bash
    python /path/to/your/python/script/arucodetect.py
    ```
