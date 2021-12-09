# ArUcoDetect
Detect ArUco Marker and visualize its pose using IntelRealsense or Azure Kinect camera and python

[Detecting 2 ArUco Markers](https://youtu.be/kOELEBrRaDs)
[Detecting 36 ArUco Markers at once](https://www.youtube.com/watch?v=2GhBSx3AGbs)

### Motivation

When the manipulator serves the food to the serving robot, the manipulator should be informed with the robot's position in a great precision. However, since the localization has been done through SLAM, its position data is not that precise (error up to 10cm).

Therefore, we try to implement ArUco, from which we can accurately know the exact pose (x,y,z) frame and transformation matrix from the camera frame. If the ArUco tag is applied to the robot, we can accurately get the robot's position data.

### Objectives

Write a python script that brings the exact pose of the ArUco tag.

- camera intrinsic
- images (streaming)

Output: 

- marker ids
- marker poses (2 markers)
- image with frames drawn
 

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
