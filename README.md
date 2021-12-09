# ArUcoDetect
Detect ArUco Marker and visualize its pose

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
    
    **ROS Releases**
    
    [https://github.com/IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros)
    
    **Realsense SDK - for realsense-viewer**
    
    [librealsense/distribution_linux.md at master · IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
    
- Azure Kinect
    
    **Azure Kinect ROS driver**
    
    [Azure-Kinect-Sensor-SDK/usage.md at develop · microsoft/Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#Installation)
    
    - 0. Install nvidia graphic driver
        
        ```bash
        apt-get install nvidia-driver-[version_name]
        ```
        
    - Update openGL to 4.6
        
        ### Install Graphic Drivers
        
        ```bash
        sudo apt install nvidia-driver-470
        ```
        
        ### Install additional libraries
        
        [Install OpenGL at Ubuntu 18.04 LTS](https://medium.com/@theorose49/install-opengl-at-ubuntu-18-04-lts-31f368d0b14e)
        
    - 1. Setup the Repositories [Ubuntu 18.04]
        
        ```bash
        curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
        
        sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
        
        sudo apt-get update
        ```
        
    - 2. Install the Kinect driver files
        
        ```bash
        apt-get install libk4a1.3 libk4a1.3-dev k4a-tools
        ```
        
    
    Once driver is installed, catkin_make will build all ROS files
    
- Detection code
    
    ```bash
    pip install numpy
    sudo -H pip install imutils
    pip install rospy
    pip install cv_bridge
    pip install opencv-contrib-python
    ```
    
    - Install opencv 4.2.0 (**Mandatory**)
        
        [Ubuntu 18.04에 OpenCV 4.2.0 설치하는 방법](https://webnautes.tistory.com/1186)
        
        Prerequisite for openCV 4.2.0
        
        ```bash
        sudo apt-get install build-essential cmakepkg-configlibjpeg-dev 
        libtiff5-dev libpng-devlibavcodec-dev libavformat-dev libswscale-dev 
        libxvidcore-dev libx264-dev libxine2-devlibv4l-dev 
        v4l-utilslibgstreamer1.0-dev libgstreamer-plugins-base1.0-devlibgtk2.0-devmesa-utils 
        libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-devlibatlas-base-dev gfortran 
        libeigen3-devpython2.7-dev python3-dev python-numpy python3-numpy
        ```
        
        1. Configuring for opencv compiling
        
        ```bash
        mkdir opencv
        cd opencv
        
        wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
        unzip opencv.zip
        
        wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
        unzip opencv_contrib.zip
        
        cd opencv-4.0.1/
        mkdir build
        cd build
        
        cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local 
        -D WITH_TBB=OFF -D WITH_IPP=OFF -D WITH_1394=OFF -D BUILD_WITH_DEBUG_INFO=OFF 
        -D BUILD_DOCS=OFF -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON 
        -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D WITH_QT=OFF 
        -D WITH_GTK=ON -D WITH_OPENGL=ON 
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.2.0/modules -D WITH_V4L=ON  
        -D WITH_FFMPEG=ON -D WITH_XINE=ON -D BUILD_NEW_PYTHON_SUPPORT=ON 
        -D OPENCV_GENERATE_PKGCONFIG=ON ../
        
        ```
        
        2. Find number of CPU core
        
        ```bash
        cat /proc/cpuinfo | grep processor | wc -l
        6
        ```
        
        3. Start compiling, adding the CPU number after -j
        
        ```bash
        time make -j6
        ```
        
        4. Install 
        
        ```bash
        sudo make install
        sudo ldconfig
        
        ```
        
        5. Test whether opencv is installed
        
        ```bash
        python /usr/local/share/opencv4/samples/python/facedetect.py --cascade "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml" --nested-cascade "/usr/local/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml" /dev/video0
        ```
        
    - [opencv-contrib-python] cv2 does not have an attribute "aruco"
        
        [Python 3 cv2 Has No Attribute Aruco](https://winkdoubleguns.com/2021/02/13/python-3-cv2-has-no-attribute-aruco/)
        
    - cv_bridge
        
        [Wiki](https://wiki.ros.org/cv_bridge)
        

### Package info

- [ROS Topic] Intel Realsense
    
    [Wiki](http://wiki.ros.org/realsense_camera)
    
- [ROS Topic] Azure Kinect
    
    [Azure_Kinect_ROS_Driver/usage.md at melodic · microsoft/Azure_Kinect_ROS_Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/docs/usage.md)
    
- [Python Package] Opencv ArUco
    
    [OpenCV: ArUco Marker Detection](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa)
    
    **Params**
    
    - rvec (roll,pitch,yaw)
        
        3D rotation vector which defines both an axis of rotation and the rotation angle about that axis, and gives the marker's orientation. It can be converted to a 3x3 rotation matrix using the **Rodrigues** function.
        
        [OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac)
        
    - tvec
        
        Translation (x,y,z) of the marker from the origin; the distance unit is whatever unit you used to define your printed calibration chart
        
### Run ROS Files and Detection Codes

- Basic ROS routine
    
    ```bash
    catkin_ws
    catkin_make
    source devel/setup.zsh
    ```
    
- Intel Realsense ROS
    
    **Get certain that the camera works well**
    
    ```bash
    realsense-viewer
    ```
    
    **Run the launch files**
    
    ```bash
    roslaunch realsense2_camera rs_camera.launch
    ```
    
- Azure Kinect ROS
    
    **Get certain that the camera works well**
    
    ```bash
    k4aviewer
    ```
    
    **Run the launch files**
    
    ```bash
    roslaunch azure_kinect_ros_driver kinect_rgbd.launch
    ```

## Relevant Cases that you can refer to:

- aruco marker recognition with ros and intel realsense 
    
    [ArUco marker recognition using ROS and RealSense](https://www.youtube.com/watch?v=BeI7DZxPadw)
    
    [https://github.com/pal-robotics/aruco_ros](https://github.com/pal-robotics/aruco_ros)
    
    [https://github.com/pal-robotics/ddynamic_reconfigure](https://github.com/pal-robotics/ddynamic_reconfigure)
    
- 6DOF Pose estimation with Aruco marker and ROS
    
    [6DOF pose estimation with Aruco marker and ROS - Robotics with ROS](https://ros-developer.com/2017/04/23/aruco-ros/)
    
- AR using ArUco Marker Detection
    
    [Augmented Reality using Aruco Marker Detection with Python OpenCV - MLK - Machine Learning Knowledge](https://machinelearningknowledge.ai/augmented-reality-using-aruco-marker-detection-with-python-opencv/)
    

### Testing

- Marker Parameters
    
    [Online ArUco markers generator](https://chev.me/arucogen/)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f691a6bc-838d-433d-ad36-f85b1831bb0c/Untitled.png)
    
    - dictionary: 6x6(50,100,250,1000)
    - marker id: 43
    - marker size (mm): 50
