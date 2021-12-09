# ArUcoDetect
Detect ArUco Marker and visualize its pose

### Motivation

When the manipulator serves the food to the serving robot, the manipulator should be informed with the robot's position in a great precision. However, since the localization has been done through SLAM, its position data is not that precise (error up to 10cm).

Therefore, we try to implement ArUco, from which we can accurately know the exact pose (x,y,z) frame and transformation matrix from the camera frame. If the ArUco tag is applied to the robot, we can accurately get the robot's position data.

### Objectives

Write a python script that brings the exact pose of the ArUco tag.
