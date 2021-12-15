#!/usr/bin/env python

from logging import raiseExceptions
import rospy
import cv2
import numpy as np
import imutils
import argparse
import itertools
import tf
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco

# Names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11 }

class ImageConverter(object):
    def __init__(self, marker_type, marker_size):
        self.bridge = CvBridge()
        # Settings
        self.marker_type = marker_type
        self.marker_size = marker_size

        # We will get pose from 2 markers; id '35' and '43'
        self.marker_pose_list = PoseArray()
        self.marker_ids = []
        # ROS Publisher
        self.tf_brodcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener
        # ROS Subscriber
        self.image_sub = rospy.Subscriber("/image_raw", Image, self.img_cb)
        self.info_sub  = rospy.Subscriber("/camera_info", CameraInfo, self.info_cb)
        
    def img_cb(self, msg): # Callback function for image msg
        try:
            self.color_msg = msg
            self.color_img = self.bridge.imgmsg_to_cv2(self.color_msg,"bgr8")
            self.color_img = imutils.resize(self.color_img, width=1000)

        except CvBridgeError as e:
            print(e)
            
        markers_img, marker_pose_list, id_list = self.detect_aruco(self.color_img)
        self.merkers_img = markers_img
        self.marker_pose_list = marker_pose_list
        self.detected_ids = id_list

    def info_cb(self, msg): 
        self.K = np.reshape(msg.K,(3,3))    # Camera matrix
        self.D = np.reshape(msg.D,(1,8))[0] # Distortion matrix. 5 for IntelRealsense, 8 for AzureKinect

    def detect_aruco(self,img):
        """
        Given an RDB image detect aruco markers. 
        ----------
        Args:
            img -- RBG image
        ----------
        Returns:
            image_with_aruco -- image with aruco markers
            marker_pose_list {PoseArray} -- list of poses of the detected markers
        """
      
        # Create parameters for marker detection
        aruco_dict = aruco.Dictionary_get(ARUCO_DICT[self.marker_type])
        parameters = aruco.DetectorParameters_create()

        # Detect aruco markers
        corners,ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
               
        marker_pose_list = PoseArray()
        id_list = []
        if len(corners) > 0:
            markerLength = self.marker_size
            cameraMatrix = self.K 
            distCoeffs   = self.D

            # For numerous markers:
            for i, marker_id in enumerate(ids):
                # Draw bounding box on the marker
                img = aruco.drawDetectedMarkers(img, [corners[i]], marker_id)

                # Outline marker's frame on the image
                rvec,tvec,_ = aruco.estimatePoseSingleMarkers([corners[i]],markerLength,cameraMatrix,distCoeffs)
                output_img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
                
                # Convert its pose to Pose.msg format in order to publish
                marker_pose = self.make_pose(id, rvec, tvec)
                marker_pose_list.poses.append(marker_pose)
                id_list.append(id)

        else:
            output_img = img
    
        return output_img, marker_pose_list, id_list

    def make_pose(self,id,rvec,tvec):
        """
        Given a marker id, euler angles and a translation vector, returns a Pose.
        ----------
        Args:
            id {int} -- id of the marker
            rvec {np.array} -- euler angles of the marker
            tvec {np.array} -- translation vector of the marker
        ----------
        Returns:
            Pose -- Pose of the marker
        """

        marker_pose = Pose()

        quat = self.eul2quat(rvec.flatten()[0],rvec.flatten()[1],rvec.flatten()[2])

        marker_pose.position.x = tvec.flatten()[0]
        marker_pose.position.y = tvec.flatten()[1]
        marker_pose.position.z = tvec.flatten()[2]

        marker_pose.orientation.x = quat[0]
        marker_pose.orientation.y = quat[1]
        marker_pose.orientation.z = quat[2]
        marker_pose.orientation.w = quat[3]

        return marker_pose

    def find_transforms(self, id_main = 1):
        pose_combinations = list(itertools.combinations(self.detected_ids, 2))
        

def main():
    rospy.loginfo("Starting ArUco node")
    rospy.init_node('aruco_marker_detect', anonymous=True)

    aruco_type = rospy.get_param("~aruco_type", "DICT_6X6_100")
    aruco_length = rospy.get_param("~aruco_length", "0.05")
    aruco_find_transform = rospy.get_param("~aruco_find_transform", "True")

    ImageConverter(aruco_type, aruco_length)

    rospy.init_node('Aruco detection started!', anonymous=True)

    while rospy.is_shutdown():
        if aruco_find_transform:
            ImageConverter.find_transforms()
    rospy.Rate(10)
    rospy.spin()

if __name__ == '__main__':
    main()
