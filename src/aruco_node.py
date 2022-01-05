#!/usr/bin/env python2

from logging import raiseExceptions
from sys import path
import os 
import rospy
import cv2
import numpy as np
import imutils
import argparse
import itertools
import tf2_ros as tf2
import tf
from collections import defaultdict
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco

import utils

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
    def __init__(self, **kwargs):
        """
        Object for detecting and tracking ArUco markers.
        ----------
        Args:
        Keyword Args:
            marker_type {string}: The type of ArUco marker to detect.
            marker_size {float}: The size of the ArUco marker in m.
            marker_transform_file {string}: The file containing the transformation matrixes between markers and desired pose.
            aruco_update_rate {float}: The rate at which the ArUco markers are updated.
            aruco_obj_id {string}: The name of the object. A TF frame with that name will be broadcasted.
            save_dir {string}: The directory where the marker_transform_file will be saved after callibration.
            camera_img_topic {string}: The topic where the camera image is published.
            camera_info_topic {string}: The topic where the camera info is published.
            camera_frame_id {string}: The frame id of the camera.
        """
        self.bridge = CvBridge()
        # Settings
        self.marker_type = kwargs["aruco_type"]
        self.marker_size = kwargs["aruco_length"]
        self.marker_transform_file = kwargs["aruco_transforms"]
        self.aruco_update_rate = kwargs["aruco_update_rate"]
        self.aruco_obj_id = kwargs["aruco_obj_id"]
        self.camera_img_topic = kwargs["camera_img_topic"]
        self.camera_info_topic = kwargs["camera_info_topic"]
        self.camera_frame_id = kwargs["camera_frame_id"]


        #--- Used when finding transforms between markers ----#
        self.marker_transforms_list = [] # Transformations between markers
        self.marker_id_list = [] # Ids of markers 
        self.marker_updates_list = []
        #-----------------------------------------------------#

        #---- Markers detected at each camera frame ----#
        self.marker_pose_list = PoseArray() # Poses of markers in camera frame
        self.detected_ids = [] # Coresponding detected ids
        #----------------------------------------------#

        #---- Used at prediction time ----#
        self.obj_transform = Pose()

        if not self.marker_transform_file is None:
            try:
                self.marker_transforms = self.load_marker_transform(self.marker_transform_file)
            except:
                ValueError("Invalid marker transform file")
        #--------------------------------#

        # ROS Publisher
        self.aruco_pub = rospy.Publisher("aruco_img", Image, queue_size=10)
        self.tf_brodcaster = tf2.TransformBroadcaster()
        self.tf_static_brodcaster = tf2.StaticTransformBroadcaster()
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer)
        # ROS Subscriber
        self.image_sub = rospy.Subscriber(
            self.camera_img_topic, Image, self.img_cb)
        self.info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.info_cb)

    def load_marker_transform(self, marker_transform_file):
        """
        Loads the marker transforms from a file.
        ----------
        Args:
            marker_transform_file {string}: The file containing the marker transforms.
        ----------
        Returns:
            dict: A dictionary containing the marker transforms.
        """
        load_unformated = np.load(marker_transform_file, allow_pickle=True)
        mk_transform = load_unformated['mk_tf_dict'][()]
        rospy.loginfo(" TF between markers successfully loaded from file.")
        return mk_transform
    
        
    def img_cb(self, msg): # Callback function for image msg
        """
        Callback when a new image is received.
        ----------
        Args:
            msg {Image}: The image message.
        ----------
            self.markers_img: An image with drawn markers.
            self.marker_pose_list {PoseArray}: A list of poses of the markers in the camera frame.
            self.detected_ids {list}: A corresponding list to self.marker_pose_list, containing the detected ids.
        """

        try:
            self.color_msg = msg
            self.color_img = self.bridge.imgmsg_to_cv2(self.color_msg,"bgr8")

        except CvBridgeError as e:
            print(e)
            
        markers_img, marker_pose_list, id_list = self.detect_aruco(self.color_img)
        self.merkers_img = markers_img
        self.marker_pose_list = marker_pose_list
        self.detected_ids = id_list


    def info_cb(self, msg):
        """
        Callback for the camera information.
        ----------
        Args:
            msg {CameraInfo}: The camera information message.
        ----------
            self.K {numpy.array}: The camera matrix.
            self.D {numpy.array}: The distortion coefficients.
        """
        self.K = np.reshape(msg.K,(3,3))    # Camera matrix
        self.D = np.array(msg.D) # Distortion matrix. 5 for IntelRealsense, 8 for AzureKinect

    def detect_aruco(self, img, broadcast_markers_tf=False):
        """
        Given an RDB image detect aruco markers. 
        ----------
        Args:
            img -- RBG image
        ----------
        Returns:
            image_with_aruco -- image with aruco markers
            marker_pose_list {PoseArray} -- list of poses of the detected markers
            id_list {list} -- list of detected ids
        """
      
        # Create parameters for marker detection
        aruco_dict = aruco.Dictionary_get(ARUCO_DICT[self.marker_type])
        parameters = aruco.DetectorParameters_create()

        parameters.minCornerDistanceRate = 0.02
        parameters.minMarkerDistanceRate = 0.02
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

        # Detect aruco markers
        corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
               
        marker_pose_list = PoseArray()
        id_list = []
        if len(corners) > 0:
            markerLength = self.marker_size
            cameraMatrix = self.K 
            distCoeffs   = self.D
            output_img = img.copy()

            # For numerous markers:
            for i, marker_id in enumerate(ids):
                # Draw bounding box on the marker
                img = aruco.drawDetectedMarkers(img, [corners[i]], marker_id)
                
                rvec,tvec,_ = aruco.estimatePoseSingleMarkers([corners[i]],markerLength, cameraMatrix, distCoeffs) 
                output_img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                
                # Convert its pose to Pose.msg format in order to publish
                marker_pose = self.make_pose(rvec, tvec)

                if broadcast_markers_tf == True:
                    tf_marker = TransformStamped()
                    tf_marker.header.stamp = rospy.Time.now()
                    tf_marker.header.frame_id = self.camera_frame_id
                    tf_marker.child_frame_id = "marker_{}".format(marker_id)
                    tf_marker.transform.translation = marker_pose.position
                    tf_marker.transform.rotation = marker_pose.orientation
                    self.tf_brodcaster.sendTransform(tf_marker)

                marker_pose_list.poses.append(marker_pose)
                id_list.append(int(marker_id))

            output_img = aruco.drawDetectedMarkers(
                img, rejected, borderColor=(100, 0, 240))

        else:
            output_img = img

        out_img = Image()
        out_img = self.bridge.cv2_to_imgmsg(output_img, "bgr8")
        self.aruco_pub.publish(out_img)
    
        return output_img, marker_pose_list, id_list

    def make_pose(self, rvec, tvec):
        """
        Given a rotation vector and a translation vector, returns a Pose.
        ----------
        Args:
            id {int} -- id of the marker
            rvec {np.array} -- rotation vector of the marker
            tvec {np.array} -- translation vector of the marker
        ----------
        Returns:
            Pose -- Pose of the marker
        """
        marker_pose = Pose()
        tvec = np.squeeze(tvec)
        rvec = np.squeeze(rvec)

        r_mat = np.eye(3)
        cv2.Rodrigues(rvec, r_mat)
        tf_mat = np.eye(4)
        tf_mat[0:3,0:3] = r_mat

        quat = tf.transformations.quaternion_from_matrix(tf_mat)

        marker_pose.position.x = tvec[0]
        marker_pose.position.y = tvec[1]
        marker_pose.position.z = tvec[2]

        marker_pose.orientation.x = quat[0]
        marker_pose.orientation.y = quat[1]
        marker_pose.orientation.z = quat[2]
        marker_pose.orientation.w = quat[3]

        return marker_pose


    def calculate_transform(self, id_main):
        """
        Given transforms of all detected markers calculate the pose of the object.
        ----------
        Args:
            id_main {int} -- id of the main marker
        ----------
        Returns:
            Pose -- Estimated pose of the object
        """
        marker_pose_list, detected_ids = self.marker_pose_list, self.detected_ids
        transforms_rot = []
        transforms_trans = []
        for i, marker_id in enumerate(detected_ids):
            
            trans, rot = utils.pose_to_quat_trans(marker_pose_list.poses[i])
            
            if marker_id == id_main:
                transforms_rot.append(rot)
                transforms_trans.append(trans)
                continue
            else:
                if not marker_id in self.marker_transforms:
                    rospy.logwarn(
                        "Unknown marker ID detected {}".format(marker_id))
                    continue
                tf_matrix = utils.quat_trans_to_matrix(trans, rot)
                full_tf = np.dot(
                    tf_matrix, self.marker_transforms[marker_id])
                trans, rot= utils.matrix_to_quat_trans(full_tf)
                transforms_rot.append(rot)
                transforms_trans.append(trans)

        transforms_rot = np.array(transforms_rot)
        transforms_trans = np.array(transforms_trans)

        if len(transforms_rot) == 0:
            return
        elif len(transforms_rot) > 2:
            rotation_mtxs = np.array(
                [tf.transformations.quaternion_matrix(rt) for rt in transforms_rot])
            z_axis = np.array([0, 0, 1, 1])
            z_rotated = np.einsum(
                "ijk,k->ij", rotation_mtxs[:, 0:3, 0:3], z_axis[0:3])
            z_rotated = np.squeeze(z_rotated)
            z_rotated_compare = np.dot(z_rotated, z_rotated.T)

            col_avg = np.average(z_rotated_compare, axis=0)
            outlier = np.argmin(col_avg)
            
            transforms_rot = np.delete(transforms_rot, outlier, axis=0)
            transforms_trans = np.delete(transforms_trans, outlier, axis=0)

        if len(transforms_rot) > 1:
            transforms_rot = np.array(transforms_rot)
            transforms_trans = np.array(transforms_trans)
            
            avg_rot = utils.average_quaternions(transforms_rot)
            avg_trans = np.average(transforms_trans, axis=0)
        else:
            avg_rot = transforms_rot[0]
            avg_trans = transforms_trans[0]

        object_tf = TransformStamped()
        object_tf.header.stamp = rospy.Time.now()
        object_tf.header.frame_id = self.camera_frame_id
        object_tf.child_frame_id = self.aruco_obj_id
        

        
        if self.aruco_update_rate >= 1:
            self.obj_transform = utils.quat_trans_to_pose(avg_trans, avg_rot)
            object_tf.transform.translation = self.obj_transform.position
            object_tf.transform.rotation = self.obj_transform.orientation
            self.tf_brodcaster.sendTransform(object_tf)
            return
        elif self.aruco_update_rate <= 0:
            raise ValueError("Aruco update rate should be between 1 and 0")
        else:
            trans_old, rot_old = utils.pose_to_quat_trans(self.obj_transform)

            trans_final = trans_old * \
                (1-(self.aruco_update_rate)**2) + \
                (self.aruco_update_rate)**2*avg_trans
            rot_final = utils.average_quaternions([rot_old, avg_rot], weights = [(1-self.aruco_update_rate), self.aruco_update_rate])
            self.obj_transform = utils.quat_trans_to_pose(trans_final, rot_final)
        
            object_tf.transform.translation = self.obj_transform.position
            object_tf.transform.rotation = self.obj_transform.orientation
            self.tf_brodcaster.sendTransform(object_tf)

    

def main():
    rospy.loginfo("Starting ArUco node")
    rospy.init_node('aruco_marker_detect')

    aruco_type = rospy.get_param("~aruco_type", "DICT_6X6_100")
    aruco_length = rospy.get_param("~aruco_length", "0.0489")
    aruco_transforms = rospy.get_param("~aruco_transforms", None)
    aruco_update_rate = rospy.get_param("~aruco_update_rate", "0.1")
    aruco_main_marker_id = rospy.get_param("~aruco_main_marker_id", "0")
    aruco_obj_id = rospy.get_param("~aruco_obj_id", "aruco_obj")
    camera_img_topic = rospy.get_param("~camera_img_topic", "/camera/rgb/image_raw")
    camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/rgb/camera_info")
    camera_frame_id = rospy.get_param("~camera_frame_id", "rgb_camera_link")

    params = {
        "aruco_type": aruco_type,
        "aruco_length": aruco_length,
        "aruco_transforms": aruco_transforms,
        "aruco_update_rate": aruco_update_rate,
        "aruco_obj_id": aruco_obj_id,
        "aruco_main_marker_id": aruco_main_marker_id,
        "camera_img_topic": camera_img_topic,
        "camera_info_topic": camera_info_topic,
        "camera_frame_id": camera_frame_id,
    }


    if aruco_transforms is None:
        raise ValueError("No marker transforms provided. Shutting Down")

    aruco_detect = ImageConverter(**params)
    start_time = rospy.get_time()

    while not rospy.is_shutdown():

        rospy.sleep(0.2)
        aruco_detect.calculate_transform(aruco_main_marker_id)


if __name__ == '__main__':
    main()
