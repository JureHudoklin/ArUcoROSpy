#!/usr/bin/env python

from __future__ import print_function
import rospy
import cv2
import numpy as np
import tf2_ros as tf2
import tf
from collections import defaultdict
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from cv_bridge import CvBridge, CvBridgeError

from aruco_detect.srv import ArucoPoseEstimate, ArucoPoseEstimateResponse

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
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11}

class ArucoDetection(object):
    def __init__(self, *args, **kwargs):
        """
        Aruco detection class.
        """


        self.bridge = CvBridge()
        # Get params
        self.marker_transform_file = kwargs.get('marker_transform_file', None)
        self.marker_type = kwargs.get('aruco_type', 'DICT_6X6_100')
        self.marker_size = kwargs.get('aruco_length', 0.05)
        self.main_marker_id = kwargs.get('main_marker_id', 0)

        # Create the service
        self.pose_estimate_srv = rospy.Service('aruco_pose_estimate',
                                ArucoPoseEstimate, self.estimate_pose_cb)

        #---- Markers detected at each camera frame ----#
        self.marker_pose_list = PoseArray()  # Poses of markers in camera frame
        self.detected_ids = []  # Coresponding detected ids
        #----------------------------------------------#

        #---- Used at prediction time ----#
        if not self.marker_transform_file is None:
            try:
                self.marker_transforms = self.load_marker_transform(
                    self.marker_transform_file)
            except:
                ValueError("Invalid marker transform file")
        #--------------------------------#
        rospy.loginfo("Aruco detection service ready.")


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
    
    def estimate_pose_cb(self, req):
        """
        Estimate the pose of the object given a request (Image, CameraInfo)
        ----------
        Response:
            ArucoPoseEstimateResponse: The estimated pose of the object. Success or failure.
        """
        image = req.image
        camera_info = req.camera_info
        K, D = self.caminfo_to_matrx_dist(camera_info)

        try:
            color_img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Detect markers
        output_img, marker_pose_list, detected_id_list = self.detect_aruco(color_img, K, D)
        
        estimated_pose = self.calculate_transform(self.main_marker_id, marker_pose_list, detected_id_list)

        response = ArucoPoseEstimateResponse()
        if estimated_pose is None:
            response.success = False
        else:
            response.success = True
            response.pose = estimated_pose

        return response

    def caminfo_to_matrx_dist(self, caminfo):
        """
        Converts a camera info message to a camera matrix and distortion coefficients.
        ----------
        Args:
            msg {CameraInfo}: The camera information message.
        ----------
        Returns:
            K {numpy.array}: The camera matrix.
            D {numpy.array}: The distortion coefficients.
        """
        K = np.reshape(caminfo.K, (3, 3))    # Camera matrix
        # Distortion matrix. 5 for IntelRealsense, 8 for AzureKinect
        D = np.array(caminfo.D)
        return K, D

    def detect_aruco(self, img, camera_matrix, dist_coeffs):
        """
        Given an RDB image detect aruco markers. 
        ----------
        Args:
            img {Image} -- RBG image
            camera_matrix {np.array} -- camera matrix 3x3
            dist_coeffs {np.array} -- distortion coefficients (len 4,5,8 or 12)
        ----------
        Returns:
            output_img -- image with aruco markers
            marker_pose_list {PoseArray} -- list of poses of the detected markers
            id_list {list} -- list of detected ids
        """

        # Create parameters for marker detection
        aruco_dict = aruco.Dictionary_get(ARUCO_DICT[self.marker_type])
        parameters = aruco.DetectorParameters_create()

        # Detect aruco markers
        corners, ids, _ = aruco.detectMarkers(
            img, aruco_dict, parameters=parameters)

        marker_pose_list = PoseArray()
        id_list = []
        if len(corners) > 0:
            markerLength = self.marker_size

            # For numerous markers:
            for i, marker_id in enumerate(ids):
                # Draw bounding box on the marker
                img = aruco.drawDetectedMarkers(img, [corners[i]], marker_id)

                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    [corners[i]], markerLength, camera_matrix, dist_coeffs)
                output_img = aruco.drawAxis(
                    img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    
                out_img = Image()
                out_img = self.bridge.cv2_to_imgmsg(output_img, "bgr8")

                # Convert its pose to Pose.msg format in order to publish
                marker_pose = self.make_pose(rvec, tvec)

                marker_pose_list.poses.append(marker_pose)
                id_list.append(int(marker_id))

        else:
            output_img = img

        return output_img, marker_pose_list, id_list

    def make_pose(self, rvec, tvec):
        """
        Given rotation vector and a translation vector, returns a Pose.
        ----------
        Args:
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
        tf_mat[0:3, 0:3] = r_mat

        quat = tf.transformations.quaternion_from_matrix(tf_mat)

        marker_pose.position.x = tvec[0]
        marker_pose.position.y = tvec[1]
        marker_pose.position.z = tvec[2]

        marker_pose.orientation.x = quat[0]
        marker_pose.orientation.y = quat[1]
        marker_pose.orientation.z = quat[2]
        marker_pose.orientation.w = quat[3]

        return marker_pose

    def calculate_transform(self, id_main, marker_pose_list, detected_ids):
        """
        Given transforms of all detected markers calculate the pose of the object.
        ----------
        Args:
            id_main {int} -- id of the main marker
            marker_pose_list {PoseArray} -- list of poses of the detected markers
            detected_ids {list} -- list of detected ids
        ----------
        Returns:
            Pose -- Estimated pose of the object
        """
        transforms_rot = []
        transforms_trans = []
        for i, marker_id in enumerate(detected_ids):

            trans, rot = utils.pose_to_quat_trans(marker_pose_list.poses[i])

            if marker_id == id_main:
                transforms_rot.append(rot)
                transforms_trans.append(trans)
                continue
            else:
                tf_matrix = utils.quat_trans_to_matrix(trans, rot)
                if marker_id in self.marker_transforms:
                    full_tf = np.dot(
                        tf_matrix, self.marker_transforms[marker_id])
                    trans, rot = utils.matrix_to_quat_trans(full_tf)
                    transforms_rot.append(rot)
                    transforms_trans.append(trans)
                else:
                    rospy.logwarn("Unknown Aruco marker present.")
                    continue

        transforms_rot = np.array(transforms_rot)
        transforms_trans = np.array(transforms_trans)

        if len(transforms_rot) == 0:
            return None
        elif len(transforms_rot) > 2:
            rotation_mtxs = np.array(
                [tf.transformations.quaternion_matrix(rt) for rt in transforms_rot])
            z_axis = np.array([0, 0, 1, 1])
            z_rotated = np.einsum(
                "ijk,k->ij", rotation_mtxs[:, 0:3, 0:3], z_axis[0:3])
            z_rotated = np.squeeze(z_rotated)
            z_rotated_compare = np.dot(z_rotated, z_rotated.T)
            row_sum = np.sum(z_rotated_compare, axis=1)
            mask = np.where(
                np.sign(z_rotated_compare[:, 0]) != np.sign(row_sum), False, True)

            if np.any(mask == False):
                print("Aruco Marker flipped. Discarding")
            transforms_rot = transforms_rot[mask]
            transforms_trans = transforms_trans[mask]

        transforms_rot = np.array(transforms_rot)
        transforms_trans = np.array(transforms_trans)

        avg_rot = utils.average_quaternions(transforms_rot)
        avg_trans = np.average(transforms_trans, axis=0)

        obj_transform = Pose()
        
        if self.aruco_update_rate >= 1:
            obj_transform = utils.quat_trans_to_pose(avg_trans, avg_rot)
            return obj_transform
        
        

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('aruco_pose_service_node')
    rospy.loginfo("Starting ArUco service")

    aruco_type = rospy.get_param("~aruco_type", "DICT_6X6_100")
    aruco_length = rospy.get_param("~aruco_length", "0.0489")
    aruco_transforms = rospy.get_param("~aruco_transforms", None)
    aruco_main_marker_id = rospy.get_param("~aruco_main_marker_id")

    params = {"aruco_type": aruco_type,
              "aruco_length": aruco_length,
              "aruco_transforms": aruco_transforms,
              "aruco_main_marker_id": aruco_main_marker_id}

    aruco_detection = ArucoDetection(**params)

    # Spin
    rospy.spin()