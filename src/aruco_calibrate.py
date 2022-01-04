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
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11}


class ArucoCalibrate(object):
    def __init__(self, *args, **kwargs):
        self.bridge = CvBridge()
        # Settings
        self.marker_type = kwargs["aruco_type"]
        self.marker_size = kwargs["aruco_length"]
        self.aruco_update_rate = kwargs["aruco_update_rate"]
        self.save_dir = kwargs["aruco_save_dir"]
        self.camera_img_topic = kwargs["camera_img_topic"]
        self.camera_info_topic = kwargs["camera_info_topic"]
        self.camera_frame_id = kwargs["camera_frame_id"]

        #--- Used when finding transforms between markers ----#
        self.marker_transforms_list = []  # Transformations between markers
        self.marker_id_list = []  # Ids of markers
        self.marker_updates_list = []
        #-----------------------------------------------------#

        #---- Markers detected at each camera frame ----#
        self.marker_pose_list = PoseArray()  # Poses of markers in camera frame
        self.detected_ids = []  # Coresponding detected ids
        #----------------------------------------------#

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
        print(mk_transform)
        return mk_transform

    def test_camera_tf(self):
        "Publish a dummy transform for the camera frame. Camera frame is anchored to world."
        tf_message = TransformStamped()
        tf_message.header.stamp = rospy.Time.now()
        tf_message.header.frame_id = "world"
        tf_message.child_frame_id = self.camera_frame_id
        tf_message.transform.translation.x = 0.0
        tf_message.transform.translation.y = 0.0
        tf_message.transform.translation.z = 0.0
        tf_message.transform.rotation.x = 0.0
        tf_message.transform.rotation.y = 0.0
        tf_message.transform.rotation.z = 0.0
        tf_message.transform.rotation.w = 1.0

        self.tf_brodcaster.sendTransform(tf_message)

    def img_cb(self, msg):  # Callback function for image msg
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
            self.color_img = self.bridge.imgmsg_to_cv2(self.color_msg, "bgr8")

        except CvBridgeError as e:
            print(e)

        markers_img, marker_pose_list, id_list = self.detect_aruco(
            self.color_img)
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
        self.K = np.reshape(msg.K, (3, 3))    # Camera matrix
        # Distortion matrix. 5 for IntelRealsense, 8 for AzureKinect
        self.D = np.array(msg.D)

    def detect_aruco(self, img, broadcast_markers_tf=True):
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
        corners, ids, rejected = aruco.detectMarkers(
            img, aruco_dict, parameters=parameters)

        marker_pose_list = PoseArray()
        id_list = []
        if len(corners) > 0:
            markerLength = self.marker_size
            cameraMatrix = self.K
            distCoeffs = self.D
            output_img = img.copy()

            # For numerous markers:
            for i, marker_id in enumerate(ids):
                # Draw bounding box on the marker
                img = aruco.drawDetectedMarkers(img, [corners[i]], marker_id)

                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    [corners[i]], markerLength, cameraMatrix, distCoeffs)
                output_img = aruco.drawAxis(
                    img, cameraMatrix, distCoeffs, rvec, tvec, 0.05)

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

    def find_transforms(self):
        """
        Given the detected markers, find and update the transforms between the markers.
        ----------
            self.marker_pose_list {PoseArray}: A list of poses of the markers in the camera frame.
            self.detected_ids {list}: A corresponding list to self.marker_pose_list, containing the detected ids.
        ----------
            self.marker_transforms_list {list}: A list of transforms between the markers.
            self.marker_id_list {list}: A list of combination of markers.
            self.marker_updates_list {list}: How many times each combination has been updated.
        """
        marker_pose_list = self.marker_pose_list
        detected_ids = self.detected_ids

        # Get all possible combinations of markers
        id_index = range(len(detected_ids))
        pose_combinations = list(itertools.combinations(id_index, 2))

        # For each possible calculation, calculate the transfromation matrix
        for i, j in pose_combinations:
            combination = [detected_ids[i], detected_ids[j]]
            if combination in self.marker_id_list:
                pose_0 = marker_pose_list.poses[i]
                pose_1 = marker_pose_list.poses[j]
            elif combination[::-1] in self.marker_id_list:
                combination = [detected_ids[j], detected_ids[i]]
                pose_0 = marker_pose_list.poses[j]
                pose_1 = marker_pose_list.poses[i]
            else:
                pose_0 = marker_pose_list.poses[i]
                pose_1 = marker_pose_list.poses[j]

            # Find the transform between the two markers
            tf_matrix_0 = utils.pose_to_matrix(pose_0)
            tf_matrix_1 = utils.pose_to_matrix(pose_1)

            tf_matrix_0_inv = tf.transformations.inverse_matrix(tf_matrix_0)

            tf_0_to_1 = np.dot(tf_matrix_0_inv, tf_matrix_1)

            trans, rotation = utils.matrix_to_quat_trans(tf_0_to_1)

            # If the transform does not yet exist add it. If it already exists update it with a weighted update.
            if (combination not in self.marker_id_list) & (combination[::-1] not in self.marker_id_list):
                self.marker_transforms_list.append(
                    [np.array(trans), np.array(rotation)])
                self.marker_id_list.append(combination)
                self.marker_updates_list.append(1)

            else:
                combination_idx = self.marker_id_list.index(combination)
                average_translation = 0.99 * \
                    self.marker_transforms_list[combination_idx][0] + \
                    0.01 * np.array(trans)
                average_rotation = utils.average_quaternions(
                    [self.marker_transforms_list[combination_idx][1], np.array(rotation)], weights=[0.9, 0.1])

                self.marker_transforms_list[combination_idx][0] = average_translation
                self.marker_transforms_list[combination_idx][1] = average_rotation
                self.marker_updates_list[combination_idx] += 1
        return

    def set_transfroms(self, id_main):
        """
        Given the transforms found in the "find_transforms" function, set the transforms between the markers.
        Shortest "path" between a marker and the main marker is used.
        The transforms are saved in the "marker_transforms.npz" file.
        ----------
        Args:
            id_main {int}: The id of the main marker.
        ----------
            self.marker_transforms {dict} : A dictionary of transforms between the markers.
        """
        graph = self.build_graph(self.marker_id_list)
        paths = {}
        mk_tf = {}
        for start in graph.keys():
            if start == id_main:
                continue
            else:
                paths[start] = self.BFS_SP(graph, start, id_main)

        for marker_id in paths.keys():
            path = paths[marker_id]
            path_len = len(path)
            curr_idx = 0
            next_idx = 1
            while next_idx < path_len:
                combination = [path[curr_idx], path[next_idx]]
                if combination in self.marker_id_list:
                    comb_idx = self.marker_id_list.index(combination)
                    marker_tf = self.marker_transforms_list[comb_idx]
                    marker_tf_mtx = utils.quat_trans_to_matrix(
                        marker_tf[0], marker_tf[1])

                else:
                    combination = [path[next_idx], path[curr_idx]]
                    comb_idx = self.marker_id_list.index(combination)
                    marker_tf = self.marker_transforms_list[comb_idx]
                    marker_tf_mtx_b = utils.quat_trans_to_matrix(
                        marker_tf[0], marker_tf[1])
                    marker_tf_mtx = tf.transformations.inverse_matrix(
                        marker_tf_mtx_b)

                if marker_id in mk_tf:
                    mk_tf[marker_id] = np.matmul(
                        marker_tf_mtx, mk_tf[marker_id])
                else:
                    mk_tf[marker_id] = marker_tf_mtx
                curr_idx = next_idx
                next_idx += 1

        self.marker_transforms = mk_tf
        np.savez(os.path.join(self.save_dir,
                 'marker_transforms.npz'), mk_tf_dict=mk_tf)

        print("The following transforms were saved:", self.load_marker_transform(
            os.path.join(self.save_dir, 'marker_transforms.npz')))
        return

    def build_graph(self, edges):
        """
        Build a node graph given a list of edges.
        ----------
        Args:
            edges {list}: A list of edges.
        ----------
        Returns:
            graph {dict}: A dictionary of nodes and its neighbors.
        """
        graph = defaultdict(list)

        # Loop to iterate over every
        # edge of the graph
        for edge in edges:
            a, b = edge[0], edge[1]

            # Creating the graph
            # as adjacency list
            graph[a].append(b)
            graph[b].append(a)
        return graph

    def BFS_SP(self, graph, start, goal):
        """
        Find the shortest path between two nodes in a graph.
        ----------
        Args:
            graph {dict}: A dictionary of nodes and its neighbors.
            start {int}: The id of the start node.
            goal {int}: The id of the goal node.
        ----------
        Returns:
            path {list}: A list of nodes in the shortest path.
        """
        explored = []

        # Queue for traversing the
        # graph in the BFS
        queue = [[start]]

        # If the desired node is
        # reached
        if start == goal:
            print("Same Node")
            return None

        # Loop to traverse the graph
        # with the help of the queue
        while queue:
            path = queue.pop(0)
            node = path[-1]

            # Condition to check if the
            # current node is not visited
            if node not in explored:
                neighbours = graph[node]

                # Loop to iterate over the
                # neighbours of the node
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)

                    # Condition to check if the
                    # neighbour node is the goal
                    if neighbour == goal:
                        return new_path
                explored.append(node)

        # Condition when the nodes
        # are not connected
        print("So sorry, but a connecting"
              "path doesn't exist :(")
        return None


def main():
    rospy.loginfo("Starting ArUco calibration")
    rospy.init_node('aruco_marker_calibration')

    aruco_type = rospy.get_param("~aruco_type", "DICT_6X6_100")
    aruco_length = rospy.get_param("~aruco_length", "0.0489")
    aruco_update_rate = rospy.get_param("~aruco_update_rate", "0.1")
    aruco_main_marker_id = rospy.get_param("~aruco_main_marker_id", "0")
    aruco_save_dir = rospy.get_param("~aruco_save_dir", None)
    camera_img_topic = rospy.get_param(
        "~camera_img_topic", "/camera/rgb/image_raw")
    camera_info_topic = rospy.get_param(
        "~camera_info_topic", "/camera/rgb/camera_info")
    camera_frame_id = rospy.get_param("~camera_frame_id", "rgb_camera_link")

    params = {
        "aruco_type": aruco_type,
        "aruco_length": aruco_length,
        "aruco_update_rate": aruco_update_rate,
        "aruco_main_marker_id": aruco_main_marker_id,
        "camera_img_topic": camera_img_topic,
        "camera_info_topic": camera_info_topic,
        "camera_frame_id": camera_frame_id,
        "aruco_save_dir": aruco_save_dir
    }

    
    rospy.logwarn(
        "Calibration startingin 3s")
    rospy.logwarn("Make sure to correctly set the Save Directory !!!!!")
    rospy.sleep(3)
    aruco_find_transform = True
    
    aruco_detect = ArucoCalibrate(**params)

    start_time = rospy.get_time()

    while not rospy.is_shutdown():

        if aruco_find_transform == True:
            aruco_detect.test_camera_tf()
            if rospy.get_time() - start_time < 60:
                aruco_detect.find_transforms()
                print(aruco_detect.detected_ids)
                rospy.sleep(0.01)
            else:
                aruco_detect.set_transfroms(aruco_main_marker_id)
                aruco_find_transform = False
                rospy.signal_shutdown("Calibration finished successfully")

if __name__ == '__main__':
    main()
