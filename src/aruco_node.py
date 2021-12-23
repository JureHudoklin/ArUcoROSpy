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
import tf2_ros as tf
from collections import defaultdict
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseArray
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
    def __init__(self, marker_type, marker_size, marker_transform_file = None, aruco_update_rate = 0.1, aruco_object_id = "aruco_marker", save_dir = None):
        self.bridge = CvBridge()
        # Settings
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.marker_transform_file = marker_transform_file
        self.aruco_update_rate = aruco_update_rate
        self.aruco_object_id = aruco_object_id
        self.save_dir = save_dir

        # We will get pose from 2 markers; id '35' and '43'

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
        if not marker_transform_file is None:
            self.marker_transforms = self.load_marker_transform(marker_transform_file)
        #--------------------------------#

        # ROS Publisher
        self.aruco_pub = rospy.Publisher("aruco_img", Image, queue_size=10)
        self.tf_brodcaster = tf.TransformBroadcaster()
        self.tf_static_brodcaster = tf.StaticTransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        # ROS Subscriber
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb)
        self.info_sub  = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_cb)

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
    
    # def test_camera_tf(self):
    #     # TEST BRODCAST CAMERA TRANSFORM
    #     self.camera_transform = Pose()
    #     self.camera_transform.position.x = 0.0
    #     self.camera_transform.position.y = 0.0
    #     self.camera_transform.position.z = 0.0
    #     self.camera_transform.orientation.x = 0.0
    #     self.camera_transform.orientation.y = 0.0
    #     self.camera_transform.orientation.z = 0.0
    #     self.camera_transform.orientation.w = 1.0
    #     self.tf_brodcaster.sendTransform((self.camera_transform.position.x, self.camera_transform.position.y, self.camera_transform.position.z),
    #                                             (self.camera_transform.orientation.x, self.camera_transform.orientation.y,
    #                                              self.camera_transform.orientation.z, self.camera_transform.orientation.w),
    #                                             rospy.Time.now(),
    #                                             "camera",
    #                                             "world")
        
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
            self.color_img = imutils.resize(self.color_img, width=1000)

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
            id_list {list} -- list of detected ids
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
                rvec,tvec,_ = aruco.estimatePoseSingleMarkers([corners[i]],markerLength,cameraMatrix, distCoeffs)
                output_img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                out_img = Image()
                out_img = self.bridge.cv2_to_imgmsg(output_img, "bgr8")
                self.aruco_pub.publish(out_img)
                
                # Convert its pose to Pose.msg format in order to publish
                marker_pose = self.make_pose(rvec, tvec)


               
                self.tf_brodcaster.sendTransform((marker_pose.position.x, marker_pose.position.y, marker_pose.position.z),
                                                (marker_pose.orientation.x, marker_pose.orientation.y,
                                                 marker_pose.orientation.z, marker_pose.orientation.w),
                                                rospy.Time.now(),
                                                "marker_{}".format(marker_id),
                                                "world")

                marker_pose_list.poses.append(marker_pose)
                id_list.append(int(marker_id))

        else:
            output_img = img
    
        return output_img, marker_pose_list, id_list

    def make_pose(self, rvec, tvec):
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

        quat = self.eul2quat(rvec.flatten()[0], rvec.flatten()[1], rvec.flatten()[2])


        marker_pose.position.x = tvec.flatten()[0]
        marker_pose.position.y = tvec.flatten()[1]
        marker_pose.position.z = tvec.flatten()[2]

        marker_pose.orientation.x = quat[0]
        marker_pose.orientation.y = quat[1]
        marker_pose.orientation.z = quat[2]
        marker_pose.orientation.w = quat[3]

        return marker_pose

    def eul2quat(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
            np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
            np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
            np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
            np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

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
            # if (combination == [55, 65]) or (combination == [65, 55]):
            #     print((-tf_matrix_0[0:3,3] + tf_matrix_1[0:3,3])-tf_0_to_1[0:3,3])

            trans, rotation = utils.matrix_to_quat_trans(tf_0_to_1)
            trans = -tf_matrix_0[0:3,3] + tf_matrix_1[0:3,3]
            
            # If the transform does not yet exist add it. If it already exists update it with a weighted update.
            if (combination not in self.marker_id_list) & (combination[::-1] not in self.marker_id_list):
                self.marker_transforms_list.append(
                    [np.array(trans), np.array(rotation)])
                self.marker_id_list.append(combination)
                self.marker_updates_list.append(1)

            else:
                combination_idx = self.marker_id_list.index(combination)
                average_translation = 0.999*self.marker_transforms_list[combination_idx][0] + 0.001 * np.array(trans)
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
                    mk_tf[marker_id] = np.matmul(marker_tf_mtx, mk_tf[marker_id])
                else:
                    mk_tf[marker_id] = marker_tf_mtx
                curr_idx = next_idx
                next_idx += 1

        self.marker_transforms = mk_tf
        np.savez(os.path.join(self.save_dir, '/marker_transforms.npz'), mk_tf_dict=mk_tf)

        print("The following transforms were saved:", self.load_marker_transform(os.path.join(self.save_dir, '/marker_transforms.npz')))
        return

    def calculate_transform(self, id_main):
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
                tf_matrix = utils.quat_trans_to_matrix(trans, rot)
                full_tf = np.dot(
                    tf_matrix, self.marker_transforms[marker_id])
                trans, rot= utils.matrix_to_quat_trans(full_tf)
                transforms_rot.append(rot)
                transforms_trans.append(trans)

        if len(transforms_rot) == 0:
            return

        transforms_rot = np.array(transforms_rot)
        transforms_trans = np.array(transforms_trans)
        
        avg_rot = utils.average_quaternions(transforms_rot)
        avg_trans = np.average(transforms_trans, axis=0)
        
        if self.aruco_update_rate >= 1:
            self.obj_transform = utils.quat_trans_to_pose(avg_trans, avg_rot)
            self.tf_static_brodcaster.sendTransform(avg_trans, avg_rot, rospy.Time.now(), self.obj_id, "camera")
            return
        elif self.aruco_update_rate <= 0:
            raise ValueError("Aruco update rate should be between 1 and 0")
        else:
            trans_old, rot_old = utils.pose_to_quat_trans(self.obj_transform)

            trans_final = trans_old*(1-self.aruco_update_rate/10) + (self.aruco_update_rate/10)*avg_trans
            rot_final = utils.average_quaternions([rot_old, avg_rot], weights = [(1-self.aruco_update_rate), self.aruco_update_rate])
            self.obj_transform = utils.quat_trans_to_pose(trans_final, rot_final)
        
            self.tf_brodcaster.sendTransform(
                trans_final, rot_final, rospy.Time.now(),  self.obj_id, "camera")

            tf.TransformBroadcaster


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
    rospy.loginfo("Starting ArUco node")
    rospy.init_node('aruco_marker_detect')

    aruco_type = rospy.get_param("~aruco_type", "DICT_6X6_100")
    aruco_length = rospy.get_param("~aruco_length", "0.1")
    aruco_transforms = rospy.get_param("~aruco_transforms", None)
    aruco_update_rate = rospy.get_param("~aruco_update_rate", "0.1")
    aruco_obj_id = rospy.get_param("~aruco_obj_id", "aruco_obj")

    if aruco_transforms is None:
        rospy.logwarn("No marker transforms provided. Calibration started in 3s")
        rospy.logwarn("Make sure to correctly set the Save Directory !!!!!")
        rospy.sleep(3)
        aruco_find_transform = True
    else:
        aruco_find_transform = False

    aruco_detect = ImageConverter(
        aruco_type, aruco_length, aruco_transforms, aruco_update_rate, aruco_obj_id)

    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        
        
        if aruco_find_transform == True:
            if rospy.get_time() - start_time < 60:
                aruco_detect.find_transforms()
                print(aruco_detect.detected_ids)
                rospy.sleep(0.01)
            else:
                aruco_detect.set_transfroms(55)
                aruco_find_transform = False
            #rospy.sleep(1)
        else:
            rospy.sleep(0.1)
            aruco_detect.test_camera_tf()
            aruco_detect.calculate_transform(55)


if __name__ == '__main__':
    main()
