import cv2
import numpy as np
import imutils
import argparse
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco



class ImageConverter(object):
    def __init__(self):
        self.bridge = CvBridge()
        # We will get pose from 2 markers; id '35' and '43'
        self.pose_35 = Pose()
        self.pose_43 = Pose()
        self.pose = [self.pose_35,self.pose_43]
        # ROS Publisher
        self.image_pub = rospy.Publisher("/detected_markers",Image, queue_size=1)
        self.id_pub    = rospy.Publisher("/ArUco_ID", String, queue_size=1)
        self.pose_35_pub = rospy.Publisher("/marker_pose/35", Pose, queue_size=1)
        self.pose_43_pub = rospy.Publisher("/marker_pose/43", Pose, queue_size=1)
        # ROS Subscriber
        self.image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.img_cb)
        self.info_sub  = rospy.Subscriber("/rgb/camera_info", CameraInfo, self.info_cb)
        # Names of each possible ArUco tag OpenCV supports
        self.ARUCO_DICT = {
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
        
    def img_cb(self, msg): # Callback function for image msg
        try:
            self.color_msg = msg
            self.color_img = self.bridge.imgmsg_to_cv2(self.color_msg,"bgr8")
            self.color_img = imutils.resize(self.color_img, width=1000)

        except CvBridgeError as e:
            print(e)
            
        markers_img, ids_list = self.detect_aruco(self.color_img)

        if ids_list is None:
            self.id_pub.publish(ids_list)
        else:
            ids_str = ''.join(str(e) for e in ids_list)
            # Publish id msg
            self.id_pub.publish(ids_str)
        try:
            # Publish image msg
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(markers_img, "bgr8"))
            # Publish pose msg
            self.pose_35_pub.publish(self.pose_35)
            self.pose_43_pub.publish(self.pose_43)
        except CvBridgeError as e:
            print(e)

    def info_cb(self, msg): 
        self.K = np.reshape(msg.K,(3,3))    # Camera matrix
        self.D = np.reshape(msg.D,(1,8))[0] # Distortion matrix. 5 for IntelRealsense, 8 for AzureKinect

    def args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--type", type=str,default="DICT_6X6_100", help="type of ArUCo tag to detect")
        ap.add_argument("-l", "--length", type=float, default="0.05", help="length of the marker in meters")
        arguments = vars(ap.parse_args())
        return arguments

    def detect_aruco(self,img):
      
        # Create parameters for marker detection
        args = self.args()
        aruco_dict = aruco.Dictionary_get(self.ARUCO_DICT[args["type"]])
        parameters = aruco.DetectorParameters_create()

        # Detect aruco markers
        corners,ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
               
        if len(corners) > 0:
            markerLength = args["length"]
            cameraMatrix = self.K 
            distCoeffs   = self.D

            # For numerous markers:
            for num in range(len(corners)):
                # Draw bounding box on the marker
                img = aruco.drawDetectedMarkers(img, [corners[num]], ids[num])  
                # Outline marker's frame on the image
                rvec,tvec,_ = aruco.estimatePoseSingleMarkers([corners[num]],markerLength,cameraMatrix,distCoeffs)
                output = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
                # Convert its pose to Pose.msg format in order to publish
                if ids.flatten()[num] == 35:
                    self.make_pose(35,rvec,tvec)
                if ids.flatten()[num] == 43:
                    self.make_pose(43,rvec,tvec)
        else:
            output = img
    
        return output, ids

    def eul2quat(self, roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]

    def make_pose(self,id,rvec,tvec):

        if id == 35:
            index = 0
        if id == 43:
            index = 1

        quat = self.eul2quat(rvec.flatten()[0],rvec.flatten()[1],rvec.flatten()[2])

        self.pose[index].position.x = tvec.flatten()[0]
        self.pose[index].position.y = tvec.flatten()[1]
        self.pose[index].position.z = tvec.flatten()[2]

        self.pose[index].orientation.x = quat[0]
        self.pose[index].orientation.y = quat[1]
        self.pose[index].orientation.z = quat[2]
        self.pose[index].orientation.w = quat[3]

def main():
    print("Initializing ROS-node")
    rospy.init_node('detect_markers', anonymous=True)
    print("Bring the aruco-ID in front of camera")
    ImageConverter()
    rospy.sleep(1)
    rospy.spin()

if __name__ == '__main__':
    main()
