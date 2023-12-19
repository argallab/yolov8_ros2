#!/usr/bin/env python3

"""
PUBLISHERS:
    + /door (geometry_msgs/msg/Point) - The 3D coordinates of detected door
    + /table (geometry_msgs/msg/Point) - The 3D coordinates of detected table

SUBSCRIBERS:
    + /camera/infra1/image_rect_raw/compressed (sensor_msgs/msg/CompressedImage) - The compressed
                                                                                   infrared image
                                                                                   stream
    + /camera/depth/image_rect_raw (sensor_msgs/msg/Image) - The raw rectified depth images
    + /camera/infra1/camera_info (sensor_msgs/msg/CameraInfo) - The camera's metadata and
                                                                calibration information
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import os
from ament_index_python.packages import get_package_prefix
import cv2
# import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np

class YOLOv8Luci(Node):
    """
    This node runs YOLOv8 object detection on LUCI camera streams
    """

    def __init__(self):
        """
        Initializes the node, creates publishers and subscribers, and loads the YOLO model
        """
        super().__init__('yolov8_luci')
        self.bridge = CvBridge()
        self.door_pub = self.create_publisher(PointStamped, 'door', 10)
        self.table_pub = self.create_publisher(PointStamped, 'table', 10)
        self.grayscale_sub = self.create_subscription(
            Image,
            '/luci/ir_right_camera',
            self.grayscale_callback,
            10)

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/awl/right_camera_points',
            self.pointcloud_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/awl/right_camera_info',
            self.camera_info_callback,
            10)
        self.frequency = 200
        self.timer = self.create_timer(1 / self.frequency, self.timer_callback)

        # Load pre-trained weights
        package_prefix_directory = get_package_prefix('object_detection')
        weights_path = os.path.join(package_prefix_directory, 'lib', 'object_detection',
                                    'doors_and_tables.pt')
        self.model = YOLO(weights_path)

        self.grayscale_image = None
        self.pointcloud = None
        self.intrinsics = None
        self.width = 640
        self.height = 360

    def camera_info_callback(self, msg):
        """
        Callback function for the subscriber that subscribes to /camera/infra1/camera_info
        
        Args: msg: Message containing camera metadata and calibration info

        Returns: None
        """
        self.intrinsics = True
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.camera_calibration_matrix = np.array(msg.k).reshape((3, 3))
        self.width = msg.width
        self.height = msg.height

    def grayscale_callback(self, msg):
        """
        Callback function for the subscriber that subscribes to
        /luci/ir_right_camera. Processes and stores the incoming image.
        
        Args: msg: Incoming message containing the grayscale image

        Returns: None
        """
        image_arr = np.array(msg.data).reshape((self.height, self.width))

        # # Rotate the image 90 degrees counterclockwise
        self.grayscale_image = cv2.rotate(image_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imshow("grayscale image", self.grayscale_image)
        # cv2.waitKey(1)

        # Replicate the grayscale channel to mimic an RGB image
        self.grayscale_image = cv2.cvtColor(self.grayscale_image, cv2.COLOR_GRAY2BGR)

    def pointcloud_callback(self, msg):
        """
        Callback function for the subscriber that subscribes to a single camera pointcloud
        (for now we use /awl/right_camera_points)
        Processes and stores the incoming pointcloud.
        
        Args: msg: Incoming message containing the pointcloud

        Returns: None
        """
        self.pointcloud = msg

    def process_image_and_pointcloud(self):
        """
        Processes the incoming images for object detection and publishes the detected objects'
        positions

        Args: None

        Returns: None
        """
        results = self.model(self.grayscale_image, verbose=False)

        # Get the current time to use as the timestamp for detections
        current_time = self.get_clock().now().to_msg()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the confidence value
                confidence = box.conf

                # Check if the confidence value is at least 0.8
                if confidence >= 0.8:
                    # Get bounding box coordinates in (top, left, bottom, right) format
                    bbox = box.xyxy[0].to('cpu').detach().numpy().copy()

                    # Calculate the center of the bounding box
                    bbox_center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]

                    # Draw a circle at the center of the bounding box
                    cv2.circle(self.grayscale_image, (bbox_center[0], bbox_center[1]), radius=5,
                               color=(0, 0, 255), thickness=-1)

                    # Extract the depth values within the bounding box
                    # depth_region = self.depth_image[int(bbox[1]):int(bbox[3]),
                    #                                 int(bbox[0]):int(bbox[2])]

                    # # Filter out zero depth values
                    # non_zero_depths = depth_region[depth_region != 0]

                    # # Calculate the median depth within the bounding box
                    # if non_zero_depths.size > 0:
                    #     median_depth = np.median(non_zero_depths)
                    # else:
                    #     median_depth = 0

                    # if median_depth != 0:
                    #     # Calculate real world coordinates
                    #     coords = rs.rs2_deproject_pixel_to_point(
                    #         self.intrinsics, bbox_center, median_depth
                    #     )

                    #     depth_scale = 0.001

                    #     object_position = PointStamped()
                    #     object_position.header.stamp = current_time
                    #     object_position.header.frame_id = 'camera_infra1_optical_frame'
                    #     object_position.point.x = coords[0] * depth_scale
                    #     object_position.point.y = coords[1] * depth_scale
                    #     object_position.point.z = coords[2] * depth_scale

                        # Publish object's position relative to the camera optical frame
                        # if self.model.names[int(box.cls)] == 'door':
                        #     self.door_pub.publish(object_position)

                        # if self.model.names[int(box.cls)] == 'table':
                        #     self.table_pub.publish(object_position)

        annotated_frame = results[0].plot()
        cv2.imshow("grayscale_image", annotated_frame)
        cv2.waitKey(1)
    
    def timer_callback(self):
        """
        Callback function for the timer. Performs object detection and publishes whenever new
        images are available.

        Args: None

        Returns: None
        """
        if self.grayscale_image is not None and self.pointcloud is not None and \
            self.intrinsics is not None:
            self.process_image_and_pointcloud()

def main(args=None):
    """
    The main function
    """
    rclpy.init(args=args)
    node = YOLOv8Luci()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()