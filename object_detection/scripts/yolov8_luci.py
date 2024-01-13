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
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from sensor_msgs_py import point_cloud2

from object_detection_interfaces.srv import FilterPointcloud

import os
from ament_index_python.packages import get_package_prefix
import cv2
from ultralytics import YOLO
import numpy as np
from scipy.spatial.transform import Rotation as rot


class YOLOv8Luci(Node):
    """
    This node runs YOLOv8 object detection on LUCI camera streams
    """

    def __init__(self):
        """
        Initializes the node, creates publishers and subscribers, and loads the YOLO model
        """
        super().__init__('yolov8_luci')

        self.__initialize_variables()
        self.__initialize_callback_groups()
        self.__initialize_publishers()
        self.__initialize_subscribers()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # service for pointcloud cone projection
        self.filter_pointcloud_cli = self.create_client(FilterPointcloud, 
                                                        '/filter_pointcloud/awl/right_camera_points',
                                                        callback_group=self.srv_callback_group)
        self.filter_pointcloud_cli.wait_for_service()
        self.get_logger().info('yolov8_luci: filter_pointcloud service found')

        self.frequency = 50
        self.timer = self.create_timer(1 / self.frequency, self.timer_callback, 
                                       callback_group=self.timer_callback_group)
        self.tf_timer = self.create_timer(1 / self.frequency, self.tf_timer_callback, 
                                          callback_group=self.tf_timer_callback_group)

        # Load pre-trained weights
        package_prefix_directory = get_package_prefix('object_detection')
        weights_path = os.path.join(package_prefix_directory, 'lib', 'object_detection',
                                    'doors_and_tables.pt')
        self.model = YOLO(weights_path)
        
    def __initialize_callback_groups(self):
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.srv_callback_group = MutuallyExclusiveCallbackGroup()
        self.subscriber_callback_group = MutuallyExclusiveCallbackGroup()

    def __initialize_publishers(self):
        self.door_pub = self.create_publisher(PointStamped, 'door', 10)
        self.table_pub = self.create_publisher(PointStamped, 'table', 10)
        self.ray_marker_pub = self.create_publisher(Marker, 'ray_marker', 10)
        self.debug_pointcloud_pub = self.create_publisher(PointCloud2, '/ray_filtered_points', 10)

    def __initialize_subscribers(self):
        self.grayscale_sub = self.create_subscription(
            Image,
            '/luci/ir_right_camera',
            self.grayscale_callback,
            10, 
            callback_group=self.subscriber_callback_group)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/awl/right_camera_info',
            self.camera_info_callback,
            10,
            callback_group=self.subscriber_callback_group)

    def __initialize_variables(self):
        self.grayscale_image = None
        self.intrinsics = None
        self.width = 640
        self.height = 360
        self.rotation_tf = None
        self.translation_tf = None

    def camera_info_callback(self, msg):
        """
        Callback function for the subscriber that subscribes to /camera/infra1/camera_info
        
        Args: msg: Message containing camera metadata and calibration info

        Returns: None
        """
        self.intrinsics = True
        self.cx = msg.k[5]
        self.cy = msg.k[2]
        self.fx = msg.k[4]
        self.fy = msg.k[0]
        
        self.camera_calibration_matrix = np.array([[self.fx, 0, self.cx], 
                                                   [0, self.fy, self.cy], 
                                                   [0, 0, 1]])

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
        image_rotate = cv2.rotate(image_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Replicate the grayscale channel to mimic an RGB image
        color_image = cv2.cvtColor(image_rotate, cv2.COLOR_GRAY2BGR)

        # NOTE: enhancing the luci camera image, referenced from 
        # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        # converting to LAB color space
        lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        self.grayscale_image = enhanced_img

        # cv2.imshow("grayscale image", self.grayscale_image)
        # cv2.waitKey(1)


    def process_image_and_pointcloud(self):
        """
        Processes the incoming images for object detection and publishes the detected objects'
        positions

        Args: None

        Returns: None
        """
        print("trying to process image and pointcloud")
        results = self.model(self.grayscale_image, verbose=False)

        # Get the current time to use as the timestamp for detections
        current_time = self.get_clock().now().to_msg()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the confidence value
                confidence = box.conf

                # Check if the confidence value is at least 0.8
                if confidence >= 0.3:
                    print("found a door!!!")
                    # Get bounding box coordinates in (top, left, bottom, right) format
                    bbox = box.xyxy[0].to('cpu').detach().numpy().copy()

                    # Calculate the center of the bounding box
                    bbox_center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]

                    # Calculate the 3d ray in normalized pixel coordinates
                    ray = np.array([[(bbox_center[0] - self.cx) / self.fx],
                                    [(bbox_center[1] - self.cy) / self.fy],
                                    [1.0]])
                    print(f"ray: {ray}")

                    req = FilterPointcloud.Request()
                    req.x_dir = ray[0, 0]
                    req.y_dir = ray[1, 0]
                    req.z_dir = ray[2, 0]

                    filter_future = self.filter_pointcloud_cli.call_async(req)

                    while not filter_future.done():
                        self.get_logger().info("...")
                    filter_result = filter_future.result()
                    if filter_result.success:
                        print("service call success --- successfully filtered pointcloud")
                        # publish the filtered pointcloud
                        filtered_cloud = filter_result.out_cloud
                        filtered_cloud.header.stamp = current_time
                        filtered_cloud.header.frame_id = 'right_camera_link'
                        self.debug_pointcloud_pub.publish(filter_result.out_cloud)
                    else:
                        print("service call failed --- failed to filter pointcloud")
                        
                    # Convert 3D ray to world coordinates
                    ray_world = self.rotation_tf.as_matrix() @ ray + self.translation_tf
                    
                    ray_marker = self.__create_3dray_marker(ray_world)
                    self.ray_marker_pub.publish(ray_marker)

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

        print("DISPLAYING YOLO RESULT---not displaying because threading")
        # annotated_frame = results[0].plot()
        # cv2.imshow("grayscale_image", annotated_frame)
        # cv2.waitKey(1)
    
    def timer_callback(self):
        """
        Callback function for the timer. Performs object detection and publishes whenever new
        images are available.

        Args: None

        Returns: None
        """
        if self.grayscale_image is not None and \
            self.intrinsics is not None and self.rotation_tf is not None:
            self.process_image_and_pointcloud()

    def tf_timer_callback(self):
        """
        Callback function for the tf listener. Updates the transformation matrix from camera frame
        to world frame. This is necessary for converting the 2D bounding box center to 3D world

        Args: None

        Returns: None
        """
        try:

            # NOTE: for now, find the transform from right_camera_link to awl_base_link
            #       when getting to test nav, maybe change this to map frame?
            # t = self.tf_buffer.lookup_transform('right_camera_link', 'awl_base_link', rclpy.time.Time())
            t = self.tf_buffer.lookup_transform('awl_base_link', 'right_camera_link', rclpy.time.Time())
            self.translation_tf = np.array([[t.transform.translation.x], [t.transform.translation.y], [t.transform.translation.z]])
            self.rotation_tf = rot.from_quat([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])
            return
        except TransformException as ex:
            print(ex)
            return

    def __create_3dray_marker(self, ray_world):
        marker = Marker()
        marker.header.frame_id = "awl_base_link"
        # marker.header.frame_id = "right_camera_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start_point = Point()
        start_point.x = self.translation_tf[0, 0]
        start_point.y = self.translation_tf[1, 0]
        start_point.z = self.translation_tf[2, 0]
        marker.points.append(start_point)

        end_point = Point()
        end_point.x = ray_world[0, 0]
        end_point.y = ray_world[1, 0]
        end_point.z = ray_world[2, 0]
        marker.points.append(end_point)
        # print(f"start point: {self.translation_tf}, \nend_point: {ray_world}")
        
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker
    
    def shutdown_hook(self):
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=6)
    # executor = SingleThreadedExecutor()   # for debugging
    node = YOLOv8Luci()
    executor.add_node(node)

    try:
        node.get_logger().info('Door detection using LUCI camera streams running, shutdown with CTRL-C')
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Keyboard interrupt, shutting down Door detection using LUCI camera streams running\n')
        node.shutdown_hook()
    finally:
        executor.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()
