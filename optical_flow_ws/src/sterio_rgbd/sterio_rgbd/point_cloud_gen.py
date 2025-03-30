import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d
from typing import List, Tuple
import threading
import queue

class StereoPointCloudNode(Node):
    def __init__(self):
        super().__init__('stereo_pointcloud_node')
        self.left_image_sub = self.create_subscription(
            Image,
            'left_camera/image_raw',
            self.left_image_callback,
            10
        )

        self.right_image_sub = self.create_subscription(
            Image,
            'right_camera/image_raw',
            self.right_image_callback,
            10
        )

        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            'pointcloud_topic',
            10
        )

        self.bridge = CvBridge()
        self.left_image_queue = queue.Queue()
        self.right_image_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.focal_length = 20  # mm
        self.baseline = 10      # mm
        self.pixel_size = 0.1   # mm
        self.min_depth = 5      # meters
        self.max_depth = 100    # meters
        self.min_disparity = 1  # pixels

    def left_image_callback(self, msg):
        self.left_image_queue.put(self.bridge.imgmsg_to_cv2(msg))

    def right_image_callback(self, msg):
        self.right_image_queue.put(self.bridge.imgmsg_to_cv2(msg))

    def compute_disparity(self, left_img, right_img, block_size=5):
        min_disp = 0
        num_disp = 16 * 5
        block_size = block_size

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32
        )

        disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        min_val = disparity.min()
        max_val = disparity.max()
        disparity = (disparity - min_val) / (max_val - min_val) * 255

        return disparity

    def create_pointcloud(self, disparity, left_img):
        points = []
        colors = []

        height, width = disparity.shape

        for y in range(height):
            for x in range(width):
                d = disparity[y, x] * self.pixel_size
                if d > 0 and d < self.max_depth and d > self.min_disparity:
                    Z = (self.focal_length * self.baseline) / d
                    if Z >= self.min_depth and Z <= self.max_depth:
                        X = (x - width / 2) * Z / self.focal_length
                        Y = (y - height / 2) * Z / self.focal_length
                        color = left_img[y, x] / 255.0
                        points.append([X, Y, Z])
                        colors.append(color)

        points = np.array(points)
        colors = np.array(colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

        return cl

    def process_images(self):
        while True:
            try:
                left_img = self.left_image_queue.get(timeout=1.0)
                right_img = self.right_image_queue.get(timeout=1.0)
                gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                disparity = self.compute_disparity(gray_left, gray_right)

                disparity_filtered = cv2.bilateralFilter(
                    disparity.astype(np.uint8),
                    5, 75, 75
                )

                pcd = self.create_pointcloud(disparity_filtered, left_img)
                header = self.get_clock().now().to_msg()
                ros_cloud = self.bridge.pcl_to_ros_point_cloud(pcd, header)
                self.pointcloud_pub.publish(ros_cloud)

            except queue.Empty:
                continue

def main(args=None):
    rclpy.init(args=args)
    node = StereoPointCloudNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
