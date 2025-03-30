import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
import open3d as o3d
from typing import List, Tuple
import threading
import queue

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        self.imu_subscriber = self.create_subscription(
            Imu,
            'imu_topic',
            self.imu_callback,
            10
        )
        self.pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            'pointcloud_topic',
            self.pointcloud_callback,
            10
        )
        self.odometry_publisher = self.create_publisher(
            PoseStamped,
            'odometry_topic',
            10
        )

        self.current_pointcloud = None
        self.previous_pointcloud = None
        self.current_pose = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]])

        self.imu_queue = queue.Queue()
        self.pointcloud_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def imu_callback(self, msg):
        """Process incoming IMU data"""
        self.imu_queue.put({
            'timestamp': msg.header.stamp.nanoseconds,
            'acceleration': [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ],
            'angular_velocity': [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ]
        })

    def pointcloud_callback(self, msg):
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])

        gen = pc2.read_points(msg, skip_nans=True)
        int_data = list(gen)

        xyz = np.empty((len(int_data), 3))
        rgb = np.empty((len(int_data), 3))

        for idx, x in enumerate(int_data):
            xyz[idx] = x[:3]
            test = x[3]
            s = struct.pack('>f',test)
            i = struct.unpack('>l',s)[0]
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>>16
            g = (pack & 0x0000FF00)>>8
            b = (pack & 0x000000FF)
            rgb[idx] = [r/255.0, g/255.0, b/255.0]

        self.pointcloud_queue.put({
            'points': xyz,
            'colors': rgb,
            'timestamp': msg.header.stamp.nanoseconds
        })

    def process_data(self):
        while True:
            try:
                imu_data = self.imu_queue.get(timeout=1.0)
                pointcloud_data = self.pointcloud_queue.get(timeout=1.0)
                if self.current_pointcloud is not None:
                    current_pcd = o3d.geometry.PointCloud()
                    current_pcd.points = o3d.utility.Vector3dVector(pointcloud_data['points'])

                    previous_pcd = o3d.geometry.PointCloud()
                    previous_pcd.points = o3d.utility.Vector3dVector(self.current_pointcloud['points'])

                    result = o3d.registration.registration_icp(
                        current_pcd, previous_pcd, 0.2,
                        self.current_pose, o3d.registration.TransformationEstimationPointToPoint()
                    )

                    self.current_pose = result.transformation @ self.current_pose
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.pose.position.x = self.current_pose[0, 3]
                    pose_msg.pose.position.y = self.current_pose[1, 3]
                    pose_msg.pose.position.z = self.current_pose[2, 3]

                    # Convert rotation matrix to quaternion
                    quat = self.rotation_matrix_to_quaternion(self.current_pose[:3, :3])
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]

                    self.odometry_publisher.publish(pose_msg)

                self.previous_pointcloud = self.current_pointcloud
                self.current_pointcloud = pointcloud_data

            except queue.Empty:
                continue

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        qw = np.sqrt(1 + rotation_matrix[0,0] + rotation_matrix[1,1] + rotation_matrix[2,2]) / 2
        qx = (rotation_matrix[2,1] - rotation_matrix[1,2]) / (4 * qw)
        qy = (rotation_matrix[0,2] - rotation_matrix[2,0]) / (4 * qw)
        qz = (rotation_matrix[1,0] - rotation_matrix[0,1]) / (4 * qw)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
