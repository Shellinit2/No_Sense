import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
import cv2
from cv_bridge import CvBridge
import serial
import threading
import queue
import time
from typing import List, Tuple
from pykalman import KalmanFilter
import numpy as np

class SensorPublisherNode(Node):
    def __init__(self):
        super().__init__('sensor_publisher_node')
        self.imu_publisher = self.create_publisher(
            Imu,
            'imu_topic',
            10
        )
        self.left_camera_publisher = self.create_publisher(
            Image,
            'left_camera/image_raw',
            10
        )
        self.right_camera_publisher = self.create_publisher(
            Image,
            'right_camera/image_raw',
            10
        )

        self.bridge = CvBridge()
        self.imu_queue = queue.Queue()
        self.left_image_queue = queue.Queue()
        self.right_image_queue = queue.Queue()

        self.serial_port = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200,
            timeout=1
        )

        self.imu_kalman = self._initialize_kalman_filter()
        self.last_imu_measurements = None

        self.imu_thread = threading.Thread(target=self.read_imu_data)
        self.camera_thread = threading.Thread(target=self.capture_images)
        self.imu_thread.daemon = True
        self.camera_thread.daemon = True
        self.imu_thread.start()
        self.camera_thread.start()

        self.processing_thread = threading.Thread(target=self.process_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Sensor Publisher Node initialized')

def _initialize_kalman_filter(self):
    # State: [ax, ay, az, wx, wy, wz, ax_dot, ay_dot, az_dot, wx_dot, wy_dot, wz_dot]
    state_dim = 12
    measurement_dim = 6

    # Transition matrix for non-constant acceleration
    dt = 0.01  # Assuming 100Hz update rate
    transition_matrix = np.array([
        [1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ])

    # Increased process noise to account for acceleration changes
    Q = np.array([
        [0.2,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
        [0,  0.2,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
        [0,    0,  0.2,      0,      0,      0,      0,      0,      0,      0,      0,      0],
        [0,    0,      0,  0.2,      0,      0,      0,      0,      0,      0,      0,      0],
        [0,    0,      0,      0,  0.2,      0,      0,      0,      0,      0,      0,      0],
        [0,    0,      0,      0,      0,  0.2,      0,      0,      0,      0,      0,      0],
        [0,    0,      0,      0,      0,      0,  0.2,      0,      0,      0,      0,      0],
        [0,    0,      0,      0,      0,      0,      0,  0.2,      0,      0,      0,      0],
        [0,    0,      0,      0,      0,      0,      0,      0,  0.2,      0,      0,      0],
        [0,    0,      0,      0,      0,      0,      0,      0,      0,  0.2,      0,      0],
        [0,    0,      0,      0,      0,      0,      0,      0,      0,      0,  0.2,      0],
        [0,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0.2]
    ])

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=self._get_observation_matrix(),
        initial_state_mean=np.zeros(state_dim),
        observation_covariance=1e-3 * np.eye(measurement_dim),
        transition_covariance=Q,
        em_vars=['transition_covariance', 'initial_state_covariance']
    )

    return kf

    def read_imu_data(self):
        """Read and parse MPU6050 data from serial port"""
        while True:
            try:
                data = self.serial_port.readline().decode('utf-8').strip()
                values = [float(x) for x in data.split(',')]
                if len(values) != 6:
                    continue
                imu_data = np.array([
                    values[0],  # ax
                    values[1],  # ay
                    values[2],  # az
                    values[3],  # wx
                    values[4],  # wy
                    values[5]   # wz
                ])

                if self.last_imu_measurements is not None:
                    (state_mean, _) = self.imu_kalman.filter_update(
                        filtered_state_mean=self.last_imu_measurements,
                        observation=imu_data
                    )
                    self.last_imu_measurements = state_mean

                    imu_msg = Imu()
                    imu_msg.header.stamp = self.get_clock().now().to_msg()
                    imu_msg.linear_acceleration.x = state_mean[0]
                    imu_msg.linear_acceleration.y = state_mean[1]
                    imu_msg.linear_acceleration.z = state_mean[2]
                    imu_msg.angular_velocity.x = state_mean[3]
                    imu_msg.angular_velocity.y = state_mean[4]
                    imu_msg.angular_velocity.z = state_mean[5]

                    self.imu_queue.put(imu_msg)
                else:
                    self.last_imu_measurements = self.imu_kalman.initial_state_mean
                    self.imu_queue.put(self._create_imu_message(imu_data))

            except Exception as e:
                self.get_logger().error(f'Error reading IMU data: {str(e)}')
                time.sleep(0.1)

    def _create_imu_message(self, data):
        """Helper function to create IMU message from raw data"""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.linear_acceleration.x = data[0]
        imu_msg.linear_acceleration.y = data[1]
        imu_msg.linear_acceleration.z = data[2]
        imu_msg.angular_velocity.x = data[3]
        imu_msg.angular_velocity.y = data[4]
        imu_msg.angular_velocity.z = data[5]
        return imu_msg

    def capture_images(self):
        left_cam = cv2.VideoCapture(0)
        right_cam = cv2.VideoCapture(1)

        if not left_cam.isOpened() or not right_cam.isOpened():
            self.get_logger().error('Failed to open one or both cameras')
            return

        while True:
            try:
                left_ret, left_frame = left_cam.read()
                right_ret, right_frame = right_cam.read()

                if not left_ret or not right_ret:
                    continue

                self.left_image_queue.put(left_frame)
                self.right_image_queue.put(right_frame)

            except Exception as e:
                self.get_logger().error(f'Error capturing images: {str(e)}')
                time.sleep(0.1)

    def process_data(self):
        """Process and publish all sensor data"""
        while True:
            try:
                imu_msg = self.imu_queue.get(timeout=1.0)
                self.imu_publisher.publish(imu_msg)
                left_frame = self.left_image_queue.get(timeout=1.0)
                right_frame = self.right_image_queue.get(timeout=1.0)
                left_img_msg = self.bridge.cv2_to_imgmsg(left_frame)
                left_img_msg.header = imu_msg.header
                self.left_camera_publisher.publish(left_img_msg)
                right_img_msg = self.bridge.cv2_to_imgmsg(right_frame)
                right_img_msg.header = imu_msg.header
                self.right_camera_publisher.publish(right_img_msg)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error processing data: {str(e)}')
                time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
