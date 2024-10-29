import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import ros2_numpy as rnp
import numpy as np


from enum import Enum
import transforms3d as t3d
import subprocess
import re
import time
import cv2
import tf2_ros

from modelling.scd_agent import StateTimeredPickingAgent, BaseAgent


def call_ros2_service(activate_controllers, deactivate_controllers):
    service_name = '/controller_manager/switch_controller'
    service_type = 'controller_manager_msgs/srv/SwitchController'
    strictness = '2'
    activate_asap = 'true'

    command = f'ros2 service call {service_name} {service_type} "{{activate_controllers: [\"{activate_controllers}\"],\
            deactivate_controllers: [\"{deactivate_controllers}\"], strictness: {strictness}, activate_asap: {activate_asap}}}"'
    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        match = re.search(r'response:\n(.*)', result.stdout, re.DOTALL)
        print(f"{activate_controllers}:", match.group(1).strip())
    except subprocess.CalledProcessError as e:
        print(f"Error calling ROS 2 service: {e}")


def check_pose_stamped_values(pose_stamped_msg):
    position = pose_stamped_msg.pose.position
    orientation = pose_stamped_msg.pose.orientation
    is_position_zero = position.x == 0.0 and position.y == 0.0 and position.z == 0.0
    
    is_orientation_zero_except_w = orientation.x == 0.0 and orientation.y == 0.0 and orientation.z == 0.0 and orientation.w == 1.0
    
    return is_orientation_zero_except_w


class ExitRosNodeError(StopIteration):
    def __init__(self, result):
        super().__init__()
        self.result = result


class PickScrewdriverSimEnv(Node):
    def __init__(self, agent: BaseAgent, num_restarts: int = 0):
        super().__init__('pick_screwdriver_simulation')
        self.agent = agent

        self.get_logger().info(f"PickScrewdriver node started (simulation, n_restarts={num_restarts})")

        self.image_subscriber = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            1)
        self.bridge = CvBridge()
        self.max_n_iters = num_restarts
        self.current_iter = 0

        self.current_iter = 0
        self.eval_results = []
        
        self.publish_timer = self.create_timer(0.1, self.execution_loop)

        self.isaac_gripper_publisher = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 1)

        self.current_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.current_pose_callback,
            1)
        self.target_pose_publisher = self.create_publisher(
            PoseStamped, '/target_frame_raw', 1
        )
        self.random_move_screwdriver = self.create_publisher(Twist, '/respawn', 1)

        self.current_pose_msg = PoseStamped()
        self.current_pose_msg.header.frame_id = 'link_base'

        self.step = 0
        
        self.image  = None
        self.action = None

        # 60 steps for sim 90 for real env
        self.max_episode_duration_s = 60
        self.observation_pose_msg = None
        self.publisher_joint_init = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 1)

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.restart()

    def restart(self):
        """
        Returns robot and screwdriver into their initial position
        """
        self.step = 0
        self._spawn_screwdriver()
        self._open_gripper()
        self._init_joints()
        self.episode_start_time = time.time()

    def is_success_achieved(self) -> bool:
        """
        Checks if screwdriver is picker up
        """
        screwdriver_target_tf = self._get_transform(
            'base_link', 'pick_target'
        )
        screwdriver_pose = rnp.numpify(screwdriver_target_tf.transform)
        screwdriver_center = screwdriver_pose @ np.array([0., 0., 0., 1.])
        screwdriver_center = screwdriver_center[:-1] / screwdriver_center[-1]
        screwdriver_z = screwdriver_center[-1]
        print(screwdriver_z)
        if screwdriver_z > 0.25:
            return True
        else:
            return False
    
    def _spawn_screwdriver(self):
        twist_msg = Twist()
        self.random_move_screwdriver.publish(twist_msg)

    def _open_gripper(self):
        gripper_state = Float64MultiArray()
        gripper_state.data = [0.0]
        self.isaac_gripper_publisher.publish(gripper_state)
    
    def _close_gripper(self):
        gripper_state = Float64MultiArray()
        gripper_state.data = [-0.01]
        self.isaac_gripper_publisher.publish(gripper_state)

    def _init_joints(self):
        self.joint_state = JointTrajectory()
        self.joint_names = ['joint1', 'joint2',
                            'joint3', 'joint4', 'joint5', 'joint6']

        point = JointTrajectoryPoint()
        point.positions = [0.00148, 0.06095, 1.164, -0.00033, 1.122, -0.00093]
        point.time_from_start.sec = 5
        point.time_from_start.nanosec = 0

        self.joint_state.points = [point]
        self.joint_state.joint_names = self.joint_names
           
        # move arm to init pose
        call_ros2_service('joint_trajectory_controller',
                              'cartesian_motion_controller')
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.publisher_joint_init.publish(self.joint_state)
        time.sleep(8)
        call_ros2_service('cartesian_motion_controller', 'joint_trajectory_controller')

    def _get_transform(self, target_frame, source_frame):
        try:
            transform = self.tfBuffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to get transform: {e}")
            return None
    
    def current_pose_callback(self, msg):
        self.observation_pose_msg = msg

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (224, 224))
        self.image = cv_image
    
    def execution_loop(self):
        """
        The main cycle that is called by ROS timer
        """
        if self.observation_pose_msg is None:
            return
        
        if self.image is None:
            return
        
        base_gripper_tf = self._get_transform('link_base', 'gripper_base_link')
        if base_gripper_tf is None:
            return

        self.current_pose_msg.header.stamp = self.get_clock().now().to_msg()

        target_state = self.agent.step(self.image, self.observation_pose_msg.pose)
        if target_state is not None:
            self.current_pose_msg.pose = target_state.pose
            self.target_pose_publisher.publish(self.current_pose_msg)

            if target_state.grip_closed:
                self._close_gripper()
            else:
                self._open_gripper()

        self.step += 1
        self.get_logger().info(f"State: {target_state} | {self.agent.agent_state}")
        episode_duration_s = time.time() - self.episode_start_time
        is_finished = (episode_duration_s > self.max_episode_duration_s) or self.agent.is_finished()
        if not is_finished:
            return
        
        if self.is_success_achieved():
            self.eval_results.append(1)
            self.get_logger().info("Success")
        else:
            self.eval_results.append(0)
            self.get_logger().info("Fail")

        if self.current_iter < self.max_n_iters:
            self.current_iter += 1 
            self.restart()
            self.agent.restart()
        else:
            self.get_logger().info("Finished")
            self.get_logger().info(f"Success rate = {sum(self.eval_results) / len(self.eval_results)}")
            self.get_logger().info(str(self.eval_results))
            raise ExitRosNodeError(self.eval_results)
