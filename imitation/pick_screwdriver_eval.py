#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import ros2_numpy as rnp
import numpy as np

from common.inference import Imitation
from common.utils import plot_action_trajectory
from ros2_numpy_tf import numpy2ros, ros2numpy

from enum import Enum
import transforms3d as t3d
from pathlib import Path
import subprocess
import re
import time
import cv2
import os
import tf2_ros


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



class OperationState(Enum):
    IDLE = 0
    INFERENCE = 1
    GO_CLOSE = 2
    CLOSE_GRIPPER = 3
    PICK_UP = 4
    OPEN_GRIPPER = 5
    END = 6


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


class PickScrewdriver(Node):

    def __init__(self, sim, n_iters: int):
        super().__init__('pick_screwdriver')
        self.get_logger().info(f"PickScrewdriver node started (sim = {sim}, n_iters={n_iters})")

        self.image_subscriber = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            1)
        self.bridge = CvBridge()
        self.max_n_iters = n_iters
        self.current_iter = 0
        self.eval_results = []
        
        self.publish_timer = self.create_timer(0.1, self.publish_pose)

        self.publisher_speed_limiter = self.create_publisher(
            PoseStamped, '/target_frame_raw', 1)

        self.isaac_gripper_publisher = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 1)

        self.current_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.current_pose_callback,
            1)
        self.current_pose_subscriber
        
        self.output_directory = Path("imitation/outputs/example")
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

        self.imitation = Imitation()
        
        self.current_pose_relativ = PoseStamped()
        self.current_pose = PoseStamped()
        self.step = 0
        
        self.image  = None
        self.action = None

        # 60 steps for sim 90 for real env
        self.max_episode_steps = 90

        self.sim = sim

        self.observation_pose = None
        self.observation_current_pose = PoseStamped()
        self.init_gripper()
        self.timer = time.time()

        self.publisher_joint_init = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 1)
        
        self.init_joints()

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.state = OperationState.IDLE
        self.observation_current_pose = PoseStamped()
        self.start_pose = None

        self.plotting_observations = []
        self.plotting_actions = []
    
    def spawn_screwdriver(self):
        self.random_move_screwdriver = self.create_publisher(Twist, '/respawn', 1)
        twist_msg = Twist()
        self.random_move_screwdriver.publish(twist_msg)

    def init_gripper(self):
        if self.sim:
            # on init open griper and move screwdriver on random position
            self.gripper_state = Float64MultiArray()
            self.gripper_state.data = [0.0]
            self.isaac_gripper_publisher.publish(self.gripper_state)
            self.spawn_screwdriver()
        else:
            self.lite6_gripper_publisher = self.create_publisher(Int32, '/gripper_switcher', 1)
            gripper_state_msg = Int32()
            gripper_state_msg.data = 1
            self.lite6_gripper_publisher.publish(gripper_state_msg)

            time.sleep(1)
            msg = Int32()
            msg.data = 2
            self.lite6_gripper_publisher.publish(msg)

    def init_joints(self):
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


    def get_transform(self, target_frame, source_frame):
        try:
            transform = self.tfBuffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to get transform: {e}")
            return None
    
    def current_pose_callback(self, msg):
        self.observation_current_pose = msg
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z

        x_ori = msg.pose.orientation.x
        y_ori = msg.pose.orientation.y
        z_ori = msg.pose.orientation.z
        w_ori = msg.pose.orientation.w

        quat = [w_ori, x_ori, y_ori, z_ori]
        euler_angle = t3d.euler.quat2euler(quat)

        self.observation_pose = [x_pos, y_pos, z_pos, euler_angle[0], euler_angle[1], euler_angle[2]]
        self.observation_current_pose = msg

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (224, 224))
        self.image = cv_image
          
    def publish_pose(self):

        if self.observation_pose is None:
            return
        
        if self.image is None:
            return

        base_gripper_tf = self.get_transform('link_base', 'gripper_base_link')
        if base_gripper_tf is None:
            return
        
        self.get_logger().info(f'========================={self.state}=========================')

        self.current_pose.header.stamp = self.get_clock().now().to_msg()
        self.current_pose.header.frame_id = 'link_base'

    
        if self.state == OperationState.IDLE:
            
            self.start_pose = self.observation_current_pose
            if self.start_pose  is not None:
                self.state = OperationState.INFERENCE
                self.timer = time.time()

        elif self.state == OperationState.INFERENCE:
            
            self.action = self.imitation.step(self.image, self.observation_pose)
            if self.action is None:
                return
            
            self.get_logger().info(f"Step {self.step}, Action: {self.action}")

            self.current_pose_relativ.header.stamp = self.get_clock().now().to_msg()
            self.current_pose_relativ.header.frame_id = 'gripper_link_base'
            self.current_pose_relativ.pose.position.x = (float(self.action[0]) / 1.5)
            self.current_pose_relativ.pose.position.y = (float(self.action[1]) / 1.0) 
            self.current_pose_relativ.pose.position.z = (float(self.action[2]) / 2.5)

            quat = t3d.euler.euler2quat((self.action[3] / 1.5), (self.action[4] / 6.5), (self.action[5] / 1.5))

            self.current_pose_relativ.pose.orientation.x = float(quat[1])
            self.current_pose_relativ.pose.orientation.y = float(quat[2])
            self.current_pose_relativ.pose.orientation.z = float(quat[3])
            self.current_pose_relativ.pose.orientation.w = float(quat[0])


            current_pose_target = ros2numpy(self.current_pose_relativ.pose)
            target_rotation = ros2numpy(self.observation_current_pose.pose) @ current_pose_target
            target_rotation = target_rotation[:3, :3]

            target_transform = ros2numpy(self.observation_current_pose.pose) @ current_pose_target

            self.plotting_actions.append(target_transform[:3, 3])
            self.plotting_observations.append(self.observation_pose[:3])

            target_transform[:3, :3] = target_rotation

            pose = numpy2ros(target_transform, Pose)
            self.start_pose.pose = pose
            self.current_pose.pose = pose

            if self.observation_current_pose.pose.position.z < 0.10:
                self.state = OperationState.OPEN_GRIPPER
                self.timer = time.time()
                self.get_logger().info('___________________________________ END ___________________________________')



            self.publisher_speed_limiter.publish(self.current_pose)

            self.step += 1

            if self.step > self.max_episode_steps:
                plot_action_trajectory(self.sim, self.plotting_observations, self.plotting_actions)
                self.state = OperationState.OPEN_GRIPPER
                self.timer = time.time()

        elif self.state == OperationState.OPEN_GRIPPER:
            # open later because make noise
            if not self.sim:
                msg = Int32()
                msg.data = 1
                self.lite6_gripper_publisher.publish(msg)

            if time.time() - self.timer > 1.5 * 3:
                self.state = OperationState.GO_CLOSE
                self.timer = time.time()

        elif self.state == OperationState.GO_CLOSE:
                self.observation_current_pose.pose.position.z = 0.085
                self.publisher_speed_limiter.publish(self.observation_current_pose)
                if time.time() - self.timer > 2.5 * 4:
                    self.state = OperationState.CLOSE_GRIPPER
                    self.timer  =time.time()
        
        elif self.state == OperationState.CLOSE_GRIPPER:
            if self.sim:
                self.gripper_state.data = [-0.01]
                self.isaac_gripper_publisher.publish(self.gripper_state)
            else:
                msg = Int32()
                msg.data = 0
                self.lite6_gripper_publisher.publish(msg)
            
            gripper_landing_limit_time_s = 5

            if time.time() -  self.timer > gripper_landing_limit_time_s:
                self.state = OperationState.PICK_UP
                self.timer = time.time()
        elif self.state == OperationState.PICK_UP:
                self.observation_current_pose.pose.position.z = 0.3
                self.publisher_speed_limiter.publish(self.observation_current_pose)

                screwdriver_target_tf = self.get_transform(
                    'base_link', 'pick_target'
                )
                screwdriver_pose = rnp.numpify(screwdriver_target_tf.transform)
                self.get_logger().info(f"Screwdriver: {screwdriver_pose}")
                if time.time() - self.timer > 10:
                    screwdriver_center = screwdriver_pose @ np.array([0., 0., 0., 1.])
                    screwdriver_center = screwdriver_center[:-1] / screwdriver_center[-1]
                    screwdriver_z = screwdriver_center[-1]
                    print("\n\n\n")
                    if screwdriver_z > 0.25:
                        print(screwdriver_center)
                        print("=================SUCCESS=================")
                        self.eval_results.append(1)
                    else:
                        print("=================FAIL=================")
                        self.eval_results.append(0)
                    print("\n\n\n")


                    self.state = OperationState.END
                    self.timer = time.time()

        elif self.state == OperationState.END:

            if time.time() - self.timer > 2.5:
                if not self.sim:
                    # shutdown gripper
                    msg = Int32()
                    msg.data = 2
                    self.lite6_gripper_publisher.publish(msg)

                self.spawn_screwdriver()
                self.init_gripper()
                self.init_joints()

                self.current_iter += 1
                if self.current_iter < self.max_n_iters:
                    self.get_logger().info(f"Launch new cycle...")
                    self.state = OperationState.INFERENCE
                    self.step = 0
                else:
                    print("Results")
                    print(np.mean(self.eval_results))
                    print(self.eval_results)
                    raise ExitRosNodeError(self.eval_results)


                #self.destroy_timer(self.publish_timer)
                #self.destroy_node()
                #rclpy.shutdown()
                #exit(0)

       
        if check_pose_stamped_values(self.current_pose_relativ):
            return

def main(args=None):

    import argparse
    parser = argparse.ArgumentParser(description='PickScrewdriver')
    parser.add_argument('--n_iters', type=int, default=5, help='Number of evaluation attempts')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    cmd_vel_publisher = PickScrewdriver(sim=True, n_iters=parsed_args.n_iters)
    try:
        rclpy.spin(cmd_vel_publisher)
    except RuntimeError as exc:
        pass
        #print(f"Success rate: {np.mean(exc.result)}")
        #print(exc.result)

    cmd_vel_publisher.get_logger().info(f"Finished publishing. Shutting down node...")
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
