from dataclasses import dataclass
import numpy as np

from enum import Enum
import time
from common.inference import Imitation

from modelling.ros2_numpy_tf import numpy2ros, ros2numpy
from geometry_msgs.msg import Pose
import transforms3d as t3d


@dataclass
class GripperState:
    pose: Pose
    grip_closed: bool

    def __str__(self) -> str:
        return f"Gripper<{self.pose.position.x:1.4f}, {self.pose.position.y:1.4f}, {self.pose.position.z:1.4f}| {self.grip_closed}>"
    

class BaseAgent:
    def is_finished(self) -> bool:
        raise NotImplementedError()
        
    def restart(self):
        raise NotImplementedError()

    def step(self, image: np.ndarray, observation_pose: Pose, **kwargs) -> GripperState | None:
        raise NotImplementedError()


class OperationState(Enum):
    IDLE = 0
    INFERENCE = 1
    GO_CLOSE = 2
    CLOSE_GRIPPER = 3
    PICK_UP = 4
    OPEN_GRIPPER = 5
    END = 6


class StateTimeredPickingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.agent_state = OperationState.IDLE
        self.imitation = Imitation()
        self.restart()

    def is_finished(self):
        return self.agent_state == OperationState.END
    
    def restart(self):
        self.agent_state = OperationState.INFERENCE
        self.step_count = 0
    
    @staticmethod
    def _prepare_pose_for_network(pose: Pose) -> np.ndarray:
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        euler_angle = t3d.euler.quat2euler(quat)
        return np.array([
            pose.position.x, pose.position.y, pose.position.z,
            euler_angle[0], euler_angle[1], euler_angle[2]
        ])
    
    def step(self, image: np.ndarray, observation_pose: Pose, **kwargs) -> GripperState | None:
        if self.agent_state == OperationState.IDLE:
            self.agent_state = OperationState.INFERENCE
            self.timer = time.time()
            return None

        elif self.agent_state == OperationState.INFERENCE:
            self.action = self.imitation.step(image, self._prepare_pose_for_network(observation_pose))

            if self.action is None:
                return None
            
            target_pose_relative = Pose()
            target_pose_relative.position.x = (float(self.action[0]) / 1.5)
            target_pose_relative.position.y = (float(self.action[1]) / 1.0) 
            target_pose_relative.position.z = (float(self.action[2]) / 2.5)

            quat = t3d.euler.euler2quat((self.action[3] / 1.5), (self.action[4] / 6.5), (self.action[5] / 1.5))
            target_pose_relative.orientation.x = float(quat[1])
            target_pose_relative.orientation.y = float(quat[2])
            target_pose_relative.orientation.z = float(quat[3])
            target_pose_relative.orientation.w = float(quat[0])

            target_pose_absolute = numpy2ros(
                ros2numpy(observation_pose) @ ros2numpy(target_pose_relative),
                Pose
            )

            if observation_pose.position.z < 0.12:
                self.agent_state = OperationState.OPEN_GRIPPER
                self.timer = time.time()

            if self.step_count > 90:
                self.agent_state = OperationState.OPEN_GRIPPER
                self.timer = time.time()

            self.step_count += 1

            return GripperState(
                target_pose_absolute, grip_closed=False
            )

        elif self.agent_state == OperationState.OPEN_GRIPPER:
            if time.time() - self.timer > 1.5 * 3:
                self.agent_state = OperationState.GO_CLOSE
                self.timer = time.time()
            return GripperState(observation_pose, grip_closed=False)

        elif self.agent_state == OperationState.GO_CLOSE:
            state = GripperState(observation_pose, grip_closed=False)
            state.pose.position.z = 0.085
            if time.time() - self.timer > 2.5 * 4:
                self.agent_state = OperationState.CLOSE_GRIPPER
                self.timer  = time.time()
            return state
        
        elif self.agent_state == OperationState.CLOSE_GRIPPER:
            gripper_landing_limit_time_s = 5

            if time.time() -  self.timer > gripper_landing_limit_time_s:
                self.agent_state = OperationState.PICK_UP
                self.timer = time.time()
            return GripperState(observation_pose, grip_closed=True)
            
        elif self.agent_state == OperationState.PICK_UP:
            state = GripperState(observation_pose, grip_closed=True)
            state.pose.position.z = 0.3

            if time.time() - self.timer > 10:
                self.agent_state = OperationState.END
                self.timer = time.time()
            return state

        elif self.agent_state == OperationState.END:
            return
