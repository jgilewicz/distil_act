from abc import ABC, abstractmethod

import numpy as np

_JOINT_NAMES = [
    "robot_base_rotation",
    "robot_pitch",
    "robot_elbow",
    "robot_wrist_pitch",
    "robot_wrist_roll",
    "robot_gripper",
]

class Expert(ABC):
    @abstractmethod
    def compute_action(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None:
        pass
