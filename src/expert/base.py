from abc import ABC, abstractmethod

import numpy as np

_JOINT_NAMES = [
    "robot_joint1",
    "robot_joint2",
    "robot_joint3",
    "robot_joint4",
    "robot_joint5",
    "robot_joint6",
    "robot_joint7",
]


class Expert(ABC):
    @abstractmethod
    def compute_action(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None:
        pass
