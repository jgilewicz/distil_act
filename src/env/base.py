from abc import ABC, abstractmethod

import mujoco
import numpy as np
from robot_descriptions import piper_mj_description


def build_model(
    scene_xml_path: str, gripper_kp: float = 200.0, gripper_kv: float = 11.0
) -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(scene_xml_path)
    robot_spec = mujoco.MjSpec.from_file(piper_mj_description.MJCF_PATH)

    gripper_act = robot_spec.actuator("gripper")
    gripper_act.gainprm[0] = gripper_kp
    gripper_act.biasprm[1] = -gripper_kp
    gripper_act.biasprm[2] = -gripper_kv

    attach_frame = scene_spec.worldbody.add_frame()
    attach_frame.attach_body(robot_spec.worldbody.first_body(), prefix="robot_")

    # link6 is the wrist link just before the gripper fingers - stable
    # reference point that doesn't move as the gripper opens/closes
    gripper_body = next((b for b in scene_spec.bodies if b.name == "robot_link6"), None)

    if gripper_body is not None:
        cam_frame = gripper_body.add_frame()
        cam_frame.add_camera(
            name="ego_cam", pos=[0, 0, 0.06], xyaxes=[-1, 0, 0, 0, 1, 0], fovy=75
        )

    return scene_spec.compile()


class Environment(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]: ...
