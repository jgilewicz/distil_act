import mujoco
from robot_descriptions import piper_mj_description
import numpy as np

from env.base import Environment


def build_model(scene_xml_path: str) -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(scene_xml_path)
    robot_spec = mujoco.MjSpec.from_file(piper_mj_description.MJCF_PATH)

    attach_frame = scene_spec.worldbody.add_frame()
    attach_frame.attach_body(robot_spec.worldbody.first_body(), prefix="robot_")

    # link6 is the wrist link just before the gripper fingers - stable
    # reference point that doesn't move as the gripper opens/closes
    gripper_body = next(
        (b for b in scene_spec.bodies if b.name == "robot_link6"), None
    )

    if gripper_body is not None:
        cam_frame = gripper_body.add_frame()
        cam_frame.add_camera(
            name="ego_cam", pos=[0, 0, 0.05], xyaxes=[1, 0, 0, 0, 0, 1], fovy=75
        )

    return scene_spec.compile()


def sample_position(
    rng: np.random.Generator,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> np.ndarray:
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    return np.array([x, y])


class PickAndPlaceEnvironment(Environment):
    def __init__(
        self,
        scene_xml_path: str = "models/pick_and_place_scene.xml",
        box_x_range: tuple[float, float] = (-0.10, -0.06),
        box_y_range: tuple[float, float] = (0.13, 0.15),
        placement_x_range: tuple[float, float] = (0.06, 0.10),
        placement_y_range: tuple[float, float] = (0.13, 0.15),
        box_height: float = 0.01,
        placement_threshold: float = 1e-1,
        ee_body_name: str = "robot_link6",
        seed: int = 0,
    ) -> None:
        self.model = build_model(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.box_x_range = box_x_range
        self.box_y_range = box_y_range
        self.placement_x_range = placement_x_range
        self.placement_y_range = placement_y_range
        self.box_height = box_height
        self.placement_threshold = placement_threshold

        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )

        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint"
        )
        self.box_qpos_adr = self.model.jnt_qposadr[box_joint_id]
        self.box_dof_adr = self.model.jnt_dofadr[box_joint_id]

        place_target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "place_target"
        )
        self.place_target_mocap_id = self.model.body_mocapid[place_target_id]

    def _get_obs(self) -> np.ndarray:
        box_pos = self.data.xpos[self.box_id].copy()
        placement_pos = self.data.mocap_pos[self.place_target_mocap_id].copy()
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos, qvel, box_pos, placement_pos])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        dist = np.linalg.norm(
            self.data.xpos[self.box_id]
            - self.data.mocap_pos[self.place_target_mocap_id]
        )
        return obs, dist < self.placement_threshold

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)

        box_xy = sample_position(self.rng, self.box_x_range, self.box_y_range)
        self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3] = [
            box_xy[0],
            box_xy[1],
            self.box_height,
        ]
        self.data.qpos[self.box_qpos_adr + 3 : self.box_qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.box_dof_adr : self.box_dof_adr + 6] = 0

        placement_xy = sample_position(
            self.rng, self.placement_x_range, self.placement_y_range
        )
        self.data.mocap_pos[self.place_target_mocap_id] = [
            placement_xy[0],
            placement_xy[1],
            self.box_height,
        ]

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
