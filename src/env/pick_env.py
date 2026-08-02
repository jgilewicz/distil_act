import mujoco
import numpy as np

from env.base import Environment, build_model


def sample_position(
    rng: np.random.Generator,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> np.ndarray:
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    return np.array([x, y])


class PickEnvironment(Environment):
    def __init__(
        self,
        scene_xml_path: str = "models/pick_scene.xml",
        box_x_range: tuple[float, float] = (-0.10, -0.06),
        box_y_range: tuple[float, float] = (0.13, 0.15),
        box_height: float = 0.025,
        # PickExpert's lift phase raises the tool (and grasped box) ~0.08m;
        # a lower bar avoids requiring the trajectory's exact final waypoint
        lift_height: float = 0.05,
        ee_body_name: str = "robot_link6",
        # collision-free ready pose for joint1-6 - mj_resetData's all-zero
        # qpos has the arm collide with the table in curobo's collision world
        ready_pose: tuple[float, ...] = (0.0, 1.57, -1.3485, 0.0, 0.0, 0.0),
        gripper_kp: float = 200.0,
        gripper_kv: float = 11.0,
        seed: int = 0,
    ) -> None:
        self.model = build_model(scene_xml_path, gripper_kp, gripper_kv)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.ready_pose = np.array(ready_pose)
        self.arm_qpos_adr = [
            self.model.joint(f"robot_joint{i}").qposadr[0] for i in range(1, 7)
        ]

        self.box_x_range = box_x_range
        self.box_y_range = box_y_range
        self.box_height = box_height
        self.lift_height = lift_height

        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )

        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint"
        )
        self.box_qpos_adr = self.model.jnt_qposadr[box_joint_id]
        self.box_dof_adr = self.model.jnt_dofadr[box_joint_id]

    def _get_obs(self) -> np.ndarray:
        box_pos = self.data.xpos[self.box_id].copy()
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos, qvel, box_pos])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        lifted = self.data.xpos[self.box_id][2] > self.box_height + self.lift_height
        return obs, lifted

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)

        for adr, pos in zip(self.arm_qpos_adr, self.ready_pose):
            self.data.qpos[adr] = pos

        box_xy = sample_position(self.rng, self.box_x_range, self.box_y_range)
        self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3] = [
            box_xy[0],
            box_xy[1],
            self.box_height,
        ]
        self.data.qpos[self.box_qpos_adr + 3 : self.box_qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.box_dof_adr : self.box_dof_adr + 6] = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
