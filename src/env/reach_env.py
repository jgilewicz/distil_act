import mujoco
import numpy as np

from env.base import Environment, build_model


def sample_target_position(
    rng: np.random.Generator,
    target_x_range: tuple[float, float],
    target_y_range: tuple[float, float],
    target_z_range: tuple[float, float],
) -> np.ndarray:
    x = rng.uniform(*target_x_range)
    y = rng.uniform(*target_y_range)
    z = rng.uniform(*target_z_range)
    return np.array([x, y, z])


class ReachEnvironment(Environment):
    def __init__(
        self,
        scene_xml_path: str = "models/reach_scene.xml",
        target_x_range: tuple[float, float] = (0.12, 0.28),
        target_y_range: tuple[float, float] = (-0.15, 0.15),
        target_z_range: tuple[float, float] = (0.10, 0.20),
        reach_threshold: float = 1e-1,
        ee_body_name: str = "robot_link6",
        seed: int = 0,
    ) -> None:
        self.model = build_model(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.target_x_range = target_x_range
        self.target_y_range = target_y_range
        self.target_z_range = target_z_range
        self.reach_threshold = reach_threshold

        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )

        target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.target_mocap_id = self.model.body_mocapid[target_body_id]

    def _get_obs(self) -> np.ndarray:
        ee_pos = self.data.xpos[self.ee_id].copy()
        target_pos = self.data.mocap_pos[self.target_mocap_id].copy()
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos, qvel, ee_pos, target_pos])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        dist = np.linalg.norm(
            self.data.xpos[self.ee_id] - self.data.mocap_pos[self.target_mocap_id]
        )
        return obs, dist < self.reach_threshold

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        target_pos = sample_target_position(
            self.rng, self.target_x_range, self.target_y_range, self.target_z_range
        )
        self.data.mocap_pos[self.target_mocap_id] = target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
