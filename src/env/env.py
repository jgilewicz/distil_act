import mujoco
import mujoco.viewer
from robot_descriptions import low_cost_robot_arm_mj_description
import numpy as np

SCENE_XML_PATH = "models/reach_scene.xml"

TARGET_XY_RANGE = (-0.16, 0.18)
TARGET_Z_RANGE = (0.03, 0.20)


def build_model() -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(SCENE_XML_PATH)
    robot_spec = mujoco.MjSpec.from_file(low_cost_robot_arm_mj_description.MJCF_PATH)

    attach_frame = scene_spec.worldbody.add_frame()
    attach_frame.attach_body(robot_spec.worldbody.first_body(), prefix="robot_")

    return scene_spec.compile()


def sample_target_position(rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(*TARGET_XY_RANGE)
    y = rng.uniform(*TARGET_XY_RANGE)
    z = rng.uniform(*TARGET_Z_RANGE)

    return np.array([x, y, z])


class ReachEnvironment:
    def __init__(
        self, ee_body_name: str = "robot_gripper_moving_finger", seed: int = 0
    ) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

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

    def _compute_reward(self) -> float:
        ee_pos = self.data.xpos[self.ee_id]
        target_pos = self.data.mocap_pos[self.target_mocap_id]

        distance = np.linalg.norm(ee_pos - target_pos)

        return -distance.astype(float)

    def step(self, action: np.ndarray) -> tuple:
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()

        terminated = False  # no usage in reach for now
        info = {}

        return obs, reward, terminated, info

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)

        target_pos = sample_target_position(self.rng)
        self.data.mocap_pos[self.target_mocap_id] = target_pos

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()


if __name__ == "__main__":
    env = ReachEnvironment()
    obs = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            for i in range(10):
                action = env.rng.uniform(-0.3, 0.3, size=env.model.nu)
                obs, reward, terminated, info = env.step(action)
                print(f"Step {i}: reward={reward:.4f}")
                viewer.sync()
