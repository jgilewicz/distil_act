import mink
import numpy as np

from env.env import ReachEnvironment
from expert.base import Expert

_JOINT_NAMES = [
    "robot_base_rotation",
    "robot_pitch",
    "robot_elbow",
    "robot_wrist_pitch",
    "robot_wrist_roll",
    "robot_gripper",
]


class ReachExpert(Expert):
    def __init__(
        self,
        env: ReachEnvironment,
        dt: float = 0.002,
        solver: str = "daqp",
        max_iters: int = 20,
        ik_pos_threshold: float = 5e-3,
    ) -> None:
        self.env = env
        self.dt = dt
        self.solver = solver
        self.max_iters = max_iters
        self.ik_pos_threshold = ik_pos_threshold

        self.dof_ids = np.array([env.model.joint(name).id for name in _JOINT_NAMES])
        self.actuator_ids = np.array(
            [env.model.actuator(name).id for name in _JOINT_NAMES]
        )

        self.configuration = mink.Configuration(env.model)

        self.ee_task = mink.FrameTask(
            frame_name="robot_gripper_moving_finger",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )

        self.limits = [
            mink.ConfigurationLimit(model=env.model),
        ]

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        target_pos = obs[-3:]
        self.configuration.update(self.env.data.qpos)
        self.ee_task.set_target(mink.SE3.from_translation(target_pos))

        for _ in range(self.max_iters):
            vel = mink.solve_ik(
                self.configuration,
                [self.ee_task],
                self.dt,
                self.solver,
                limits=self.limits,
                damping=1e-5,
            )
            self.configuration.integrate_inplace(vel, self.dt)

            err = self.ee_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) <= self.ik_pos_threshold:
                break

        action = np.zeros(self.env.model.nu)
        action[self.actuator_ids] = self.configuration.q[self.dof_ids]
        return action


if __name__ == "__main__":
    from utils.logger import Logger
    from utils.config import load_config

    cfg = load_config("config.yaml")
    log = Logger("logs/expert_test.log")

    env = ReachEnvironment(**cfg["env"])
    obs = env.reset()
    expert = ReachExpert(env, **cfg["expert"])

    log.info(f"target: {obs[-3:]}")

    for step in range(300):
        action = expert.compute_action(obs)
        obs, terminated = env.step(action)
        if terminated:
            log.info(f"reached at step {step + 1}")
            break
        if (step + 1) % 50 == 0:
            dist = np.linalg.norm(env.data.xpos[env.ee_id] - obs[-3:])
            log.info(f"step {step + 1:3d} | dist {dist:.4f} m")
