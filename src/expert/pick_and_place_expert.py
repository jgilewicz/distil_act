import mink
import numpy as np

from env.pick_and_place_env import PickAndPlaceEnvironment
from expert.base import _JOINT_NAMES


class PickAndPlaceExpert:
    def __init__(
        self,
        env: PickAndPlaceEnvironment,
        dt: float = 0.002,
        solver: str = "daqp",
        max_iters: int = 20,
        ik_pos_threshold: float = 5e-3,
        approach_height: float = 0.2,
        gripper_open_threshold: float = 0.05,
        grasp_clearance: float = 0.02,
    ) -> None:
        self.env = env
        self.dt = dt
        self.solver = solver
        self.max_iters = max_iters
        self.ik_pos_threshold = ik_pos_threshold
        self.approach_height = approach_height
        self.gripper_open_threshold = gripper_open_threshold
        self.grasp_clearance = grasp_clearance

        self.dof_ids = np.array(
            [env.model.jnt_qposadr[env.model.joint(name).id] for name in _JOINT_NAMES]
        )
        self.actuator_ids = np.array(
            [env.model.actuator(name).id for name in _JOINT_NAMES]
        )

        self.gripper_actuator_id = env.model.actuator("robot_gripper").id
        self.gripper_open_ctrl = env.model.joint("robot_gripper").range[0]
        self.gripper_close_ctrl = env.model.joint("robot_gripper").range[1]
        self.gripper_qpos_adr = env.model.jnt_qposadr[
            env.model.joint("robot_gripper").id
        ]
        self.ee_id = env.model.body("robot_gripper_moving_finger").id

        self.configuration = mink.Configuration(env.model)

        self.ee_task = mink.FrameTask(
            frame_name="robot_gripper_moving_finger",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        # gripper's reach axis is local -X; rotating -90deg about Y points that
        # axis at world -Z, i.e. straight down, perpendicular to the tabletop.
        self.gripper_down_rotation = mink.SO3.from_y_radians(-np.pi / 2)

        self.limits = [
            mink.ConfigurationLimit(model=env.model),
        ]

        self.phase = "approach"

    def reset(self) -> None:
        self.phase = "approach"

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        box_pos = obs[-6:-3]
        approach_pos = box_pos + np.array([0.0, 0.0, self.approach_height])

        if self.phase == "approach":
            ee_pos = self.env.data.xpos[self.ee_id]
            gripper_qpos = self.env.data.qpos[self.gripper_qpos_adr]
            ee_converged = (
                np.linalg.norm(ee_pos - approach_pos) <= self.ik_pos_threshold
            )
            gripper_open = (
                abs(gripper_qpos - self.gripper_open_ctrl)
                <= self.gripper_open_threshold
            )
            if ee_converged and gripper_open:
                self.phase = "descend"

        if self.phase == "approach":
            target_pos = approach_pos
            gripper_ctrl = self.gripper_open_ctrl
        else:
            target_pos = box_pos + np.array([0.0, 0.0, self.grasp_clearance])
            gripper_ctrl = self.gripper_close_ctrl

        self.configuration.update(self.env.data.qpos)
        self.ee_task.set_target(
            mink.SE3.from_rotation_and_translation(
                self.gripper_down_rotation, target_pos
            )
        )

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
        action[self.gripper_actuator_id] = gripper_ctrl
        return action
