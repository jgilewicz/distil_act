import mink
import numpy as np

from env.pick_and_place_env import PickAndPlaceEnvironment
from expert.base import Expert
from expert.base import _JOINT_NAMES


class PickAndPlaceExpert(Expert):
    GRIPPER_OPEN = -1.6
    GRIPPER_CLOSED = 0.032
    # static-finger frame origin to finger-pad center along the reach axis
    # (local -X), measured from the MJCF finger geometry
    PAD_OFFSET = 0.063
    # static pad sits ~8.5mm from the frame origin in world +y; shift the
    # target so the pad clears the box face (IK leaves ~2.5mm y overshoot)
    # and the closing finger pushes the box into the pinch instead of the
    # pad landing on the box edge
    STRADDLE_OFFSET = 0.008
    # pinch slightly above the box center: squeezing below center pops the
    # box up and out of the grip
    GRASP_Z_OFFSET = 0.004
    APPROACH_HEIGHT = 0.06
    PLACE_CLEARANCE = 0.02
    PHASE_POS_THRESHOLD = 0.01
    GRIPPER_OPEN_THRESHOLD = -1.4
    # the box blocks the finger from reaching the closed setpoint, so wait
    # a fixed dwell that covers free close + servo-follow squeeze
    GRASP_HOLD_STEPS = 700
    # two-stage close: stepping ctrl straight to closed slams the finger
    # into the box at ~4.7 rad/s (500N contact spikes, the finger bounces
    # off). free-close to just short of contact (~-0.27), then advance the
    # setpoint delta ahead of the measured joint: approach speed ~kp*d/kd
    # and squeeze torque capped at kp*d once the box blocks the finger
    GRIP_PRECLOSE = -0.5
    GRIP_FOLLOW_DELTA = 0.03

    def __init__(
        self,
        env: PickAndPlaceEnvironment,
        dt: float = 0.02,
        solver: str = "daqp",
        max_iters: int = 20,
        ik_pos_threshold: float = 5e-3,
    ) -> None:
        self.env = env
        self.dt = dt
        self.solver = solver
        self.max_iters = max_iters
        self.ik_pos_threshold = ik_pos_threshold

        joint_ids = np.array([env.model.joint(name).id for name in _JOINT_NAMES])
        self.dof_ids = env.model.jnt_qposadr[joint_ids]
        self.actuator_ids = np.array(
            [env.model.actuator(name).id for name in _JOINT_NAMES]
        )
        self.gripper_actuator_id = env.model.actuator("robot_gripper").id
        self.gripper_qpos_adr = env.model.jnt_qposadr[
            env.model.joint("robot_gripper").id
        ]
        self.ee_body_id = env.model.body("robot_gripper_static_finger").id

        self.configuration = mink.Configuration(env.model)

        # position in meters vs orientation in radians: at 1:1 the rotation
        # term dominates and leaves ~5mm positional error, too much for a
        # 2cm box
        self.ee_task = mink.FrameTask(
            frame_name="robot_gripper_static_finger",
            frame_type="body",
            position_cost=5.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        # reach axis is local -X; -90deg about Y points it at world -Z (down)
        self.rotation = mink.SO3.from_y_radians(-np.pi / 2)

        self.limits = [
            mink.ConfigurationLimit(model=env.model),
        ]

        self.phase = "approach"
        self.grasp_steps = 0
        self.grasp_target = np.zeros(3)

    def reset(self) -> None:
        self.phase = "approach"
        self.grasp_steps = 0
        self.grasp_target = np.zeros(3)

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        box_pos = obs[-6:-3]
        place_pos = obs[-3:]
        self._advance_phase(box_pos, place_pos)
        target_pos, gripper_ctrl = self._phase_target(box_pos, place_pos)
        if gripper_ctrl == self.GRIPPER_CLOSED:
            gripper_ctrl = self._close_cmd()
        action = self._solve_arm(target_pos)
        action[self.gripper_actuator_id] = gripper_ctrl
        return action

    def _close_cmd(self) -> float:
        gripper_q = self.env.data.qpos[self.gripper_qpos_adr]
        if gripper_q < self.GRIP_PRECLOSE - 0.05:
            return self.GRIP_PRECLOSE
        return min(gripper_q + self.GRIP_FOLLOW_DELTA, self.GRIPPER_CLOSED)

    def _phase_target(
        self, box_pos: np.ndarray, place_pos: np.ndarray
    ) -> tuple[np.ndarray, float]:
        above = np.array([0.0, 0.0, self.PAD_OFFSET + self.APPROACH_HEIGHT])
        drop = np.array([0.0, 0.0, self.PAD_OFFSET + self.PLACE_CLEARANCE])
        # z from the known resting height, not the live box z: the box can
        # get pressed into the table softness and drag the target with it
        descend_target = np.array(
            [
                box_pos[0],
                box_pos[1] + self.STRADDLE_OFFSET,
                self.env.box_height + self.GRASP_Z_OFFSET + self.PAD_OFFSET,
            ]
        )
        targets: dict[str, tuple[np.ndarray, float]] = {
            "approach": (box_pos + above, self.GRIPPER_OPEN),
            "descend": (descend_target, self.GRIPPER_OPEN),
            "grasp": (self.grasp_target, self.GRIPPER_CLOSED),
            "transport": (place_pos + above, self.GRIPPER_CLOSED),
            "lower": (place_pos + drop, self.GRIPPER_CLOSED),
            "release": (place_pos + drop, self.GRIPPER_OPEN),
        }
        return targets[self.phase]

    def _advance_phase(self, box_pos: np.ndarray, place_pos: np.ndarray) -> None:
        target_pos, _ = self._phase_target(box_pos, place_pos)
        ee_pos = self.env.data.xpos[self.ee_body_id]
        reached = np.linalg.norm(ee_pos - target_pos) < self.PHASE_POS_THRESHOLD

        if self.phase == "approach":
            gripper_open = (
                self.env.data.qpos[self.gripper_qpos_adr] < self.GRIPPER_OPEN_THRESHOLD
            )
            if reached and gripper_open:
                self.phase = "descend"
        elif self.phase == "descend":
            if reached:
                # freeze the target: tracking the live box position while
                # closing chases the box as contact nudges it
                self.grasp_target = target_pos.copy()
                self.phase = "grasp"
        elif self.phase == "grasp":
            self.grasp_steps += 1
            if self.grasp_steps >= self.GRASP_HOLD_STEPS:
                self.phase = "transport"
        elif self.phase == "transport":
            if reached:
                self.phase = "lower"
        elif self.phase == "lower":
            if reached:
                self.phase = "release"

    def _solve_arm(self, target_pos: np.ndarray) -> np.ndarray:
        self.configuration.update(self.env.data.qpos)
        target = mink.SE3.from_translation(target_pos) @ mink.SE3.from_rotation(
            self.rotation
        )
        self.ee_task.set_target(target)

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
