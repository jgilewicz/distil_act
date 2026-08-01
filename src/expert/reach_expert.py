import os

import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.robot_builder import RobotBuilder
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose, ToolPoseCriteria

from env.reach_env import ReachEnvironment
from expert.base import Expert
from expert.base import _JOINT_NAMES
from utils.logger import Logger

logger = Logger("logs/reach_expert.log")


class ReachExpert(Expert):
    def __init__(
        self,
        env: ReachEnvironment,
        urdf_path: str = "models/piper.urdf",
        asset_path: str = "models",
        config_path: str = "models/config.yaml",
        sphere_density: float = 1.0,
        coverage_weight: float = 1000.0,
        protrusion_weight: float = 10.0,
        compute_metrics: bool = False,
        prune_collisions: bool = True,
        num_collision_samples: int = 100,
        num_seeds: int = 4,
        device: str = "cuda",
    ) -> None:
        self.env = env
        self.device = device
        device_cfg = DeviceCfg(device=device)

        # curobo resolves robot= paths against its own config dir, not cwd
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            # link7/link8 are fixed leaves of link6 - pick it explicitly
            # rather than relying on leaf auto-detection
            builder = RobotBuilder(
                urdf_path=urdf_path,
                asset_path=asset_path,
                tool_frames=["link6"],
                device_cfg=device_cfg,
            )
            builder.fit_collision_spheres(
                sphere_density=sphere_density,
                coverage_weight=coverage_weight,
                protrusion_weight=protrusion_weight,
                compute_metrics=compute_metrics,
            )
            builder.compute_collision_matrix(
                prune_collisions=prune_collisions, num_samples=num_collision_samples
            )
            config = builder.build()
            builder.save(config, config_path)

        config = InverseKinematicsCfg.create(
            robot=config_path,
            num_seeds=num_seeds,
            device_cfg=device_cfg,
            use_cuda_graph=(device != "cpu"),
        )
        self.ik = InverseKinematics(config)
        self.target_link = self.ik.tool_frames[0]

        # Pose() defaults orientation to identity, ~85deg off link6's rest
        # pose - zero the rotation axes for true position-only IK
        self.ik.update_tool_pose_criteria(
            {
                self.target_link: ToolPoseCriteria(
                    terminal_pose_axes_weight_factor=[1, 1, 1, 0, 0, 0],
                    non_terminal_pose_axes_weight_factor=[1, 1, 1, 0, 0, 0],
                    device_cfg=device_cfg,
                )
            }
        )

        # last entry in _JOINT_NAMES is the gripper - not commanded by IK
        arm_joint_names = _JOINT_NAMES[:-1]
        self.arm_actuator_ids = np.array(
            [env.model.actuator(name).id for name in arm_joint_names]
        )
        # ik.joint_names is unprefixed; _JOINT_NAMES is robot_-prefixed - map by name
        self.ik_joint_order = [
            self.ik.joint_names.index(name.removeprefix("robot_"))
            for name in arm_joint_names
        ]
        # seeds solve_pose with live state so IK doesn't jump between solutions
        self.ik_qpos_idx = [
            env.model.joint(f"robot_{name}").qposadr[0] for name in self.ik.joint_names
        ]

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        target_pos = obs[-3:]

        goal_pose = Pose(
            position=torch.tensor(
                np.array([target_pos]), device=self.device, dtype=torch.float32
            )
        )

        current_q = self.env.data.qpos[self.ik_qpos_idx]
        current_state = JointState.from_position(
            torch.tensor(
                np.array([current_q]), device=self.device, dtype=torch.float32
            ),
            joint_names=self.ik.joint_names,
        )

        result = self.ik.solve_pose(
            GoalToolPose.from_poses({self.target_link: goal_pose}, num_goalset=1),
            current_state=current_state,
        )

        if not bool(result.success.reshape(-1)[0]):
            logger.warning(
                f"curobo IK did not converge (pos_error="
                f"{result.position_error.reshape(-1)[0].item():.4f})"
            )

        solution = result.js_solution.position.reshape(-1).detach().cpu().numpy()
        if solution.shape[0] != len(self.ik.joint_names):
            raise RuntimeError(
                f"curobo IK returned {solution.shape[0]} joint values, expected "
                f"{len(self.ik.joint_names)} for joints {self.ik.joint_names}"
            )

        action = np.zeros(self.env.model.nu)
        action[self.arm_actuator_ids] = solution[self.ik_joint_order]
        return action
