import os

import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.robot_builder import RobotBuilder
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose, ToolPoseCriteria

from env.pick_env import PickEnvironment
from expert.base import Expert
from utils.logger import Logger

logger = Logger("logs/pick_expert.log")


class PickExpert(Expert):
    def __init__(
        self,
        env: PickEnvironment,
        urdf_path: str = "models/piper.urdf",
        asset_path: str = "models",
        config_path: str = "models/config.yaml",
        world_config_path: str = "models/pick.yaml",
        sphere_density: float = 1.0,
        coverage_weight: float = 1000.0,
        protrusion_weight: float = 10.0,
        compute_metrics: bool = False,
        prune_collisions: bool = True,
        num_collision_samples: int = 100,
        max_goalset: int = 10,
        num_warmup_iterations: int = 10,
        enable_graph: bool = True,
        close_steps: int = 250,
        device: str = "cuda:0",
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        device_cfg = DeviceCfg(device=self.device)

        # absolute path for curobot
        config_path = os.path.abspath(config_path)
        world_config_path = os.path.abspath(world_config_path)

        if not os.path.exists(config_path):
            # link6 is the parent of end effector links
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

        config = MotionPlannerCfg.create(
            robot=config_path,
            scene_model=world_config_path,
            device_cfg=device_cfg,
            max_goalset=max_goalset,
            use_cuda_graph=(device != "cpu"),
        )

        self.planner = MotionPlanner(config)
        self.planner.warmup(
            enable_graph=enable_graph, num_warmup_iterations=num_warmup_iterations
        )
        self.target_link = self.planner.tool_frames[0]

        # positions of joints - passed to the planner for the current state
        self.planner_qpos_idx = [
            env.model.joint(f"robot_{name}").qposadr[0]
            for name in self.planner.joint_names
        ]

        # actuator ids for actuator action pushing
        self.arm_actuator_ids = np.array(
            [
                env.model.actuator(f"robot_{name}").id
                for name in self.planner.joint_names
            ]
        )

        self.planner.update_tool_pose_criteria(
            {
                self.target_link: ToolPoseCriteria(
                    terminal_pose_axes_weight_factor=[1, 1, 1, 1, 1, 0],
                    non_terminal_pose_axes_weight_factor=[1, 1, 1, 0, 0, 0],
                    device_cfg=device_cfg,
                )
            }
        )
        self.planner.disable_link_collision(["base_link"])

        self.gripper_actuator_id = env.model.actuator("robot_gripper").id
        self.gripper_open = 0.035
        self.gripper_closed = 0.0
        self.close_steps = close_steps

        self.phases = [
            "approach",
            "grasp",
            "close",
            "lift",
        ]
        self.gripper_by_phase = {
            "approach": self.gripper_open,
            "grasp": self.gripper_open,
            "close": self.gripper_closed,
            "lift": self.gripper_closed,
        }

        self.trajectories = None
        self.phase_idx = 0
        self.step_idx = -1

    def reset(self) -> None:
        self.trajectories = None
        self.phase_idx = 0
        self.step_idx = -1

    def _calculate_trajectory(self, obs: np.ndarray) -> None:
        box_position = obs[-3:] + np.array([0, 0, 0.135])  # grasp offset

        grasp_pose = Pose(
            position=torch.tensor(
                np.array([box_position]), device=self.device, dtype=torch.float32
            ),
            quaternion=torch.tensor(
                np.array([[0.0, 1.0, 0.0, 0.0]]),
                device=self.device,
                dtype=torch.float32,
            ),
        )
        current_q = self.env.data.qpos[self.planner_qpos_idx]
        current_state = JointState.from_position(
            torch.tensor(
                np.array([current_q]), device=self.device, dtype=torch.float32
            ),
            joint_names=self.planner.joint_names,
        )

        result = self.planner.plan_grasp(
            GoalToolPose.from_poses({self.target_link: grasp_pose}, num_goalset=1),
            current_state=current_state,
            grasp_approach_offset=-0.08,
            grasp_lift_offset=-0.08,
        )

        if not result.success.any():
            logger.error(f"plan_grasp failed: {result.status}")

        n_joints = len(self.planner.joint_names)
        hold_current = np.array([current_q])

        def to_trajectory(
            joint_state: JointState | None, last_tstep: torch.Tensor | None
        ) -> np.ndarray:
            # not planned trajectory
            if joint_state is None:
                return hold_current
            traj = joint_state.position.reshape(-1, n_joints).detach().cpu().numpy()

            # cropping interpolated buffor
            if last_tstep is not None:
                traj = traj[: int(last_tstep.item()) + 1]
            return traj

        grasp_trajectory = to_trajectory(
            result.grasp_interpolated_trajectory, result.grasp_interpolated_last_tstep
        )

        close_trajectory = np.repeat(grasp_trajectory[-1:], self.close_steps, axis=0)

        self.trajectories = {
            "approach": to_trajectory(
                result.approach_interpolated_trajectory,
                result.approach_interpolated_last_tstep,
            ),
            "grasp": grasp_trajectory,
            "close": close_trajectory,
            "lift": to_trajectory(
                result.lift_interpolated_trajectory, result.lift_interpolated_last_tstep
            ),
        }

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        if self.trajectories is None:
            self._calculate_trajectory(obs)
        else:
            self.step_idx += 1

        if self.step_idx == -1:
            self.step_idx = 0

        phase = self.phases[self.phase_idx]
        trajectory = self.trajectories[phase]

        # advance to the next phase once the current one is exhausted
        while (
            self.step_idx >= len(trajectory) and self.phase_idx < len(self.phases) - 1
        ):
            self.phase_idx += 1
            self.step_idx = 0
            phase = self.phases[self.phase_idx]
            trajectory = self.trajectories[phase]

        # hold the final waypoint once the whole plan is exhausted
        index = min(self.step_idx, len(trajectory) - 1)

        action = np.zeros(self.env.model.nu)
        action[self.arm_actuator_ids] = trajectory[index]
        action[self.gripper_actuator_id] = self.gripper_by_phase[phase]
        return action
