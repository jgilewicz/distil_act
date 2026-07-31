import argparse
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from env.base import Environment
from env.pick_and_place_env import PickAndPlaceEnvironment
from env.reach_env import ReachEnvironment
from expert.base import Expert
from expert.pick_and_place_expert import PickAndPlaceExpert
from expert.reach_expert import ReachExpert
from renderer.renderer import SceneRenderer
from dataset.dataset_manager import DatasetManager
from utils.logger import Logger
from utils.config import load_config
from utils.hub import push_dataset

SHOW_VIEWER = os.environ.get("SHOW_VIEWER", "true").lower() != "false"


@dataclass
class TaskSpec:
    env_cls: type[Environment]
    expert_cls: type[Expert]
    final_dist: Callable[[Environment, np.ndarray], float]
    success_verb: str


def _reach_final_dist(env: ReachEnvironment, obs: np.ndarray) -> float:
    return float(np.linalg.norm(env.data.xpos[env.ee_id] - obs[-3:]))


def _pick_and_place_final_dist(env: PickAndPlaceEnvironment, obs: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            env.data.xpos[env.box_id] - env.data.mocap_pos[env.place_target_mocap_id]
        )
    )


TASKS: dict[str, TaskSpec] = {
    "reach": TaskSpec(ReachEnvironment, ReachExpert, _reach_final_dist, "reached"),
    "pick_and_place": TaskSpec(
        PickAndPlaceEnvironment,
        PickAndPlaceExpert,
        _pick_and_place_final_dist,
        "placed",
    ),
}


def collect_episode(
    env: Environment,
    expert: Expert,
    renderer: SceneRenderer,
    dataset: DatasetManager,
    task: TaskSpec,
    episode_idx: int,
    n_steps: int,
    log: Logger,
) -> tuple[float, bool]:
    obs = env.reset()
    expert.reset()

    recorder = dataset.create_new_episode()
    terminated = False
    try:
        for _ in range(n_steps):
            action = expert.compute_action(obs)
            obs, terminated, frames = renderer.render_step(action)
            stacked = np.stack([frames[cam] for cam in renderer.camera_names])
            recorder.append_step(
                frames=stacked,
                joint_pos=env.data.qpos.copy().astype("float32"),
                timestamp=env.data.time,
            )
            if terminated:
                break
    finally:
        recorder.close()

    final_dist = task.final_dist(env, obs)

    if not terminated:
        dataset.discard_last_episode(recorder)
        log.info(
            f"Episode {episode_idx:3d} | truncated (discarded) | final dist {final_dist:.4f} m"
        )
    else:
        log.info(
            f"Episode {episode_idx:3d} | {task.success_verb} | final dist {final_dist:.4f} m"
        )

    return final_dist, terminated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    args = parser.parse_args()
    task = TASKS[args.task]

    cfg = load_config()
    tcfg = cfg["collect"]["tasks"][args.task]
    col = tcfg["collection"]

    log = Logger(col["log_file"])
    log.info(
        f"Starting {args.task} data collection — {col['n_episodes']} successful episodes → {col['dataset_dir']}"
    )
    n_cameras = len(col["render_cameras"])
    log.info(
        f"viewer: {'on' if SHOW_VIEWER else 'off'} | steps/episode: {col['n_steps']} | cameras: {col['render_cameras']}"
    )

    env = task.env_cls(**tcfg["env"])
    expert = task.expert_cls(env, **tcfg["expert"])
    dataset = DatasetManager(
        root_dir=col["dataset_dir"],
        img_shape=(n_cameras, cfg["renderer"]["height"], cfg["renderer"]["width"], 3),
        joint_dim=env.model.nq,
    )

    distances = []
    n_target = col["n_episodes"]
    attempt = 0
    with SceneRenderer(
        env,
        height=cfg["renderer"]["height"],
        width=cfg["renderer"]["width"],
        camera_list=col["render_cameras"],
        show_viewer=SHOW_VIEWER,
    ) as renderer:
        while len(distances) < n_target:
            dist, success = collect_episode(
                env,
                expert,
                renderer,
                dataset,
                task,
                attempt,
                col["n_steps"],
                log,
            )
            attempt += 1
            if success:
                distances.append(dist)
            log.info(f"Progress: {len(distances)}/{n_target} successful episodes")

    success_rate = len(distances) / attempt
    dataset.save_metadata(
        {
            "n_episodes": n_target,
            "n_attempts": attempt,
            "success_rate": success_rate,
            "n_steps": col["n_steps"],
            "cameras": col["render_cameras"],
            "img_shape": [
                n_cameras,
                cfg["renderer"]["height"],
                cfg["renderer"]["width"],
                3,
            ],
            "joint_dim": env.model.nq,
            "mean_final_dist_m": float(np.mean(distances)),
        }
    )

    log.info(
        f"Done. {n_target} episodes collected in {attempt} attempts ({success_rate:.1%} success rate)"
    )
    log.info(f"Mean final distance: {np.mean(distances):.4f} m")
    log.info(f"Dataset saved to: {os.path.abspath(col['dataset_dir'])}")

    if col["hub"]["auto_push"]:
        push_dataset(col["dataset_dir"], col["hub"]["repo_id"], log)


if __name__ == "__main__":
    main()
