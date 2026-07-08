import os

import numpy as np

from env.env import ReachEnvironment
from expert.base import Expert
from expert.reach_expert import ReachExpert
from renderer.renderer import SceneRenderer
from dataset.dataset_manager import DatasetManager
from utils.logger import Logger
from utils.config import load_config
from utils.hub import push_dataset

SHOW_VIEWER = os.environ.get("SHOW_VIEWER", "true").lower() != "false"


def collect_episode(
    env: ReachEnvironment,
    expert: Expert,
    renderer: SceneRenderer,
    dataset: DatasetManager,
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

    final_dist = float(np.linalg.norm(env.data.xpos[env.ee_id] - obs[-3:]))

    if not terminated:
        dataset.discard_last_episode(recorder)
        log.info(
            f"Episode {episode_idx:3d} | truncated (discarded) | final dist {final_dist:.4f} m"
        )
    else:
        log.info(f"Episode {episode_idx:3d} | reached | final dist {final_dist:.4f} m")

    return final_dist, terminated


def main() -> None:
    cfg = load_config()
    col = cfg["collection"]

    log = Logger(col["log_file"])
    log.info(
        f"Starting data collection — {col['n_episodes']} successful episodes → {col['dataset_dir']}"
    )
    n_cameras = len(col["render_cameras"])
    log.info(
        f"viewer: {'on' if SHOW_VIEWER else 'off'} | steps/episode: {col['n_steps']} | cameras: {col['render_cameras']}"
    )

    env = ReachEnvironment(**cfg["env"])
    expert = ReachExpert(env, **cfg["expert"])
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
