import os

import numpy as np

from env.env import ReachEnvironment
from expert.base import Expert
from expert.reach_expert import ReachExpert
from renderer.renderer import SceneRenderer
from dataset.dataset_manager import DatasetManager
from utils.logger import Logger
from utils.config import load_config

SHOW_VIEWER = os.environ.get("SHOW_VIEWER", "true").lower() != "false"
CONFIG_PATH = "config.yaml"


def collect_episode(
    env: ReachEnvironment,
    expert: Expert,
    renderer: SceneRenderer,
    dataset: DatasetManager,
    episode_idx: int,
    n_steps: int,
    record_camera: str,
    log: Logger,
) -> float:
    obs = env.reset()
    expert.reset()

    recorder = dataset.create_new_episode()
    terminated = False
    try:
        for _ in range(n_steps):
            action = expert.compute_action(obs)
            obs, terminated, frames = renderer.render_step(action)
            recorder.append_step(
                frame=frames[record_camera],
                joint_pos=env.data.qpos.copy().astype("float32"),
                timestamp=env.data.time,
            )
            if terminated:
                break
    finally:
        recorder.close()

    final_dist = float(np.linalg.norm(env.data.xpos[env.ee_id] - obs[-3:]))
    outcome = "reached" if terminated else "truncated"
    log.info(f"Episode {episode_idx:3d} | {outcome} | final dist {final_dist:.4f} m")
    return final_dist


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    col = cfg["collection"]

    log = Logger(col["log_file"])
    log.info(
        f"Starting data collection — {col['n_episodes']} episodes → {col['dataset_dir']}"
    )
    log.info(
        f"viewer: {'on' if SHOW_VIEWER else 'off'} | steps/episode: {col['n_steps']} | camera: {col['record_camera']}"
    )

    env = ReachEnvironment(**cfg["env"])
    expert = ReachExpert(env, **cfg["expert"])
    dataset = DatasetManager(
        root_dir=col["dataset_dir"],
        img_shape=(cfg["renderer"]["height"], cfg["renderer"]["width"], 3),
        joint_dim=env.model.nq,
    )

    distances = []
    with SceneRenderer(
        env,
        height=cfg["renderer"]["height"],
        width=cfg["renderer"]["width"],
        camera_list=col["render_cameras"],
        show_viewer=SHOW_VIEWER,
    ) as renderer:
        for i in range(col["n_episodes"]):
            dist = collect_episode(
                env,
                expert,
                renderer,
                dataset,
                i,
                col["n_steps"],
                col["record_camera"],
                log,
            )
            distances.append(dist)

    dataset.save_metadata(
        {
            "n_episodes": col["n_episodes"],
            "n_steps": col["n_steps"],
            "record_camera": col["record_camera"],
            "img_shape": [cfg["renderer"]["height"], cfg["renderer"]["width"], 3],
            "joint_dim": env.model.nq,
            "mean_final_dist_m": float(np.mean(distances)),
        }
    )

    log.info(f"Done. Mean final distance: {np.mean(distances):.4f} m")
    log.info(f"Dataset saved to: {os.path.abspath(col['dataset_dir'])}")


if __name__ == "__main__":
    main()
