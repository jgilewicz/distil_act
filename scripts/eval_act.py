import os

import cv2
import torch

from algorithms.act_policy import ACT
from algorithms.chunking_buffer import ChunkingBuffer
from env.env import ReachEnvironment
from renderer.renderer import SceneRenderer
from utils.config import load_config
from utils.logger import Logger

logger = Logger("logs/eval.log")


def _render_current(renderer: SceneRenderer, cameras: list[str]) -> dict:
    frames = {}
    for cam in cameras:
        renderer.renderer.update_scene(renderer.env.data, camera=cam)
        rgb = renderer.renderer.render()
        frames[cam] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return frames


def _frames_to_tensor(
    frames: dict, cameras: list[str], device: torch.device
) -> torch.Tensor:
    imgs = [
        torch.from_numpy(frames[cam]).permute(2, 0, 1).float() / 255.0
        for cam in cameras
    ]
    return torch.stack(imgs).unsqueeze(0).to(device)  # (1, K, 3, H, W)


def eval():
    cfg = load_config("config.yaml")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    checkpoint = torch.load(
        "artifacts/act_model_final.pt", map_location=device, weights_only=True
    )
    norm_mean = checkpoint["norm_mean"].to(device)
    norm_std = checkpoint["norm_std"].to(device)

    t = cfg["training"]
    act = ACT(
        action_dim=t["action_dim"],
        embed_dim=t["embed_dim"],
        latent_dim=t["latent_dim"],
        joint_dim=t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=t["nhead"],
        num_layers=t["num_layers"],
        num_cameras=t["num_cameras"],
    )
    act.load_state_dict(checkpoint["model"])
    act = act.to(device)
    act.eval()
    logger.info("Model loaded")

    e = cfg["env"]
    env = ReachEnvironment(
        scene_xml_path=e["scene_xml_path"],
        target_x_range=tuple(e["target_x_range"]),
        target_y_range=tuple(e["target_y_range"]),
        target_z_range=tuple(e["target_z_range"]),
        reach_threshold=e["reach_threshold"],
        seed=1,
    )

    r = cfg["renderer"]
    cameras = cfg["collection"]["render_cameras"]
    chunk_size = t["chunk_size"]
    max_steps = cfg["collection"]["n_steps"]

    buffer = ChunkingBuffer(chunk_size=chunk_size, action_size=t["action_dim"])
    query_every = max(1, chunk_size // 5)

    os.makedirs("artifacts", exist_ok=True)
    video_path = "artifacts/eval.mp4"
    video_writer = None
    terminated = False

    with SceneRenderer(
        env, height=r["height"], width=r["width"], camera_list=cameras, show_viewer=True
    ) as renderer:
        obs = env.reset()
        buffer.reset()
        frames = _render_current(renderer, cameras)

        for step in range(max_steps):
            if renderer.viewer is not None and not renderer.viewer.is_running():
                break

            if step % query_every == 0:
                qpos = obs[:6]
                qpos_t = (
                    torch.from_numpy(qpos).float().to(device) - norm_mean
                ) / norm_std
                images_t = _frames_to_tensor(frames, cameras, device)

                with torch.inference_mode():
                    pred_norm = act(images_t, qpos_t.unsqueeze(0)).squeeze(
                        0
                    )  # (chunk_size, action_dim)

                buffer.add(pred_norm, step)

            action_norm = buffer.get_action(step)
            action = (action_norm * norm_std + norm_mean).cpu().numpy()

            obs, terminated = env.step(action)

            frames = _render_current(renderer, cameras)
            if renderer.viewer is not None and renderer.viewer.is_running():
                renderer.viewer.sync()

            overhead = frames.get("overhead_cam")
            if overhead is not None:
                if video_writer is None:
                    h, w = overhead.shape[:2]
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30,
                        (w, h),
                    )
                video_writer.write(overhead)

            if terminated:
                logger.info(f"Target reached at step {step + 1}")
                break

    if not terminated:
        logger.info(f"Did not reach target within {max_steps} steps")

    if video_writer is not None:
        video_writer.release()
        logger.info(f"Video saved to {video_path}")


if __name__ == "__main__":
    eval()
