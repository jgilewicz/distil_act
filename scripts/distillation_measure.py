import json
import os
import platform
import resource
import time

import cv2
import numpy as np
import torch

from algorithms.act_policy import ACT
from algorithms.chunking_buffer import ChunkingBuffer
from env.env import ReachEnvironment
from renderer.renderer import SceneRenderer
from utils.config import load_config
from utils.hub import ensure_checkpoint
from utils.logger import Logger


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def render_current(renderer, cameras):
    frames = {}
    for cam in cameras:
        renderer.renderer.update_scene(renderer.env.data, camera=cam)
        rgb = renderer.renderer.render()
        frames[cam] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return frames


def frames_to_tensor(frames, cameras, device):
    imgs = [
        torch.from_numpy(frames[cam]).permute(2, 0, 1).float() / 255.0
        for cam in cameras
    ]
    return torch.stack(imgs).unsqueeze(0).to(device)


def load_model(checkpoint_path, model_kwargs, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ACT(**model_kwargs)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    norm_mean = checkpoint["norm_mean"].to(device)
    norm_std = checkpoint["norm_std"].to(device)
    return model, norm_mean, norm_std


def run_episode(
    model,
    norm_mean,
    norm_std,
    env_cfg,
    r_cfg,
    cameras,
    chunk_size,
    action_dim,
    max_steps,
    seed,
    device,
):
    env = ReachEnvironment(
        scene_xml_path=env_cfg["scene_xml_path"],
        target_x_range=tuple(env_cfg["target_x_range"]),
        target_y_range=tuple(env_cfg["target_y_range"]),
        target_z_range=tuple(env_cfg["target_z_range"]),
        reach_threshold=env_cfg["reach_threshold"],
        seed=seed,
    )
    buffer = ChunkingBuffer(chunk_size=chunk_size, action_size=action_dim)
    query_every = max(1, chunk_size // 5)
    infer_times = []
    joints = []
    ee_pos = []
    terminated = False

    with SceneRenderer(
        env,
        height=r_cfg["height"],
        width=r_cfg["width"],
        camera_list=cameras,
        show_viewer=False,
    ) as renderer:
        obs = env.reset()
        buffer.reset()
        frames = render_current(renderer, cameras)

        for step in range(max_steps):
            if step % query_every == 0:
                qpos = obs[:6]
                qpos_t = (
                    torch.from_numpy(qpos).float().to(device) - norm_mean
                ) / norm_std
                images_t = frames_to_tensor(frames, cameras, device)

                start = time.perf_counter()
                with torch.inference_mode():
                    pred_norm = model(images_t, qpos_t.unsqueeze(0)).squeeze(0)
                infer_times.append(time.perf_counter() - start)

                buffer.add(pred_norm, step)

            action_norm = buffer.get_action(step)
            action = (action_norm * norm_std + norm_mean).cpu().numpy()
            obs, terminated = env.step(action)

            joints.append(env.data.qpos[:6].copy())
            ee_pos.append(env.data.xpos[env.ee_id].copy())

            frames = render_current(renderer, cameras)

            if terminated:
                break

    return {
        "success": terminated,
        "sim_time": env.data.time,
        "infer_times": infer_times,
        "joints": np.array(joints),
        "ee_pos": np.array(ee_pos),
    }


def measure_model(
    name,
    checkpoint_path,
    model_kwargs,
    chunk_size,
    action_dim,
    cfg,
    device,
    logger,
):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, norm_mean, norm_std = load_model(checkpoint_path, model_kwargs, device)

    env_cfg = cfg["env"]
    r_cfg = cfg["renderer"]
    cameras = cfg["collection"]["render_cameras"]
    max_steps = cfg["collection"]["n_steps"]
    n_episodes = cfg["eval"]["measure"]["n_episodes"]

    successes = 0
    sim_times = []
    infer_times = []
    joints = []
    ee_pos = []

    for seed in range(n_episodes):
        result = run_episode(
            model,
            norm_mean,
            norm_std,
            env_cfg,
            r_cfg,
            cameras,
            chunk_size,
            action_dim,
            max_steps,
            seed,
            device,
        )
        if result["success"]:
            successes += 1
            sim_times.append(result["sim_time"])
        infer_times.extend(result["infer_times"])
        joints.append(result["joints"])
        ee_pos.append(result["ee_pos"])
        logger.info(
            f"{name} seed {seed}: success={result['success']} sim_time={result['sim_time']:.3f}s"
        )

    joints = np.concatenate(joints, axis=0)
    ee_pos = np.concatenate(ee_pos, axis=0)

    vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    # macOS reports ru_maxrss in bytes; Linux reports it in KiB
    _rss_divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _rss_divisor
    size_mb = os.path.getsize(checkpoint_path) / (1024**2)

    metrics = {
        "success_rate": successes / n_episodes,
        "mean_convergence_time_s": float(np.mean(sim_times)) if sim_times else None,
        "mean_inference_time_ms": float(np.mean(infer_times) * 1000),
        "model_size_mb": size_mb,
        "vram_mb": vram_mb,
        "ram_mb": ram_mb,
        "joint_mean": joints.mean(axis=0).tolist(),
        "joint_std": joints.std(axis=0).tolist(),
        "ee_pos_mean": ee_pos.mean(axis=0).tolist(),
        "ee_pos_std": ee_pos.std(axis=0).tolist(),
    }
    logger.info(f"{name} results: {metrics}")
    return metrics


def main():
    cfg = load_config()
    m = cfg["eval"]["measure"]
    logger = Logger(m["log_file"])
    device = get_device()
    logger.info(f"Using device: {device}")

    teacher_ev = cfg["eval"]["teacher"]
    student_ev = cfg["eval"]["student"]
    t = cfg["training"]
    d = cfg["distillation"]

    if teacher_ev["auto_pull"]:
        ensure_checkpoint(
            teacher_ev["checkpoint"], teacher_ev["repo_id"], teacher_ev["filename"]
        )
    if student_ev["auto_pull"]:
        ensure_checkpoint(
            student_ev["checkpoint"], student_ev["repo_id"], student_ev["filename"]
        )

    teacher_kwargs = dict(
        action_dim=t["action_dim"],
        embed_dim=t["embed_dim"],
        latent_dim=t["latent_dim"],
        joint_dim=t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=t["nhead"],
        num_layers=t["num_layers"],
        num_cameras=t["num_cameras"],
    )
    student_kwargs = dict(
        action_dim=t["action_dim"],
        embed_dim=d["embed_dim"],
        latent_dim=d["latent_dim"],
        joint_dim=t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=d["nhead"],
        num_layers=d["num_layers"],
        num_cameras=d["num_cameras"],
        teacher_latent_dim=t["latent_dim"],
        distil_act=True,
    )

    teacher_metrics = measure_model(
        "teacher",
        teacher_ev["checkpoint"],
        teacher_kwargs,
        t["chunk_size"],
        t["action_dim"],
        cfg,
        device,
        logger,
    )
    student_metrics = measure_model(
        "student",
        student_ev["checkpoint"],
        student_kwargs,
        t["chunk_size"],
        t["action_dim"],
        cfg,
        device,
        logger,
    )

    results = {"teacher": teacher_metrics, "student": student_metrics}
    os.makedirs(os.path.dirname(m["output_path"]), exist_ok=True)
    with open(m["output_path"], "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {m['output_path']}")


if __name__ == "__main__":
    main()
