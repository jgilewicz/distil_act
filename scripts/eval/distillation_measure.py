import argparse
import json
import os
import platform
import resource
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

from algorithms.act_policy import ACT
from algorithms.chunking_buffer import ChunkingBuffer
from renderer.renderer import SceneRenderer
from utils.config import load_config
from utils.hub import ensure_checkpoint
from utils.logger import Logger
from utils.tasks import ENV_CLASSES


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


def _norm_stats(checkpoint, device):
    return (
        checkpoint["norm_mean"].to(device),
        checkpoint["norm_std"].to(device),
        checkpoint["action_norm_mean"].to(device),
        checkpoint["action_norm_std"].to(device),
    )


def load_model(checkpoint_path, model_kwargs, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ACT(**model_kwargs)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return (model, *_norm_stats(checkpoint, device))


class OnnxModel:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

    def __call__(self, images, joints):
        feeds = {
            "images": images.detach().cpu().numpy().astype(np.float32),
            "joints": joints.detach().cpu().numpy().astype(np.float32),
        }
        return torch.from_numpy(self.session.run(None, feeds)[0])


def load_quantized_model(onnx_path, norm_checkpoint, device):
    model = OnnxModel(onnx_path)
    checkpoint = torch.load(norm_checkpoint, map_location=device, weights_only=True)
    return (model, *_norm_stats(checkpoint, device))


class TensorRtModel:
    def __init__(self, onnx_path, engine_path, logger=None):
        from utils.tensorrt import TensorRtRuntime, build_engine

        engine_bytes = build_engine(onnx_path, engine_path, logger=logger)
        self._runtime = TensorRtRuntime(engine_bytes)

    def __call__(self, images, joints):
        feeds = {
            "images": images.detach().cpu().numpy().astype(np.float32),
            "joints": joints.detach().cpu().numpy().astype(np.float32),
        }
        return torch.from_numpy(self._runtime.infer(feeds))

    def close(self):
        self._runtime.close()


def load_tensorrt_model(onnx_path, engine_path, norm_checkpoint, device, logger=None):
    model = TensorRtModel(onnx_path, engine_path, logger=logger)
    checkpoint = torch.load(norm_checkpoint, map_location=device, weights_only=True)
    return (model, *_norm_stats(checkpoint, device))


def run_episode(
    model,
    norm_mean,
    norm_std,
    action_mean,
    action_std,
    env_cls,
    env_cfg,
    r_cfg,
    cameras,
    chunk_size,
    joint_dim,
    action_dim,
    max_steps,
    seed,
    device,
):
    env = env_cls(**env_cfg, seed=seed)
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
                qpos = obs[:joint_dim]
                qpos_t = (
                    torch.from_numpy(qpos).float().to(device) - norm_mean
                ) / norm_std
                images_t = frames_to_tensor(frames, cameras, device)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    pred_norm = model(images_t, qpos_t.unsqueeze(0)).squeeze(0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_times.append(time.perf_counter() - start)

                buffer.add(pred_norm, step)

            action_norm = buffer.get_action(step)
            action = (action_norm * action_std + action_mean).cpu().numpy()
            obs, terminated = env.step(action)

            joints.append(env.data.qpos.copy())
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
    joint_dim,
    action_dim,
    cfg,
    task,
    device,
    logger,
    quantized=False,
    norm_checkpoint=None,
    engine_path=None,
):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _rss_divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    ram_before_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _rss_divisor

    if engine_path is not None:
        model, norm_mean, norm_std, action_mean, action_std = load_tensorrt_model(
            checkpoint_path, engine_path, norm_checkpoint, device, logger=logger
        )
        size_path = engine_path
    elif quantized:
        model, norm_mean, norm_std, action_mean, action_std = load_quantized_model(
            checkpoint_path, norm_checkpoint, device
        )
        size_path = checkpoint_path
    else:
        model, norm_mean, norm_std, action_mean, action_std = load_model(
            checkpoint_path, model_kwargs, device
        )
        size_path = checkpoint_path

    task_cfg = cfg["collect"]["tasks"][task]
    env_cls = ENV_CLASSES[task]
    env_cfg = task_cfg["env"]
    r_cfg = cfg["renderer"]
    cameras = task_cfg["collection"]["render_cameras"]
    max_steps = task_cfg["collection"]["n_steps"]
    n_episodes = cfg["eval"]["tasks"][task]["measure"]["n_episodes"]

    successes = 0
    sim_times = []
    infer_times = []
    joints = []
    ee_pos = []

    try:
        for seed in range(n_episodes):
            result = run_episode(
                model,
                norm_mean,
                norm_std,
                action_mean,
                action_std,
                env_cls,
                env_cfg,
                r_cfg,
                cameras,
                chunk_size,
                joint_dim,
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
    finally:
        if isinstance(model, TensorRtModel):
            model.close()

    joints = np.concatenate(joints, axis=0)
    ee_pos = np.concatenate(ee_pos, axis=0)

    vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    ram_after_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _rss_divisor
    ram_delta_mb = max(0.0, ram_after_mb - ram_before_mb)
    size_mb = os.path.getsize(size_path) / (1024**2)

    metrics = {
        "success_rate": successes / n_episodes,
        "mean_convergence_time_s": float(np.mean(sim_times)) if sim_times else None,
        "mean_inference_time_ms": float(np.mean(infer_times) * 1000),
        "model_size_mb": size_mb,
        "vram_mb": vram_mb,
        "ram_delta_mb": ram_delta_mb,
        "joint_mean": joints.mean(axis=0).tolist(),
        "joint_std": joints.std(axis=0).tolist(),
        "ee_pos_mean": ee_pos.mean(axis=0).tolist(),
        "ee_pos_std": ee_pos.std(axis=0).tolist(),
    }
    logger.info(f"{name} results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(ENV_CLASSES), required=True)
    args = parser.parse_args()
    task = args.task

    cfg = load_config()
    m = cfg["eval"]["tasks"][task]["measure"]
    logger = Logger(m["log_file"])
    device = get_device()
    logger.info(f"Using device: {device}")

    teacher_ev = cfg["eval"]["tasks"][task]["teacher"]
    student_ev = cfg["eval"]["tasks"][task]["student"]
    t = cfg["training"]
    d = cfg["distillation"]
    task_t = t["tasks"][task]

    if teacher_ev["auto_pull"]:
        ensure_checkpoint(
            teacher_ev["checkpoint"], teacher_ev["repo_id"], teacher_ev["filename"]
        )
    if student_ev["auto_pull"]:
        ensure_checkpoint(
            student_ev["checkpoint"], student_ev["repo_id"], student_ev["filename"]
        )

    teacher_kwargs = dict(
        action_dim=task_t["action_dim"],
        embed_dim=t["embed_dim"],
        latent_dim=t["latent_dim"],
        joint_dim=task_t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=t["nhead"],
        num_layers=t["num_layers"],
        num_cameras=t["num_cameras"],
    )
    student_kwargs = dict(
        action_dim=task_t["action_dim"],
        embed_dim=d["embed_dim"],
        latent_dim=d["latent_dim"],
        joint_dim=task_t["joint_dim"],
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
        task_t["joint_dim"],
        task_t["action_dim"],
        cfg,
        task,
        device,
        logger,
    )
    student_metrics = measure_model(
        "student",
        student_ev["checkpoint"],
        student_kwargs,
        t["chunk_size"],
        task_t["joint_dim"],
        task_t["action_dim"],
        cfg,
        task,
        device,
        logger,
    )

    results = {
        "teacher": teacher_metrics,
        "student": student_metrics,
    }

    if task == "reach":
        quantized_variants = {
            "student_ptq": cfg["ptq"]["output_path"],
            "student_dyn": cfg["ptq"]["dyn_path"],
            "student_fp16_onnx": cfg["ptq"]["fp16_path"],
        }
        for name, onnx_path in quantized_variants.items():
            results[name] = measure_model(
                name,
                onnx_path,
                student_kwargs,
                t["chunk_size"],
                task_t["joint_dim"],
                task_t["action_dim"],
                cfg,
                task,
                torch.device("cpu"),
                logger,
                quantized=True,
                norm_checkpoint=student_ev["checkpoint"],
            )

        trt_variants = {
            "student_trt_fp32": (
                cfg["ptq"]["fp32_path"],
                cfg["ptq"]["engine_fp32_path"],
            ),
            "student_trt_fp16": (
                cfg["ptq"]["fp16_path"],
                cfg["ptq"]["engine_fp16_path"],
            ),
        }
        for name, (onnx_path, engine_path) in trt_variants.items():
            results[name] = measure_model(
                name,
                onnx_path,
                student_kwargs,
                t["chunk_size"],
                task_t["joint_dim"],
                task_t["action_dim"],
                cfg,
                task,
                torch.device("cpu"),
                logger,
                norm_checkpoint=student_ev["checkpoint"],
                engine_path=engine_path,
            )
    else:
        logger.info(
            f"Skipping quantized/TensorRT variants for task={task} — "
            "the ONNX export/PTQ pipeline (config/quant.yaml) is reach-only."
        )

    os.makedirs(os.path.dirname(m["output_path"]), exist_ok=True)
    with open(m["output_path"], "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {m['output_path']}")


if __name__ == "__main__":
    main()
