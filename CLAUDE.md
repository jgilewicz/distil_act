# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACT Distillation — distilling an ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

Phase 2 is complete: expert demonstrations collected via IK, ACT trained with CVAE + temporal ensembling, evaluated in the MuJoCo reach environment. Phase 3 adds distillation of the teacher ACT into a smaller student (MobileNetV3 backbone, distillation KL + hard/soft action losses). Phase 4 adds quantization of the distilled student: ONNX Runtime post-training quantization (static QDQ int8 + dynamic weight-only int8) and eager quantization-aware training (QAT). All quantized variants are exported to ONNX and benchmarked against the fp32 teacher/student in `distillation_measure.py`.

## Commands

```bash
uv sync                  # install / sync dependencies
just                     # list all available tasks
just collect             # collect demos with viewer (macOS: uses mjpython)
just collect-headless    # collect headless
just train               # train teacher ACT policy
just distill             # distill the teacher into a smaller student
just eval                # run the trained teacher with viewer (macOS: uses mjpython)
just eval-distill        # run the distilled student with viewer (macOS: uses mjpython)
just ptq                 # post-training quantization (ONNX Runtime): static + dynamic int8 student
just qat                 # quantization-aware fine-tune of the student, exported to ONNX
just measure             # compare teacher, student, and quantized variants: success rate, latency, size, VRAM/RAM
just push-data           # push collected dataset to the Hub
just push-teacher        # push teacher checkpoint to the Hub
just push-student        # push student checkpoint to the Hub
just test                # run test suite (pytest)
just lint                # ruff check
just fix                 # ruff check --fix + ruff format
just clean               # remove generated logs, dataset, and pycache
```

On macOS, anything that calls `mujoco.viewer.launch_passive` must run under `mjpython`. The justfile handles this — `just collect` and `just eval` use `uv run mjpython`. `just measure` is headless (no viewer), so it runs under plain `python3` and works cross-platform.

All configuration lives in `config/*.yaml`. `load_config()` (`src/utils/config.py`) merges every YAML file in `config/` into a single dict keyed by top-level section (`env`, `expert`, `renderer`, `collection`, `training`, `distillation`, `eval`, `qat`, `ptq`) — split into separate files (`simulation.yaml`, `collection.yaml`, `train.yaml`, `distill.yaml`, `eval.yaml`, `quant.yaml`) purely for navigability, not namespacing. `quant.yaml` holds the top-level `qat` and `ptq` sections. No hardcoded constants in source files.

Each stage's Hub interaction (dataset/checkpoint download and upload) is config-driven, not a separate manual step: every stage config carries a `hub` block with `repo_id`/`filename` plus `auto_pull` (download if the local file/dir is missing — via `src/utils/hub.py`) and `auto_push` (upload once the stage finishes). The dedicated `push-*` scripts/just recipes remain for pushing on demand (e.g. re-pushing without retraining).

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`: lint → test → docker build. The Docker image (`Docker/Dockerfile`) is `python:3.14-slim` with `MUJOCO_GL=disabled` for headless physics.

## Architecture

### Data flow

```
ReachEnvironment → ReachExpert → SceneRenderer → EpisodeRecorder → DatasetManager
  (physics)          (IK)         (rendering)      (HDF5 write)     (file layout)
                                                                          ↓
                                                                   EpisodeDataset
                                                                  (PyTorch loader)
                                                                          ↓
                                                                  ACT teacher training
                                                                          ↓
                                                               act_model_final.pt ──────────────┐
                                                                          ↓                      ↓
                                                          eval_act.py + ChunkingBuffer   ACT student distillation
                                                                                                  ↓
                                                                                    distil_act_model_final.pt
                                                                                                  ↓
                                                                              eval_distil.py + ChunkingBuffer
```

Quantization branch (Phase 4) — all read `distil_act_model_final.pt`, all write ONNX, all measured through ONNX Runtime:

```
distil_act_model_final.pt ──> ptq.py  ──> distil_act_model_ptq.onnx  (static QDQ int8)
                          └──> ptq.py  ──> distil_act_model_dyn.onnx  (dynamic weight-only int8)
                          └──> qat.py  ──> distil_act_model_qat.onnx  (QAT int8, fine-tuned)
                                                    ↓
                                distillation_measure.py (OnnxModel via ONNX Runtime)
```

### Key files

**`src/env/base.py` — `Environment`**
- Abstract base class with `reset() -> np.ndarray` and `step(action) -> (obs, terminated)`.

**`src/env/env.py` — `ReachEnvironment(Environment)`**
- Merges `models/reach_scene.xml` with the `low_cost_robot_arm` MJCF from `robot_descriptions` at runtime; attaches an `ego_cam` to `robot_gripper_static_finger`.
- All robot bodies/joints/actuators are prefixed `robot_` after merging.
- `step(action)` returns `(obs, terminated)` — terminated when EE distance < `reach_threshold`.
- Observation vector: `[qpos(6), qvel(6), ee_pos(3), target_pos(3)]`. Target is a mocap body.

**`src/expert/reach_expert.py` — `ReachExpert`**
- Uses `mink` (IK library) with `daqp` solver and `mink.FrameTask` on `robot_gripper_moving_finger` (`frame_type="body"` — robot has no sites).
- Each `compute_action(obs)` call syncs mink config from live `data.qpos`, runs up to `max_iters` IK iterations (early-exit at `ik_pos_threshold`), returns a ctrl array for the position actuators.
- No `VelocityLimit` — lets IK jump to solution aggressively for fast expert convergence.
- Not all targets in the configured range are reachable within 400 steps; seed=1 is a known-good target for testing.

**`src/renderer/renderer.py` — `SceneRenderer`**
- Context manager; `render_step(action)` returns `(obs, terminated, frames)` where `frames` is a `dict[camera_name → BGR ndarray]`.
- Available cameras: `overhead_cam` (scene XML) and `ego_cam` (gripper-mounted).

**`src/dataset/`**
- `EpisodeRecorder`: single HDF5 file with resizable datasets `frames` (uint8, shape `T×K×H×W×3`), `joints` (float32), `timestamps` (float64). K = number of cameras.
- `DatasetManager`: manages `root_dir/episodes/episode_N.h5` layout; `img_shape` must be `(K, H, W, 3)`; auto-increments index from existing files; writes `metadata.json`.
- `EpisodeDataset` (`src/dataset/dataloader.py`): PyTorch `Dataset`; `__getitem__` returns `{"images": (K,3,H,W), "qpos": (J,), "actions": (chunk_size,J)}`; batched to `(B,K,3,H,W)`. `make_dataloader()` returns a configured `DataLoader`.
- Joint positions are z-score normalised using `mean`/`std` computed across the full training split; both are saved into `act_model_final.pt` for inference.

**`src/algorithms/`**
- `embedding.py` — `ImageEmbedding`: frozen EfficientNet-B3 backbone + AdaptiveAvgPool → linear projection; adds per-camera and positional embeddings; output shape `(B, K*P, embed_dim)` where P=49 patches. On MPS, the AdaptiveAvgPool runs on CPU (non-divisible sizes unsupported on MPS).
- `act_policy.py` — `ACT`: full encoder-decoder Transformer. Training: takes `(images, qpos, actions)`, runs CVAE encoder for latent z, returns `(pred_actions, mu, logvar)`. Inference: z=0, returns `pred_actions` only.
- `chunking_buffer.py` — `ChunkingBuffer`: stores overlapping action chunk predictions; `get_action(t)` returns exponentially weighted average over all chunks that cover timestep t; evicts chunks older than `chunk_size` steps.

**`scripts/distillation_measure.py`**
- Evaluates the fp32 teacher and student plus three quantized ONNX variants (`student_ptq`, `student_dyn`, `student_qat`) over `eval.measure.n_episodes` randomly-seeded episodes.
- fp32 models load via `load_model` (torch) on the selected device; the quantized ONNX variants load via `load_quantized_model` → `OnnxModel` (an `onnxruntime.InferenceSession` wrapper) on CPU, reusing the student checkpoint's `norm_mean`/`norm_std` (the ONNX files carry no norm stats).
- Per-episode: runs the full policy loop headlessly, records inference times, joint trajectories, EE positions, and success.
- Aggregates: success rate, mean convergence time, mean inference latency, model size, peak VRAM/RAM.
- Writes `eval.measure.output_path` (JSON) and logs to `eval.measure.log_file`.

**`scripts/ptq.py`**
- Post-training quantization of the distilled student via ONNX Runtime, on CPU.
- Exports the student's inference path (`actions=None`) to fp32 ONNX (`torch.onnx.export(..., dynamo=True)`, input names `images`/`joints`), runs `quant_pre_process(skip_symbolic_shape=True)` (symbolic shape inference chokes on the transformer's `Loop` node), then produces two int8 models: static QDQ via `quantize_static` + an `ActCalibrationReader` over the val split (`ptq.output_path`), and dynamic weight-only via `quantize_dynamic` (`ptq.dyn_path`). Before dynamic quant it strips `graph.value_info` (the dynamo export's stale shapes trip `quantize_dynamic`'s strict shape inference).

**`scripts/qat.py`**
- Eager quantization-aware fine-tune of the distilled student. `apply_qat_qconfig` sets a `get_default_qat_qconfig(qat.backend)` (fbgemm/x86) only on plain `nn.Linear` (`type(module) is nn.Linear` — excludes MHA's `NonDynamicallyQuantizableLinear`), a weight-only qconfig on `nn.Embedding`, and skips the `encoder_cvae` branch (unused at inference).
- Trains on CUDA when available with `prepare_qat` + fake-quant, using hard (vs dataset actions) + soft (vs teacher predictions) MSE — no CVAE/KL terms since the inference path returns only `pred_actions`. Backbone BN is frozen (`image_embedding.eval()`). Moves to CPU and exports to ONNX QDQ (`qat.output_path`).

**`scripts/train_distil.py`**
- Loads the frozen teacher (`training` dims) and trains a smaller student `ACT` (`distillation` dims, `distil_act=True` → MobileNetV3-Large backbone instead of EfficientNet-B3).
- Loss = `alpha * hard_loss + (1-alpha) * soft_loss + beta * prior_kl + gamma * distill_kl`; `distill_kl` matches the student's latent (projected to `teacher_latent_dim`) against the teacher's CVAE posterior at `temperature`.

**`scripts/eval_act.py`** / **`scripts/eval_distil.py`**
- Load `eval.teacher.checkpoint` / `eval.student.checkpoint` respectively (model weights + `norm_mean` + `norm_std`). The student script builds its `ACT` with `distillation` dims and `distil_act=True` — these must match the architecture the checkpoint was trained with.
- Queries ACT every `chunk_size // 5` physics steps; `ChunkingBuffer` provides temporally ensembled actions for intermediate steps.
- Renders passive viewer via `SceneRenderer`; writes the configured `video_path` from the overhead camera.

**`src/utils/`**
- `logger.py` — `Logger(filename)`: logs `[INFO]`/`[WARNING]`/`[ERROR]` to stdout and file simultaneously.
- `config.py` — `load_config(config_dir="config")`: merges every `*.yaml` in the directory into one dict; raises on duplicate top-level keys.
- `hub.py` — `ensure_checkpoint`/`ensure_dataset` (download if missing) and `push_checkpoint`/`push_dataset` (upload), shared by the training scripts' auto-pull/auto-push hooks and the manual `push_*_to_hub.py` scripts.

### Style rules
- No `sys.path` manipulation — packages are installed via `uv sync` (hatchling src layout).
- No multi-line comments or docstrings — single-line `#` only where the WHY is non-obvious.
- No `print` — use `Logger`.
