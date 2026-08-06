# Architecture

## Data flow

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
                                                    ↓
                                distillation_measure.py (OnnxModel via ONNX Runtime)
```

## Key files

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
- Evaluates the fp32 teacher and student plus two quantized ONNX variants (`student_ptq`, `student_dyn`) over `eval.measure.n_episodes` randomly-seeded episodes.
- fp32 models load via `load_model` (torch) on the selected device; the quantized ONNX variants load via `load_quantized_model` → `OnnxModel` (an `onnxruntime.InferenceSession` wrapper) on CPU, reusing the student checkpoint's `norm_mean`/`norm_std` (the ONNX files carry no norm stats).
- Per-episode: runs the full policy loop headlessly, records inference times, joint trajectories, EE positions, and success.
- Aggregates: success rate, mean convergence time, mean inference latency, model size, peak VRAM/RAM.
- Writes `eval.measure.output_path` (JSON) and logs to `eval.measure.log_file`.

**`scripts/ptq.py`**
- Post-training quantization of the distilled student via ONNX Runtime, on CPU.
- Exports the student's inference path (`actions=None`) to fp32 ONNX (`torch.onnx.export(..., dynamo=True)`, input names `images`/`joints`), runs `quant_pre_process(skip_symbolic_shape=True)` (symbolic shape inference chokes on the transformer's `Loop` node), then produces two int8 models: static QDQ via `quantize_static` + an `ActCalibrationReader` over the val split (`ptq.output_path`), and dynamic weight-only via `quantize_dynamic` (`ptq.dyn_path`). Before dynamic quant it strips `graph.value_info` (the dynamo export's stale shapes trip `quantize_dynamic`'s strict shape inference).

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

## Project structure

```
src/
  env/              # MuJoCo reach environment
  expert/           # IK-based scripted expert (mink) + abstract base
  renderer/         # off-screen rendering + passive viewer
  dataset/          # HDF5 episode recording + PyTorch dataset/dataloader
  algorithms/       # ACT policy, ImageEmbedding, ChunkingBuffer
  utils/            # logger, config loader, Hugging Face Hub helpers
scripts/
  collect_data.py           # expert demo collection
  train_act.py              # ACT teacher training loop
  train_distil.py           # student distillation loop
  eval_act.py               # teacher evaluation + video export
  eval_distil.py            # student evaluation + video export
  ptq.py                    # post-training quantization → static + dynamic int8 ONNX
  distillation_measure.py   # teacher / student / quantized benchmark (success, latency, memory)
  push_data_to_hub.py       # manual dataset push
  push_teacher_to_hub.py    # manual teacher checkpoint push
  push_student_to_hub.py    # manual student checkpoint push
models/
  reach_scene.xml
tests/
config/
  simulation.yaml   # env, expert, renderer
  collection.yaml   # data collection
  train.yaml        # teacher training
  distill.yaml      # student distillation
  eval.yaml         # evaluation + measurement
  quant.yaml        # quantization (ptq)
justfile
Docker/Dockerfile
.github/workflows/ci.yml
```

## Style rules
- No `sys.path` manipulation — packages are installed via `uv sync` (hatchling src layout).
- No multi-line comments or docstrings — single-line `#` only where the WHY is non-obvious.
- No `print` — use `Logger`.
