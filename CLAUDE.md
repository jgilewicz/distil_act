# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACT Distillation — distilling an ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

Phase 2 is complete: expert demonstrations collected via IK, ACT trained with CVAE + temporal ensembling, evaluated in the MuJoCo reach environment. Phase 3 adds distillation of the teacher ACT into a smaller student (MobileNetV3 backbone, distillation KL + hard/soft action losses). Phase 4 adds post-training quantization of the distilled student via ONNX Runtime (static QDQ int8 + dynamic weight-only int8), exported to ONNX and benchmarked against the fp32 teacher/student in `distillation_measure.py`. A second task, pick-and-place (box pickup + placement), has been added at the env/expert layer alongside reach — collection, teacher training, distillation, evaluation, and measurement are all `--task {reach,pick_and_place}`-parameterized end to end; only ONNX export/PTQ quantization (`export_onnx.py`/`ptq.py`, `config/quant.yaml`) remains reach-only.

## Commands

```bash
uv sync                          # install / sync dependencies
just                              # list all available tasks
just collect <task>               # collect demos with viewer (macOS: uses mjpython); task = reach | pick_and_place
just collect-headless <task>      # collect demos headless; task = reach | pick_and_place
just train <task>                 # train teacher ACT policy; task = reach | pick_and_place
just distill <task>               # distill the teacher into a smaller student; task = reach | pick_and_place
just eval <task>                  # run the trained teacher with viewer (macOS: uses mjpython); task = reach | pick_and_place
just eval-distill <task>          # run the distilled student with viewer (macOS: uses mjpython); task = reach | pick_and_place
just export-onnx                  # export the distilled student to ONNX: fp32 + fp16 (modelopt autocast); reach only
just ptq                          # post-training quantization (ONNX Runtime): static + dynamic int8 student, from the exported fp32 ONNX; reach only
just measure <task>               # compare teacher, student, and (reach only) quantized variants: success rate, latency, size, VRAM/RAM
just push-data <task>             # push collected dataset to the Hub; task = reach | pick_and_place
just push-teacher <task>          # push teacher checkpoint to the Hub; task = reach | pick_and_place
just push-student <task>          # push student checkpoint to the Hub; task = reach | pick_and_place
just test                         # run test suite (pytest)
just lint                         # ruff check
just fix                          # ruff check --fix + ruff format
just clean                        # remove generated logs, dataset, and pycache
```

On macOS, anything that calls `mujoco.viewer.launch_passive` must run under `mjpython`. The justfile handles this — `just collect` and `just eval` use `uv run mjpython`. `just measure` is headless (no viewer), so it runs under plain `python3` and works cross-platform.

All configuration lives in `config/*.yaml`. `load_config()` (`src/utils/config.py`) merges every YAML file in `config/` into a single dict keyed by top-level section (`collect`, `renderer`, `training`, `distillation`, `eval`, `ptq`) — split into separate files (`collect.yaml`, `simulation.yaml`, `train.yaml`, `distill.yaml`, `eval.yaml`, `quant.yaml`) purely for navigability, not namespacing. `collect.yaml`, `train.yaml`, `distill.yaml`, and `eval.yaml` all hold `<section>.tasks.<reach|pick_and_place>` blocks with identically-shaped sub-sections per task, plus shared (task-independent) keys directly under `<section>`:
- `collect.tasks.<task>`: `env`/`expert`/`collection`.
- `training.tasks.<task>`: `action_dim`/`joint_dim`/`action_qpos_offset`/`log_file`/`checkpoint_dir`/`checkpoint_prefix`/`hub`. The two tasks record different qpos dims — reach: 6 robot joints (`joint_dim=action_dim=6`, `offset=0`); pick_and_place: box's 7-dof freejoint + 6 robot joints (`joint_dim=13`, `action_dim=6`, `offset=7`) — the model conditions on the full recorded qpos but only predicts/controls the robot's slice of it.
- `distillation.tasks.<task>`: `teacher`/`log_file`/`checkpoint_dir`/`checkpoint_prefix`/`hub`.
- `eval.tasks.<task>`: `teacher`/`student`/`measure`.

Every task-aware script takes `--task <task>` and reads the matching nested block: `collect_data.py`, `push_data_to_hub.py`, `train_act.py`, `push_teacher_to_hub.py`, `train_distil.py`, `push_student_to_hub.py`, `eval_act.py`, `eval_distil.py`, `distillation_measure.py`. `src/utils/tasks.py`'s `ENV_CLASSES` dict maps `"reach"`/`"pick_and_place"` to their `Environment` subclass and is the shared registry the eval/measure scripts use to build the right environment. `simulation.yaml` holds only `renderer` (shared, task-independent). `quant.yaml` (`ptq`) remains reach-only, unparameterized by task. No hardcoded constants in source files.

Each stage's Hub interaction (dataset/checkpoint download and upload) is config-driven, not a separate manual step: every stage config carries a `hub` block with `repo_id`/`filename` plus `auto_pull` (download if the local file/dir is missing — via `src/utils/hub.py`) and `auto_push` (upload once the stage finishes). The dedicated `push-*` scripts/just recipes remain for pushing on demand (e.g. re-pushing without retraining).

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`: lint → test → docker build. The Docker image (`Docker/Dockerfile`) is `python:3.14-slim` with `MUJOCO_GL=disabled` for headless physics.

## Architecture

### Data flow

```
ENV_CLASSES[task] → task's Expert → SceneRenderer → EpisodeRecorder → DatasetManager
  (physics)            (IK)          (rendering)      (HDF5 write)     (file layout)
                                                                             ↓
                                                                      EpisodeDataset
                                                                     (PyTorch loader,
                                                                    per-task action slice)
                                                                             ↓
                                                                     ACT teacher training
                                                                             ↓
                                                          <prefix>_final.pt ──────────────┐
                                                                             ↓            ↓
                                                          eval_act.py + ChunkingBuffer   ACT student distillation
                                                                                                     ↓
                                                                                    <distil_prefix>_final.pt
                                                                                                     ↓
                                                                             eval_distil.py + ChunkingBuffer
```

`task` is `reach` or `pick_and_place`; `<prefix>`/`<distil_prefix>` are that task's `checkpoint_prefix` from `training.tasks.<task>`/`distillation.tasks.<task>` (`act_model`/`distil_act_model` for reach, `pick_and_place_model`/`pick_and_place_distil_model` for pick-and-place). `distillation_measure.py` runs the same teacher→student→(measure) path headlessly for either task, per `eval.tasks.<task>.measure`.

Quantization branch (Phase 4, **reach-only**) — export to ONNX is a separate step from PTQ itself; both write ONNX, both measured through ONNX Runtime:

```
distil_act_model_final.pt ──> export_onnx.py ──> distil_act_model_32.onnx  (fp32, torch.onnx.export)
                                             └──> distil_act_model_16.onnx  (fp16, modelopt autocast)
                                                    ↓
                                distil_act_model_32.onnx ──> ptq.py ──> distil_act_model_ptq.onnx  (static QDQ int8)
                                                         └──> ptq.py ──> distil_act_model_dyn.onnx  (dynamic weight-only int8)
                                                                                ↓
                                                        distillation_measure.py (OnnxModel via ONNX Runtime)
```

### Key files

**`src/env/base.py` — `Environment`**
- Abstract base class with `reset() -> np.ndarray` and `step(action) -> (obs, terminated)`.

**`src/env/reach_env.py` — `ReachEnvironment(Environment)`**
- Merges `models/reach_scene.xml` with the `low_cost_robot_arm` MJCF from `robot_descriptions` at runtime; attaches an `ego_cam` to `robot_gripper_static_finger`.
- All robot bodies/joints/actuators are prefixed `robot_` after merging.
- `step(action)` returns `(obs, terminated)` — terminated when EE distance < `placement_threshold`.
- Observation vector: `[qpos(6), qvel(6), ee_pos(3), target_pos(3)]`. Target is a mocap body.

**`src/expert/reach_expert.py` — `ReachExpert`**
- Uses `mink` (IK library) with `daqp` solver and `mink.FrameTask` on `robot_gripper_moving_finger` (`frame_type="body"` — robot has no sites).
- Each `compute_action(obs)` call syncs mink config from live `data.qpos`, runs up to `max_iters` IK iterations (early-exit at `ik_pos_threshold`), returns a ctrl array for the position actuators.
- No `VelocityLimit` — lets IK jump to solution aggressively for fast expert convergence.
- Not all targets in the configured range are reachable within 400 steps; seed=1 is a known-good target for testing.

**`src/env/pick_and_place_env.py` — `PickAndPlaceEnvironment(Environment)`**
- Second task: pick up a free-floating `box` body and place it at a mocap `place_target`. `step(action)` terminates when the box-to-target distance < `placement_threshold`.
- Observation vector: `[qpos, qvel, box_pos(3), place_target_pos(3)]`.
- `reset()` randomizes both the box's starting `(x, y)` and the placement target's `(x, y)` independently within their configured ranges.

**`src/expert/pick_and_place_expert.py` — `PickAndPlaceExpert(Expert)`**
- Six-phase state machine (`approach → descend → grasp → transport → lower → release`) driven by `_advance_phase`/`_phase_target`, each phase an IK target for `robot_gripper_static_finger` (not the moving finger — the static pad is the frame that needs to clear the box face) plus a gripper setpoint.
- `grasp` freezes the descend-phase target instead of tracking the live box position (contact would otherwise make it chase the box), and holds for a fixed `GRASP_HOLD_STEPS` dwell rather than a convergence check, since the box physically blocks the finger from reaching its commanded closed position.
- Closing is two-stage (`_close_cmd`): free-close to `GRIP_PRECLOSE` first, then advance the setpoint by `GRIP_FOLLOW_DELTA` per step ahead of the measured joint — stepping straight to the closed setpoint slams the finger into the box.

**`src/renderer/renderer.py` — `SceneRenderer`**
- Context manager; `render_step(action)` returns `(obs, terminated, frames)` where `frames` is a `dict[camera_name → BGR ndarray]`.
- Available cameras: `overhead_cam` (scene XML) and `ego_cam` (gripper-mounted).

**`src/dataset/`**
- `EpisodeRecorder`: single HDF5 file with resizable datasets `frames` (uint8, shape `T×K×H×W×3`), `joints` (float32), `timestamps` (float64). K = number of cameras.
- `DatasetManager`: manages `root_dir/episodes/episode_N.h5` layout; `img_shape` must be `(K, H, W, 3)`; auto-increments index from existing files; writes `metadata.json`.
- `EpisodeDataset` (`src/dataset/dataloader.py`), `EpisodeDataset(cfg, task, split)`: PyTorch `Dataset`; `__getitem__` returns `{"images": (K,3,H,W), "qpos": (joint_dim,), "actions": (chunk_size, action_dim)}`; batched to `(B,K,3,H,W)`. `qpos` is the full recorded joints vector; `actions` is sliced to `[action_qpos_offset : action_qpos_offset + action_dim]` from that same recorded-joints array (the task's robot-only qpos range) — for reach this slice is the identity (`offset=0`, `action_dim=joint_dim`). `make_dataloader(cfg, task, split)` returns a configured `DataLoader`.
- Joint positions are z-score normalised using `mean`/`std` computed across the full training split (`joint_dim`-sized, for `qpos`); `action_mean`/`action_std` are the corresponding slice of those same stats (`action_dim`-sized, for `actions`). All four are saved into the checkpoint (`norm_mean`/`norm_std`/`action_norm_mean`/`action_norm_std`) for inference — the eval scripts normalise the input `qpos` with the first pair and denormalise the predicted `action` with the second before sending it to `env.step()`.

**`src/algorithms/`**
- `embedding.py` — `ImageEmbedding`: frozen EfficientNet-B3 backbone + AdaptiveAvgPool → linear projection; adds per-camera and positional embeddings; output shape `(B, K*P, embed_dim)` where P=49 patches. On MPS, the AdaptiveAvgPool runs on CPU (non-divisible sizes unsupported on MPS).
- `act_policy.py` — `ACT`: full encoder-decoder Transformer. Training: takes `(images, qpos, actions)`, runs CVAE encoder for latent z, returns `(pred_actions, mu, logvar)`. Inference: z=0, returns `pred_actions` only.
- `chunking_buffer.py` — `ChunkingBuffer`: stores overlapping action chunk predictions; `get_action(t)` returns exponentially weighted average over all chunks that cover timestep t; evicts chunks older than `chunk_size` steps.

**`scripts/eval/distillation_measure.py --task {reach,pick_and_place}`**
- Evaluates the fp32 teacher and student over `eval.tasks.<task>.measure.n_episodes` randomly-seeded episodes; for `task=reach` only, also three ONNX Runtime variants (`student_ptq`, `student_dyn`, `student_fp16_onnx`) and two TensorRT engine variants (`student_trt_fp32`, `student_trt_fp16`), since the ONNX/PTQ pipeline is reach-only — `pick_and_place` logs a note and skips straight to writing teacher/student results.
- fp32 models load via `load_model` (torch) on the selected device; the ONNX Runtime variants load via `load_quantized_model` → `OnnxModel` (an `onnxruntime.InferenceSession` wrapper) on CPU; the TensorRT variants load via `load_tensorrt_model` → `TensorRtModel`, which builds/caches an engine with `utils.tensorrt.build_engine` (`ptq.engine_fp32_path`/`ptq.engine_fp16_path`) and runs it with `utils.tensorrt.run_inference`. All loaders return `(model, norm_mean, norm_std, action_norm_mean, action_norm_std)` — non-torch variants reuse the student checkpoint's four norm-stat tensors (the ONNX/engine files carry none themselves).
- Per-episode (`run_episode`): builds the task's `Environment` via `ENV_CLASSES[task]`, runs the full policy loop headlessly, records inference times, joint trajectories, EE positions, and success — predicted actions are denormalised with `action_norm_mean`/`action_norm_std` and applied directly as `ctrl`.
- Aggregates: success rate, mean convergence time, mean inference latency, model size (engine file size for TRT variants), peak VRAM/RAM.
- Writes `eval.tasks.<task>.measure.output_path` (JSON) and logs to `eval.tasks.<task>.measure.log_file`.

**`scripts/quantization/export_onnx.py`**
- Exports the distilled student's inference path (`actions=None`) to ONNX; loading the checkpoint and building the model is only done here, not in `ptq.py`.
- Exports fp32 via `torch.onnx.export(..., dynamo=True)` (input names `images`/`joints`) to `ptq.fp32_path`, then converts that to a mixed-precision fp16 graph via `modelopt.onnx.autocast.convert_to_mixed_precision` (`keep_io_types=True`) to `ptq.fp16_path`. TensorRT 11.x removed the builder-side `BuilderFlag.FP16`/`INT8` flags in favor of this offline ModelOpt AutoCast pass — precision is now baked into the ONNX graph (explicit `Cast` nodes) before it ever reaches TensorRT or ONNX Runtime.

**`scripts/quantization/ptq.py`**
- Post-training quantization of the already-exported fp32 ONNX (`ptq.fp32_path`) via ONNX Runtime, on CPU. Does not touch the checkpoint or the model class — run `export_onnx.py` first.
- Runs `quant_pre_process(skip_symbolic_shape=True)` (symbolic shape inference chokes on the transformer's `Loop` node), then produces two int8 models: static QDQ via `quantize_static` + an `ActCalibrationReader` over the val split (`ptq.output_path`), and dynamic weight-only via `quantize_dynamic` (`ptq.dyn_path`). Before dynamic quant it strips `graph.value_info` (the dynamo export's stale shapes trip `quantize_dynamic`'s strict shape inference).

**`scripts/training/train_distil.py --task {reach,pick_and_place}`**
- Loads the frozen teacher (`training.tasks.<task>` dims, from `distillation.tasks.<task>.teacher.checkpoint`, auto-pulled if `auto_pull`) and trains a smaller student `ACT` (`distillation` dims, `distil_act=True` → MobileNetV3-Large backbone instead of EfficientNet-B3, same `action_dim`/`joint_dim` as the teacher).
- Loss = `alpha * hard_loss + (1-alpha) * soft_loss + beta * prior_kl + gamma * distill_kl`; `distill_kl` matches the student's latent (projected to `teacher_latent_dim`) against the teacher's CVAE posterior at `temperature`.
- Writes `<checkpoint_prefix>_step_<N>.pt`/`<checkpoint_prefix>_final.pt` into `distillation.tasks.<task>.checkpoint_dir`, carrying the same four norm-stat tensors as the teacher checkpoint.

**`scripts/eval/eval_act.py --task {reach,pick_and_place}`** / **`scripts/eval/eval_distil.py --task {reach,pick_and_place}`**
- Load `eval.tasks.<task>.teacher.checkpoint` / `.student.checkpoint` respectively (model weights + `norm_mean`/`norm_std`/`action_norm_mean`/`action_norm_std`). The student script builds its `ACT` with `distillation` dims and `distil_act=True` — these must match the architecture the checkpoint was trained with.
- Builds the task's `Environment` via `ENV_CLASSES[task]` (`src/utils/tasks.py`) from `collect.tasks.<task>.env`.
- Queries ACT every `chunk_size // 5` physics steps; `ChunkingBuffer` provides temporally ensembled actions for intermediate steps. Predicted actions are denormalised with `action_norm_mean`/`action_norm_std` and applied directly as `env.step()`'s `ctrl`.
- Renders passive viewer via `SceneRenderer`; writes the configured `video_path` from the overhead camera.

**`src/utils/`**
- `logger.py` — `Logger(filename)`: logs `[INFO]`/`[WARNING]`/`[ERROR]` to stdout and file simultaneously.
- `config.py` — `load_config(config_dir="config")`: merges every `*.yaml` in the directory into one dict; raises on duplicate top-level keys.
- `hub.py` — `ensure_checkpoint`/`ensure_dataset` (download if missing) and `push_checkpoint`/`push_dataset` (upload), shared by the training scripts' auto-pull/auto-push hooks and the manual `push_*_to_hub.py` scripts.
- `tasks.py` — `ENV_CLASSES: dict[str, type[Environment]]` mapping `"reach"`/`"pick_and_place"` to `ReachEnvironment`/`PickAndPlaceEnvironment`. Shared by `eval_act.py`, `eval_distil.py`, `distillation_measure.py`, and `collect_data.py`'s `TaskSpec` registry — the single place a new task's env class gets registered.
- `tensorrt.py` — `build_engine(onnx_path, engine_path)` (parses an ONNX graph, builds a serialized TensorRT engine, caches it to `engine_path`) and `run_inference(engine_bytes, inputs)` (raw `cuda-python`/`cudart` H2D copy → `execute_async_v3` → D2H copy). Shared between `scripts/quantization/inference.py` (standalone CLI smoke test) and `distillation_measure.py` (fp32/fp16 TensorRT benchmark variants).

### Style rules
- No `sys.path` manipulation — packages are installed via `uv sync` (hatchling src layout).
- No multi-line comments or docstrings — single-line `#` only where the WHY is non-obvious.
- No `print` — use `Logger`.
