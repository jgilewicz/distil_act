# Architecture

## Data flow

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

`task` is `reach` or `pick`; `<prefix>`/`<distil_prefix>` are that task's `checkpoint_prefix` from `training.tasks.<task>`/`distillation.tasks.<task>` (`act_model`/`distil_act_model` for reach, `pick_model`/`pick_distil_model` for pick). `distillation_measure.py` runs the same teacher→student→(measure) path headlessly for either task, per `eval.tasks.<task>.measure`.

Quantization branch (`--task {reach,pick}`) — export to ONNX is a separate step from PTQ itself; both write ONNX, measured through either ONNX Runtime or a built/cached TensorRT engine:

```
<distil_prefix>_final.pt ──> export_onnx.py --task <task> ──> <distil_prefix>_32.onnx  (fp32, torch.onnx.export)
                                                         └──> <distil_prefix>_16.onnx  (fp16, modelopt autocast)
                                                                ↓
                            <distil_prefix>_32.onnx ──> ptq.py --task <task> ──> <distil_prefix>_ptq.onnx       (static QDQ int8, ONNX Runtime)
                                                                            └──> <distil_prefix>_dyn.onnx       (dynamic weight-only int8, ONNX Runtime)
                                                                            └──> <distil_prefix>_int8_qdq.onnx (int8 QDQ, modelopt.onnx.quantization.quantize,
                                                                                                                  calibrated from a .npz built by build_calibration_npz)
                                                                                        ↓
                                        distillation_measure.py — OnnxModel (ONNX Runtime, CPU) for the four ONNX variants;
                                        TensorRtModel (utils.tensorrt.build_engine, cached to engine_{fp32,fp16,int8}_path) for the three TensorRT variants
```

## Key files

**`src/env/base.py` — `Environment`, `build_model`**
- `Environment`: abstract base class with `reset() -> np.ndarray` and `step(action) -> (obs, terminated)`.
- `build_model(scene_xml_path, gripper_kp, gripper_kv)`: shared by `ReachEnvironment`/`PickEnvironment` — merges a scene XML with the `piper` MJCF (`robot_` prefix), tunes the gripper actuator's gain/bias, and attaches `ego_cam` to `robot_link6`.

**`src/env/reach_env.py` — `ReachEnvironment(Environment)`**
- Merges `models/reach_scene.xml` with the `piper` (AgileX PiPER) MJCF from `robot_descriptions` at runtime; attaches an `ego_cam` to `robot_link6` (the wrist link just before the gripper fingers — stable regardless of gripper state).
- All robot bodies/joints/actuators are prefixed `robot_` after merging. PiPER has 8 qpos dims (`joint1`–`joint6` arm + `joint7`/`joint8` gripper fingers) but only 7 actuators (`nu=7`) — `joint8` mirrors `joint7` via an MJCF equality constraint and has no actuator of its own.
- `step(action)` returns `(obs, terminated)` — terminated when EE distance < `reach_threshold`.
- Observation vector: `[qpos(8), qvel(8), ee_pos(3), target_pos(3)]`. Target is a mocap body.

**`src/env/pick_env.py` — `PickEnvironment(Environment)`**
- Second task: pick up a free-floating `box` body and lift it clear of the table — no placement target. `step(action)` terminates when the box's z position rises above `box_height + lift_height` (a margin below `PickExpert`'s actual lift height, so termination doesn't require the exact final trajectory waypoint).
- Observation vector: `[qpos, qvel, box_pos(3)]`.
- `reset()` randomizes the box's starting `(x, y)` within its configured range; z is always `box_height`.

**`src/expert/reach_expert.py` — `ReachExpert`** / **`src/expert/pick_expert.py` — `PickExpert`**
- Both use NVIDIA `curobo` against `models/piper.urdf`; reach uses `InverseKinematics` (position-only tool-frame IK), pick uses `MotionPlanner`/`plan_grasp` (a four-phase `approach → grasp → close → lift` state machine over `models/pick.yaml`'s collision world). Gripper ctrl is set directly per phase, never part of the IK solve. See [curobo.md](curobo.md) for the full patterns (tool-frame gotcha, position-only IK, joint ordering).

**`src/renderer/renderer.py` — `SceneRenderer`**
- Context manager; `render_step(action)` returns `(obs, terminated, frames)` where `frames` is a `dict[camera_name → BGR ndarray]`.
- Available cameras: `overhead_cam` (scene XML) and `ego_cam` (gripper-mounted).

**`src/dataset/`**
- `EpisodeRecorder`: single HDF5 file with resizable datasets `frames` (uint8, shape `T×K×H×W×3`), `joints` (float32), `timestamps` (float64). K = number of cameras.
- `DatasetManager`: manages `root_dir/episodes/episode_N.h5` layout; `img_shape` must be `(K, H, W, 3)`; auto-increments index from existing files; writes `metadata.json`.
- `EpisodeDataset(cfg, task, split)` (`src/dataset/dataloader.py`): PyTorch `Dataset`; `__getitem__` returns `{"images": (K,3,H,W), "qpos": (joint_dim,), "actions": (chunk_size, action_dim)}`; batched to `(B,K,3,H,W)`. `qpos` is the full recorded joints vector; `actions` is sliced to `[action_qpos_offset : action_qpos_offset + action_dim]` from that same recorded-joints array (for reach this slice is the identity). `make_dataloader(cfg, task, split)` returns a configured `DataLoader`.
- Joint positions are z-score normalised using `mean`/`std` computed across the full training split (`joint_dim`-sized, for `qpos`); `action_mean`/`action_std` are the corresponding slice of those same stats (`action_dim`-sized, for `actions`). All four are saved into the checkpoint (`norm_mean`/`norm_std`/`action_norm_mean`/`action_norm_std`) for inference — the eval scripts normalise the input `qpos` with the first pair and denormalise the predicted `action` with the second before sending it to `env.step()`.

**`src/algorithms/`**
- `embedding.py` — `ImageEmbedding`: frozen EfficientNet-B3 backbone + AdaptiveAvgPool → linear projection; adds per-camera and positional embeddings; output shape `(B, K*P, embed_dim)` where P=49 patches. On MPS, the AdaptiveAvgPool runs on CPU (non-divisible sizes unsupported on MPS).
- `act_policy.py` — `ACT`: full encoder-decoder Transformer. Training: takes `(images, qpos, actions)`, runs CVAE encoder for latent z, returns `(pred_actions, mu, logvar)`. Inference: z=0, returns `pred_actions` only.
- `chunking_buffer.py` — `ChunkingBuffer`: stores overlapping action chunk predictions; `get_action(t)` returns exponentially weighted average over all chunks that cover timestep t; evicts chunks older than `chunk_size` steps.

**`scripts/eval/distillation_measure.py --task {reach,pick}`**
- Evaluates the fp32 teacher and student over `eval.tasks.<task>.measure.n_episodes` randomly-seeded episodes, plus that task's four ONNX Runtime variants (`student_ptq`, `student_dyn`, `student_fp16_onnx`, `student_int8_qdq`) and three TensorRT engine variants (`student_trt_fp32`, `student_trt_fp16`, `student_trt_int8`), reading paths from `cfg["ptq"]["tasks"][task]`.
- fp32 models load via `load_model` (torch); the ONNX Runtime variants via `load_quantized_model` → `OnnxModel` (an `onnxruntime.InferenceSession` wrapper) on CPU; the TensorRT variants via `load_tensorrt_model` → `TensorRtModel`, which builds/caches an engine with `utils.tensorrt.build_engine` and runs it via `TensorRtRuntime.infer`. All loaders return `(model, norm_mean, norm_std, action_norm_mean, action_norm_std)` — non-torch variants reuse the student checkpoint's four norm-stat tensors.
- Per-episode (`run_episode`): builds the task's `Environment` via `ENV_CLASSES[task]`, runs the full policy loop headlessly, records inference times, joint trajectories, EE positions, and success — predicted actions are denormalised with `action_norm_mean`/`action_norm_std` and applied directly as `ctrl`.
- Aggregates: success rate, mean convergence time, mean inference latency, model size (engine file size for TRT variants), peak VRAM/RAM. Writes `eval.tasks.<task>.measure.output_path` (JSON) and logs to `.log_file`.

**`scripts/quantization/export_onnx.py --task {reach,pick}`**
- Exports the distilled student's inference path (`actions=None`) to ONNX; loading the checkpoint and building the model is only done here, not in `ptq.py`.
- Exports fp32 via `torch.onnx.export(..., dynamo=True)` (input names `images`/`joints`) to `ptq.tasks.<task>.fp32_path`, then converts that to a mixed-precision fp16 graph via `modelopt.onnx.autocast.convert_to_mixed_precision` (`keep_io_types=True`) to `ptq.tasks.<task>.fp16_path`. TensorRT 11.x removed the builder-side `BuilderFlag.FP16`/`INT8` flags in favor of this offline ModelOpt AutoCast pass — precision is now baked into the ONNX graph (explicit `Cast` nodes) before it reaches TensorRT or ONNX Runtime.

**`scripts/quantization/ptq.py --task {reach,pick}`**
- Post-training quantization of that task's already-exported fp32 ONNX (`ptq.tasks.<task>.fp32_path`), on CPU. Does not touch the checkpoint or the model class — run `export_onnx.py --task <task>` first.
- Runs `quant_pre_process(skip_symbolic_shape=True)` (symbolic shape inference chokes on the transformer's `Loop` node), then produces three int8 models: static QDQ via `quantize_static` + an `ActCalibrationReader` over the val split (`output_path`); dynamic weight-only via `quantize_dynamic` (`dyn_path`) — before which it strips `graph.value_info` (the dynamo export's stale shapes trip strict shape inference); and int8 QDQ via `modelopt.onnx.quantization.quantize` (`int8_qdq_path`), calibrated from a `.npz` (`calib_npz_path`) that `build_calibration_npz` stacks from the same val split.

**`scripts/training/train_act.py`** / **`scripts/training/train_distil.py`** (`--task {reach,pick}`)
- `train_act.py` trains the ACT teacher; `train_distil.py` loads the frozen teacher (`training.tasks.<task>` dims, from `distillation.tasks.<task>.teacher.checkpoint`) and trains a smaller student `ACT` (`distillation` dims, `distil_act=True` → MobileNetV3-Large, same `action_dim`/`joint_dim` as the teacher). See [distillation.md](distillation.md) for the loss. Both write checkpoints carrying the four norm-stat tensors.

**`scripts/eval/eval_act.py --task {reach,pick}`** / **`scripts/eval/eval_distil.py --task {reach,pick}`**
- Load `eval.tasks.<task>.teacher.checkpoint` / `.student.checkpoint` (weights + the four norm-stat tensors). The student script builds its `ACT` with `distillation` dims and `distil_act=True`.
- Build the task's `Environment` via `ENV_CLASSES[task]` from `collect.tasks.<task>.env`. Query ACT every `chunk_size // 5` physics steps; `ChunkingBuffer` provides temporally ensembled actions. Predicted actions are denormalised with `action_norm_mean`/`action_norm_std` and applied as `env.step()`'s `ctrl`. Render passive viewer via `SceneRenderer`; write `video_path` from the overhead camera.

**`src/utils/`**
- `logger.py` — `Logger(filename)`: logs `[INFO]`/`[WARNING]`/`[ERROR]` to stdout and file simultaneously.
- `config.py` — `load_config(config_dir="config")`: merges every `*.yaml` in the directory into one dict; raises on duplicate top-level keys.
- `hub.py` — `ensure_checkpoint`/`ensure_dataset` (download if missing) and `push_checkpoint`/`push_dataset` (upload). Only place `huggingface_hub` is imported; see [hugging-face.md](hugging-face.md).
- `tasks.py` — `ENV_CLASSES: dict[str, type[Environment]]` mapping `"reach"`/`"pick"` to `ReachEnvironment`/`PickEnvironment`. Shared by `eval_act.py`, `eval_distil.py`, `distillation_measure.py`, and `collect_data.py`'s `TaskSpec` registry — the single place a new task's env class gets registered.
- `tensorrt.py` — `build_engine(onnx_path, engine_path)` (parses an ONNX graph, builds a serialized TensorRT engine, caches it) and `TensorRtRuntime` (wraps engine deserialization + `infer(inputs)`: raw `cuda-python`/`cudart` H2D copy → `execute_async_v3` → D2H copy). Shared between `scripts/quantization/inference.py` (standalone CLI smoke test) and `distillation_measure.py`.

## Project structure

```
src/
  env/              # MuJoCo reach + pick environments
  expert/           # curobo-based scripted experts (reach + pick) + abstract base
  renderer/         # off-screen rendering + passive viewer
  dataset/          # HDF5 episode recording + PyTorch dataset/dataloader
  algorithms/       # ACT policy, ImageEmbedding, ChunkingBuffer
  utils/            # logger, config loader, Hub helpers, task→Environment registry, TensorRT
scripts/
  collection/collect_data.py       # expert demo collection, --task {reach,pick}
  training/train_act.py            # ACT teacher training loop, --task {reach,pick}
  training/train_distil.py         # student distillation loop, --task {reach,pick}
  eval/eval_act.py                  # teacher evaluation + video export, --task {reach,pick}
  eval/eval_distil.py               # student evaluation + video export, --task {reach,pick}
  eval/distillation_measure.py      # teacher / student / quantized / TensorRT benchmark
  quantization/export_onnx.py       # export distilled student → fp32 + fp16 ONNX
  quantization/ptq.py               # PTQ → ONNX Runtime static/dynamic + modelopt int8 QDQ
  quantization/inference.py         # TensorRT engine build + inference
  hub/push_data_to_hub.py           # manual dataset push, --task {reach,pick}
  hub/push_teacher_to_hub.py        # manual teacher checkpoint push
  hub/push_student_to_hub.py        # manual student checkpoint push
models/
  reach_scene.xml
  pick_scene.xml
  pick.yaml         # curobo collision world for pick (table + box), mirrors pick_scene.xml
  piper.urdf        # AgileX PiPER kinematics, converted from the robot_descriptions MJCF - curobo IK input
  meshes/           # piper.urdf mesh assets
tests/
config/
  collect.yaml      # per-task (reach, pick) env, expert, collection
  simulation.yaml   # shared renderer
  train.yaml        # per-task teacher training (dims, checkpoint, hub) + shared architecture/optimizer
  distill.yaml      # per-task student distillation (teacher, checkpoint, hub) + shared architecture/loss weights
  eval.yaml         # per-task evaluation + measurement (teacher, student, measure)
  quant.yaml        # quantization (ptq, per-task tasks.reach/tasks.pick)
justfile
Docker/Dockerfile
.github/workflows/ci.yml
```

## Style rules
- No `sys.path` manipulation — packages are installed via `uv sync` (hatchling src layout).
- No multi-line comments or docstrings — single-line `#` only where the WHY is non-obvious.
- No `print` — use `Logger`.
