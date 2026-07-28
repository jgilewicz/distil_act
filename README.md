# ACT Distillation

A research project exploring in-simulation imitation learning and policy distillation. A scripted expert collects demonstrations in MuJoCo, an ACT teacher is trained via behaviour cloning, and a smaller student is distilled from it for edge deployment — all within a single, config-driven pipeline.

The reach task implemented here is a proof of concept. The env/expert layer is deliberately thin and abstract (`Environment` + `Expert` base classes), so the rest of the pipeline — data collection, ACT training, distillation, evaluation — carries over to any new task without modification. A natural next step would be packaging this as a Python library for general in-simulation policy learning and distillation.

## Stack

- **Simulation** — MuJoCo 3 + `low_cost_robot_arm` via `robot_descriptions`
- **IK** — `mink` with `daqp` solver
- **Dataset** — HDF5 (`h5py`), multi-camera frames + joints + timestamps per episode; hosted on the Hugging Face Hub
- **Teacher policy** — ACT in PyTorch; EfficientNet-B3 image backbone, CVAE encoder, Transformer encoder-decoder
- **Student policy** — same ACT architecture, smaller dims + MobileNetV3-Large backbone; trained with a hard action loss, a soft loss against the teacher's predictions, and a latent-space distillation KL
- **Training** — AdamW + linear warmup + cosine decay; KL-weighted ELBO loss; logged to W&B
- **Quantization** — ONNX Runtime post-training quantization (static QDQ + dynamic int8), exported to ONNX
- **Inference** — `ChunkingBuffer` temporal ensembling over overlapping action chunks

## The pipeline

The pipeline is four separate stages — run them one at a time, in order. Nothing chains automatically into the next stage; each `just` recipe does exactly one job.

```
just collect   →   just train   →   just distill   →   just eval / just eval-distill
 (gather demos)    (train teacher)   (distill student)   (watch it run, save video)
```

```bash
uv sync                  # install / sync dependencies

just collect-headless    # 1. gather expert demonstrations (no viewer)
just train               # 2. train the ACT teacher on the dataset
just distill             # 3. distill the teacher into a smaller student
just eval                # 4a. watch the teacher policy, save a video
just eval-distill        # 4b. watch the distilled student policy, save a video
```

Each stage reads its own settings from `config/`, downloads whatever inputs it needs from the Hugging Face Hub automatically if they're not already on disk, and writes its own logs. See [Configuration](#configuration) below.

## Configuration

All settings live under `config/`, split into one file per pipeline stage so you can jump straight to what you're changing instead of scrolling one huge file:

| File | Section(s) | Used by |
|---|---|---|
| `config/simulation.yaml` | `reach_env.py`, `expert`, `renderer` | collection, both eval scripts |
| `config/collect_reach.yaml` | `collection` | `collect_data.py`, `push_data_to_hub.py` |
| `config/train.yaml` | `training` | `train_act.py`, `push_teacher_to_hub.py` |
| `config/distill.yaml` | `distillation` | `train_distil.py`, `push_student_to_hub.py` |
| `config/eval.yaml` | `eval` | `eval_act.py`, `eval_distil.py`, `distillation_measure.py` |
| `config/quant.yaml` | `ptq` | `export_onnx.py`, `ptq.py` |

`load_config()` merges every `*.yaml` in `config/` into a single dict, so scripts just do `cfg["training"]`, `cfg["eval"]["teacher"]`, etc. regardless of which file the section lives in.

### Hugging Face Hub automation

Every stage that touches a checkpoint or dataset has a `hub` block in its config:

```yaml
hub:
  repo_id: "jgilewicz/distil_act_teacher"
  filename: "act_model_final.pt"
  auto_push: false   # upload once this stage finishes
```

(pull-side blocks — `collection.hub`, `distillation.teacher`, `eval.teacher`/`eval.student` — use `auto_pull` instead of `auto_push`.)

- **`auto_pull: true`** (default) — if the local dataset/checkpoint is missing, it's downloaded from `repo_id` before the stage runs. Nothing to do manually.
- **`auto_push: true`** — once the stage finishes (dataset collected, training/distillation complete), the result is uploaded to `repo_id` automatically.

Leave `auto_push` off and use `just push-data` / `just push-teacher` / `just push-student` whenever you want to push on demand instead (e.g. re-pushing an existing checkpoint without retraining). Requires `huggingface-cli login` first.

Logging is likewise automatic and config-driven: every stage writes to the `log_file` path set in its own config section via `Logger` — nothing to wire up per run.

## Commands

```bash
just                     # list all available tasks
just collect             # collect demos with viewer (macOS: uses mjpython)
just collect-headless    # collect headless
just train               # train the ACT teacher (logs to W&B, saves to artifacts/)
just distill             # distill the teacher into a smaller student
just eval                # run the trained teacher with viewer (macOS: uses mjpython)
just eval-distill        # run the distilled student with viewer (macOS: uses mjpython)
just export-onnx         # export the distilled student to ONNX: fp32 + fp16 (modelopt autocast)
just ptq                 # post-training quantization (ONNX): static + dynamic int8 student, from the exported fp32 ONNX
just measure             # compare teacher, student, and quantized variants: success rate, latency, size, VRAM/RAM
just push-data           # push the collected dataset to the Hub
just push-teacher        # push the teacher checkpoint to the Hub
just push-student        # push the student checkpoint to the Hub
just test                # run test suite
just lint                # ruff check
just fix                 # ruff check --fix + ruff format
just clean               # remove generated logs, dataset, and pycache
```

## Training

Training reads its dataset from `collection.dataset_dir` (auto-pulled from the Hub if absent, per `collection.hub.auto_pull`) and writes into `training.checkpoint_dir`:

- `act_model_step_<N>.pt` — periodic checkpoints (state dict only)
- `act_model_final.pt` — final checkpoint including `norm_mean` / `norm_std` for inference

Set `WANDB_API_KEY` in a `.env` file or shell environment before running.

```bash
cp .env.example .env   # fill in WANDB_API_KEY
just train
```

## Distillation

`just distill` loads the teacher checkpoint (`distillation.teacher.checkpoint`, auto-pulled from `distillation.teacher.repo_id` if missing) and trains a smaller student — same ACT architecture at reduced `embed_dim`/`latent_dim`/`num_layers` with a MobileNetV3-Large backbone instead of EfficientNet-B3. The student's latent is projected up to `teacher_latent_dim` and matched against the teacher's CVAE posterior (`distillation_kl`), alongside a hard action loss and a soft loss against the teacher's predictions.

Writes `distil_act_model_step_<N>.pt` / `distil_act_model_final.pt` into `distillation.checkpoint_dir`, same layout as the teacher.

## Quantization

Compressing the distilled student for edge deployment is two separate steps — export to ONNX, then quantize. Settings for both live in `config/quant.yaml` (`ptq` section); outputs land in `artifacts/`.

```bash
just export-onnx   # distil_act_model_final.pt → fp32 ONNX (torch.onnx.export) + fp16 ONNX (modelopt autocast)
just ptq           # fp32 ONNX → static + dynamic int8 (CPU, ONNX Runtime)
```

`export_onnx.py` loads the checkpoint and exports the student's inference path to fp32 ONNX, then converts it to a mixed-precision fp16 graph via `modelopt.onnx.autocast` — the offline replacement for TensorRT's removed `BuilderFlag.FP16`/`INT8` builder flags (TensorRT 11.x now expects precision baked into the ONNX graph ahead of time).

`ptq.py` only touches the already-exported fp32 ONNX (run `export-onnx` first) and writes two int8 models, both benchmarked by `just measure` alongside the fp32 models:

- **static** QDQ (`distil_act_model_ptq.onnx`) — calibrated on the validation split.
- **dynamic** weight-only (`distil_act_model_dyn.onnx`) — no calibration.

The quantized ONNX models carry no normalisation stats — the measurement loads `norm_mean`/`norm_std` from the student checkpoint and runs the ONNX graphs on CPU via ONNX Runtime.

## Evaluation

```bash
just eval            # teacher
just eval-distill     # student
```

Each loads its checkpoint (auto-pulled per `eval.teacher`/`eval.student.auto_pull`), runs the policy in the MuJoCo reach environment, renders the passive viewer, and writes a video to `eval.teacher.video_path` / `eval.student.video_path` (overhead camera).

The policy is queried every `chunk_size // 5` physics steps; `ChunkingBuffer` handles temporal ensembling for intermediate steps.

## Measurement

```bash
just measure
```

Runs `scripts/eval/distillation_measure.py`: evaluates the fp32 teacher and student, three ONNX Runtime variants (`student_ptq`, `student_dyn`, `student_fp16_onnx`), and two TensorRT variants (`student_trt_fp32`, `student_trt_fp16`) over `eval.measure.n_episodes` randomly-seeded episodes and writes `artifacts/distillation_metrics.json` with, per model:

- `success_rate` — fraction of episodes where the EE reached the target
- `mean_convergence_time_s` — average sim time for successful episodes
- `mean_inference_time_ms` — average forward-pass latency
- `model_size_mb` — checkpoint size on disk
- `vram_mb` — peak VRAM on CUDA (`torch.cuda.max_memory_allocated`, reset per variant)
- `ram_delta_mb` — growth in process RSS high-water mark (`ru_maxrss`) attributable to this variant's episodes; not an isolated peak, since all variants share one process
- `joint_mean` / `joint_std` / `ee_pos_mean` / `ee_pos_std` — trajectory statistics across all episodes

## Results

**Stale — predates fixes to `distillation_measure.py`'s CUDA timing sync, TensorRT engine reuse, and RSS accounting; re-run `just measure` to regenerate.**

Evaluated over 20 randomly-seeded episodes on an x86 machine. The fp32 teacher and student run on **CUDA** (torch); `student_ptq`/`student_dyn`/`student_fp16_onnx` run on **CPU** via ONNX Runtime; `student_trt_fp32`/`student_trt_fp16` run on **CUDA** via TensorRT — so latency is only directly comparable within each group.

| Metric | Teacher (fp32, GPU) | Student (fp32, GPU) | Student fp16 (ONNX, CPU) | Student PTQ static (int8, CPU) | Student PTQ dynamic (int8, CPU) | Student TRT (fp32, GPU) | Student TRT (fp16, GPU) |
|---|---|---|---|---|---|---|---|
| Success rate | 95% | 80% | 85% | 55% | 30% | 80% | 85% |
| Mean convergence time † | 0.77 s | 0.95 s | 0.95 s | 0.77 s | 0.64 s | 0.94 s | 0.93 s |
| Inference latency | 23.3 ms | 11.7 ms | 61.3 ms | 92.8 ms | 394.9 ms | 192.5 ms | 170.6 ms |
| Model size | 107.6 MB | 26.3 MB | 10.9 MB | 6.7 MB | 6.4 MB | 23.9 MB | 12.6 MB |
| Peak VRAM | 307.0 MB | 125.8 MB | — | — | — | — | — |
| Peak RAM | 2193.2 MB | 2241.5 MB | 2562.4 MB | 2482.0 MB | 2560.2 MB | 3948.7 MB | 5251.3 MB |

† Averaged only over *successful* episodes, so columns with fewer successes (e.g. the 30% `student_dyn` row) are noisier and less comparable.

**Distillation is a solid compression win, at some accuracy cost.** The student is **4.1× smaller** and **~2× faster** than the teacher on GPU, using ~2.4× less VRAM, but success rate drops from 95% to 80% — the CVAE-distilled policy generalizes slightly worse than the teacher it was trained from.

**int8 PTQ still trades accuracy for size, but per-channel calibration narrows the gap.** Static QDQ int8 (`student_ptq`) now reaches 55% success (up from a per-tensor baseline that collapsed to ~10%) at 6.7 MB — **16× smaller than the teacher, 3.9× smaller than the fp32 student** — while dynamic weight-only quantization (`student_dyn`) is both less accurate (30%) and far slower (394.9 ms), since its unoptimised CPU kernels do more work per call than the statically calibrated QDQ graph. Both remain well below the fp32 student's 80% success rate.

**fp16 is the best accuracy/size trade-off of the quantized variants.** The plain fp16 ONNX graph (`student_fp16_onnx`, CPU) matches or beats the fp32 student's success rate (85% vs 80%) at less than half the size (10.9 MB), though at ~5× the latency of the GPU fp32 student since it runs on CPU.

**TensorRT is not a latency win here.** Both TRT engines match or beat the fp32 student on success rate (80% / 85%) and shrink further on disk, but their measured inference latency (170–193 ms) is far higher than either the torch fp32 student (11.7 ms) or the CPU ONNX Runtime variants — most likely dominated by per-call H2D/D2H copy overhead in the raw `cuda-python` inference path (`src/utils/tensorrt.py`) rather than the engine's actual compute time, since this model's per-step batch is tiny. TRT fp16 (12.6 MB) does halve the engine size and cut latency versus TRT fp32 (23.9 MB), consistent with fp16 compute/memory savings once that fixed overhead is factored in.

Bottom line: distillation remains the headline compression step; among the post-training quantization options, **fp16 ONNX gives the best size/accuracy balance without a GPU**, per-channel int8 PTQ is usable but lossy, dynamic int8 is not recommended, and the current TensorRT integration needs its inference path optimized (batched calls, persistent buffers) before its latency numbers are meaningful.

## Headless rendering (no display)

MuJoCo reads `MUJOCO_GL` **at import time** — it must be set in the shell before the process starts.

`just collect-headless` already sets `MUJOCO_GL=egl`. If you run the script directly:

```bash
MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collection/collect_data.py
```

### Choosing a backend

| Backend | When to use | Requirement |
|---------|-------------|-------------|
| `egl` | GPU or any Mesa EGL (recommended for servers) | `libegl1` |
| `osmesa` | No GPU / EGL unavailable (software fallback) | `libosmesa6` |
| `disabled` | Physics-only, no rendering at all | nothing |

Install EGL (Mesa, CPU-only machines):

```bash
apt-get install -y libegl1 libgl1
```

If EGL still fails (`gladLoadGL error`), fall back to OSMesa:

```bash
apt-get install -y libosmesa6
MUJOCO_GL=osmesa SHOW_VIEWER=false uv run python3 scripts/collection/collect_data.py
```

The Docker image ships with `libegl1` + `libgl1` and sets `MUJOCO_GL=disabled` for tests (physics only). Switch it to `egl` for any container that needs to render frames.

## Porting to a new task

The pipeline above the env/expert layer is fully task-agnostic — dataset recording, ACT training, distillation, evaluation, and all utilities need zero changes. Only two files need to be written and two scripts need their imports swapped.

### 1. New environment

Implement the same interface as `ReachEnvironment`:

```python
# src/env/my_env.py
from env.base import Environment

class MyEnvironment(Environment):
    def reset(self) -> np.ndarray:
        # randomise initial state / target, return obs vector
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        # apply action, step physics; bool = task success / termination
        ...
```

The obs vector must have `qpos` in its first `joint_dim` elements — the eval scripts index `obs[:joint_dim]` to extract joint positions for normalisation.

### 2. New expert

Extend the `Expert` ABC in `src/expert/base.py`:

```python
# src/expert/my_expert.py
from expert.base import Expert

class MyExpert(Expert):
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        # deterministic obs → joint control vector
        # any method works: IK, analytical solution, motion primitive
        ...
```

The expert only needs to produce good-enough demonstrations — it is discarded after data collection.

### 3. Config (`config/simulation.yaml`)

Replace the reach-specific env block with your task's parameters and point `scene_xml_path` at your MuJoCo XML. If your robot has a different DOF, also update `action_dim` and `joint_dim` in `config/train.yaml` and `config/distill.yaml`.

### 4. Swap imports in two scripts

`scripts/collection/collect_data.py` and `scripts/eval/eval_act.py` / `scripts/eval/eval_distil.py` import `ReachEnvironment` and `ReachExpert` directly — replace those with your new classes. Everything else (`train_act.py`, `train_distil.py`, `distillation_measure.py`, all of `src/`) is unchanged.

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
  collection/
    collect_data.py          # expert demo collection
  training/
    train_act.py             # ACT teacher training loop
    train_distil.py          # student distillation loop
  eval/
    eval_act.py               # teacher evaluation + video export
    eval_distil.py            # student evaluation + video export
    distillation_measure.py   # teacher / student / quantized benchmark (success, latency, memory)
  quantization/
    export_onnx.py            # export distilled student → fp32 + fp16 ONNX
    ptq.py                     # post-training quantization → static + dynamic int8 ONNX
    inference.py               # TensorRT engine build + inference
  hub/
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
