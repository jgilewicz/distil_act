# ACT Distillation

A research project exploring in-simulation imitation learning and policy distillation. A scripted expert collects demonstrations in MuJoCo, an ACT teacher is trained via behaviour cloning, and a smaller student is distilled from it for edge deployment — all within a single, config-driven pipeline.

The reach task implemented here is a proof of concept. The env/expert layer is deliberately thin and abstract (`Environment` + `Expert` base classes), so the rest of the pipeline — data collection, ACT training, distillation, evaluation, measurement — carries over to any new task without modification. A second task, pick-and-place (box pickup + placement, `src/env/pick_and_place_env.py` + `src/expert/pick_and_place_expert.py`), follows this same pattern. The rest of the pipeline (`just train/distill/eval[-distill]/measure pick_and_place`) is wired up, but `PickAndPlaceExpert` is currently a stub (`compute_action` raises `NotImplementedError`) pending migration to the new IK backend, so `just collect pick_and_place` doesn't work yet. Quantization (`export_onnx.py`/`ptq.py`) remains reach-only. A natural next step would be packaging this as a Python library for general in-simulation policy learning and distillation.

## Stack

- **Simulation** — MuJoCo 3 + `piper` (AgileX PiPER, 6-DOF arm + gripper) via `robot_descriptions`
- **IK** — NVIDIA `curobo`, position-only tool-frame IK against `models/piper.urdf` (reach only — pick-and-place's expert is currently unimplemented)
- **Dataset** — HDF5 (`h5py`), multi-camera frames + joints + timestamps per episode; hosted on the Hugging Face Hub
- **Teacher policy** — ACT in PyTorch; EfficientNet-B3 image backbone, CVAE encoder, Transformer encoder-decoder
- **Student policy** — same ACT architecture, smaller dims + MobileNetV3-Large backbone; trained with a hard action loss, a soft loss against the teacher's predictions, and a latent-space distillation KL
- **Training** — AdamW + linear warmup + cosine decay; KL-weighted ELBO loss; logged to W&B
- **Quantization** — ONNX Runtime post-training quantization (static QDQ + dynamic int8), exported to ONNX
- **Inference** — `ChunkingBuffer` temporal ensembling over overlapping action chunks

## The pipeline

The pipeline is four separate stages — run them one at a time, in order. Nothing chains automatically into the next stage; each `just` recipe does exactly one job.

```
just collect reach   →   just train reach   →   just distill reach   →   just eval reach / just eval-distill reach
 (gather demos)          (train teacher)        (distill student)        (watch it run, save video)
```

```bash
uv sync                          # install / sync dependencies

just collect-headless reach      # 1. gather expert demonstrations (no viewer)
just train reach                 # 2. train the ACT teacher on the dataset
just distill reach                # 3. distill the teacher into a smaller student
just eval reach                   # 4a. watch the teacher policy, save a video
just eval-distill reach           # 4b. watch the distilled student policy, save a video
```

`just collect[-headless]`, `just train`, `just distill`, `just eval[-distill]`, and `just measure` all take `reach` or `pick_and_place`. Collection, teacher training, distillation, evaluation, and benchmarking are fully task-agnostic end to end. Quantization (`just export-onnx` / `just ptq`, `config/quant.yaml`) remains **reach-only** — `just measure pick_and_place` skips the quantized/TensorRT variants accordingly.

Pick-and-place's recorded qpos mixes the box's free-floating pose (7 dims: pos + quat, unactuated) with the robot's 6 actuated joints — the box comes first in the compiled model, so `joint_dim=13` (full state, used as model context) but `action_dim=6` (robot-only, sliced out of that same recorded qpos at `action_qpos_offset=7` — see `config/train.yaml`). That's what the model actually predicts and what gets sent to `env.step()` as `ctrl`.

Each stage reads its own settings from `config/`, downloads whatever inputs it needs from the Hugging Face Hub automatically if they're not already on disk, and writes its own logs. See [Configuration](#configuration) below.

## Configuration

All settings live under `config/`, split into one file per pipeline stage so you can jump straight to what you're changing instead of scrolling one huge file:

| File | Section(s) | Used by |
|---|---|---|
| File | Section(s) | Used by |
|---|---|---|
| `config/collect.yaml` | `collect.tasks.reach`, `collect.tasks.pick_and_place` (each: `env`, `expert`, `collection`) | `collect_data.py --task <task>`, `push_data_to_hub.py --task <task>` |
| `config/simulation.yaml` | `renderer` | all collection/eval/measure scripts |
| `config/train.yaml` | `training.tasks.reach`, `training.tasks.pick_and_place` (each: `action_dim`, `joint_dim`, `action_qpos_offset`, `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`) plus shared architecture/optimizer keys | `train_act.py --task <task>`, `push_teacher_to_hub.py --task <task>` |
| `config/distill.yaml` | `distillation.tasks.reach`, `distillation.tasks.pick_and_place` (each: `teacher`, `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`) plus shared student architecture/loss-weight keys | `train_distil.py --task <task>`, `push_student_to_hub.py --task <task>` |
| `config/eval.yaml` | `eval.tasks.reach`, `eval.tasks.pick_and_place` (each: `teacher`, `student`, `measure`) | `eval_act.py --task <task>`, `eval_distil.py --task <task>`, `distillation_measure.py --task <task>` |
| `config/quant.yaml` | `ptq` (reach-only, unparameterized) | `export_onnx.py`, `ptq.py` |

`load_config()` merges every `*.yaml` in `config/` into a single dict, so scripts just do `cfg["training"]["tasks"]["reach"]`, `cfg["eval"]["tasks"][task]["teacher"]`, `cfg["collect"]["tasks"][task]["env"]`, etc. regardless of which file the section lives in. Every task-aware file uses identical key names nested under the task name — the only thing that switches between reach and pick-and-place is which task key you read (or pass via `--task`). `src/utils/tasks.py`'s `ENV_CLASSES` dict maps a task name to its `Environment` subclass and is shared by the collection, eval, and measurement scripts.

### Hugging Face Hub automation

Every stage that touches a checkpoint or dataset has a `hub` block in its config:

```yaml
hub:
  repo_id: "jgilewicz/distil_act_teacher"
  filename: "act_model_final.pt"
  auto_push: false   # upload once this stage finishes
```

(pull-side blocks — `collect.tasks.<task>.collection.hub`, `distillation.tasks.<task>.teacher`, `eval.tasks.<task>.teacher`/`.student` — use `auto_pull` instead of `auto_push`.)

- **`auto_pull: true`** (default) — if the local dataset/checkpoint is missing, it's downloaded from `repo_id` before the stage runs. Nothing to do manually.
- **`auto_push: true`** — once the stage finishes (dataset collected, training/distillation complete), the result is uploaded to `repo_id` automatically.

Leave `auto_push` off and use `just push-data <task>` / `just push-teacher <task>` / `just push-student <task>` whenever you want to push on demand instead (e.g. re-pushing an existing checkpoint without retraining). Requires `huggingface-cli login` first.

Logging is likewise automatic and config-driven: every stage writes to the `log_file` path set in its own config section via `Logger` — nothing to wire up per run.

## Commands

```bash
just                              # list all available tasks
just collect <task>               # collect demos with viewer (macOS: uses mjpython); task = reach | pick_and_place
just collect-headless <task>      # collect demos headless; task = reach | pick_and_place
just train <task>                 # train the ACT teacher (logs to W&B, saves to artifacts/); task = reach | pick_and_place
just distill <task>               # distill the teacher into a smaller student; task = reach | pick_and_place
just eval <task>                  # run the trained teacher with viewer (macOS: uses mjpython); task = reach | pick_and_place
just eval-distill <task>          # run the distilled student with viewer (macOS: uses mjpython); task = reach | pick_and_place
just export-onnx                  # export the distilled student to ONNX: fp32 + fp16 (modelopt autocast); reach only
just ptq                          # post-training quantization (ONNX): static + dynamic int8 student, from the exported fp32 ONNX; reach only
just measure <task>               # compare teacher, student, and (reach only) quantized variants: success rate, latency, size, VRAM/RAM
just push-data <task>             # push the collected dataset to the Hub; task = reach | pick_and_place
just push-teacher <task>          # push the teacher checkpoint to the Hub; task = reach | pick_and_place
just push-student <task>          # push the student checkpoint to the Hub; task = reach | pick_and_place
just test                         # run test suite
just lint                         # ruff check
just fix                          # ruff check --fix + ruff format
just clean                        # remove generated logs, dataset, and pycache
```

## Training

`just train <task>` reads its dataset from `collect.tasks.<task>.collection.dataset_dir` (auto-pulled from the Hub if absent, per `collect.tasks.<task>.collection.hub.auto_pull`) and writes into `training.tasks.<task>.checkpoint_dir`, using that task's `checkpoint_prefix`:

- `<prefix>_step_<N>.pt` — periodic checkpoints (state dict only)
- `<prefix>_final.pt` — final checkpoint including `norm_mean` / `norm_std` for inference (`act_model_final.pt` for reach, `pick_and_place_model_final.pt` for pick-and-place)

`action_dim`/`joint_dim`/`action_qpos_offset` are per-task: reach's recorded qpos is the 6 robot joints only (`joint_dim=action_dim=6`, `offset=0`); pick-and-place's qpos is the box's 7-dof freejoint followed by the 6 robot joints (`joint_dim=13`, `action_dim=6`, `offset=7`) — the model conditions on full state but only ever predicts/controls the robot. `EpisodeDataset` (`src/dataset/dataloader.py`) slices the recorded-qpos "actions" target to `[offset:offset+action_dim]` per task; `qpos` (the context input) stays the full `joint_dim`-length vector. Checkpoints save two independent normalisation pairs — `norm_mean`/`norm_std` (qpos, for the input) and `action_norm_mean`/`action_norm_std` (actions, for denormalising the model's output before it's sent to `env.step()`). For reach the two pairs are numerically identical (`offset=0`, `action_dim=joint_dim`). Everything else (`embed_dim`, `latent_dim`, `nhead`, `num_layers`, `chunk_size`, `lr`, …) is shared architecture/optimizer config under `training` directly.

Set `WANDB_API_KEY` in a `.env` file or shell environment before running.

```bash
cp .env.example .env   # fill in WANDB_API_KEY
just train reach
just train pick_and_place
```

## Distillation

`just distill <task>` loads that task's teacher checkpoint (`distillation.tasks.<task>.teacher.checkpoint`, auto-pulled from `.teacher.repo_id` if missing) and trains a smaller student — same ACT architecture at reduced `embed_dim`/`latent_dim`/`num_layers` with a MobileNetV3-Large backbone instead of EfficientNet-B3, and the same task-specific `action_dim`/`joint_dim` as the teacher (from `training.tasks.<task>`). The student's latent is projected up to `teacher_latent_dim` and matched against the teacher's CVAE posterior (`distillation_kl`), alongside a hard action loss and a soft loss against the teacher's predictions.

Writes `<checkpoint_prefix>_step_<N>.pt` / `<checkpoint_prefix>_final.pt` into `distillation.tasks.<task>.checkpoint_dir` (`distil_act_model_*` for reach, `pick_and_place_distil_model_*` for pick-and-place), same layout and normalisation-stats contract as the teacher.

## Quantization

Compressing the distilled student for edge deployment is two separate steps — export to ONNX, then quantize. Settings for both live in `config/quant.yaml` (`ptq` section, **reach-only**); outputs land in `artifacts/`.

```bash
just export-onnx   # distil_act_model_final.pt → fp32 ONNX (torch.onnx.export) + fp16 ONNX (modelopt autocast)
just ptq           # fp32 ONNX → static + dynamic int8 (CPU, ONNX Runtime)
```

`export_onnx.py` loads the checkpoint and exports the student's inference path to fp32 ONNX, then converts it to a mixed-precision fp16 graph via `modelopt.onnx.autocast` — the offline replacement for TensorRT's removed `BuilderFlag.FP16`/`INT8` builder flags (TensorRT 11.x now expects precision baked into the ONNX graph ahead of time).

`ptq.py` only touches the already-exported fp32 ONNX (run `export-onnx` first) and writes two int8 models, both benchmarked by `just measure` alongside the fp32 models:

- **static** QDQ (`distil_act_model_ptq.onnx`) — calibrated on the validation split.
- **dynamic** weight-only (`distil_act_model_dyn.onnx`) — no calibration.

The quantized ONNX models carry no normalisation stats — the measurement loads `norm_mean`/`norm_std`/`action_norm_mean`/`action_norm_std` from the student checkpoint and runs the ONNX graphs on CPU via ONNX Runtime.

## Evaluation

```bash
just eval reach              # teacher, reach
just eval-distill reach      # student, reach
just eval pick_and_place     # teacher, pick-and-place
```

Each loads its checkpoint (auto-pulled per `eval.tasks.<task>.teacher`/`.student.auto_pull`), builds the task's `Environment` (`src/utils/tasks.py`'s `ENV_CLASSES[task]`), runs the policy, renders the passive viewer, and writes a video to `eval.tasks.<task>.teacher.video_path` / `.student.video_path` (overhead camera).

The policy is queried every `chunk_size // 5` physics steps; `ChunkingBuffer` handles temporal ensembling for intermediate steps. The model's predicted action is denormalised with the task's `action_norm_mean`/`action_norm_std` and applied directly as `env.step()`'s `ctrl` — for pick-and-place this is already robot-only (see [Training](#training)), so no extra slicing is needed at eval time.

## Measurement

```bash
just measure reach
just measure pick_and_place
```

Runs `scripts/eval/distillation_measure.py --task <task>`: evaluates the fp32 teacher and student over `eval.tasks.<task>.measure.n_episodes` randomly-seeded episodes, and writes `eval.tasks.<task>.measure.output_path` (`artifacts/distillation_metrics_<task>.json`) with, per model:

- `success_rate` — fraction of episodes that reached the task's termination condition (EE-to-target for reach, box-to-place-target for pick-and-place)
- `mean_convergence_time_s` — average sim time for successful episodes
- `mean_inference_time_ms` — average forward-pass latency
- `model_size_mb` — checkpoint size on disk
- `vram_mb` — peak VRAM on CUDA (`torch.cuda.max_memory_allocated`, reset per variant)
- `ram_delta_mb` — growth in process RSS high-water mark (`ru_maxrss`) attributable to this variant's episodes; not an isolated peak, since all variants share one process
- `joint_mean` / `joint_std` / `ee_pos_mean` / `ee_pos_std` — trajectory statistics across all episodes

For `task=reach` only, three additional ONNX Runtime variants (`student_ptq`, `student_dyn`, `student_fp16_onnx`) and two TensorRT variants (`student_trt_fp32`, `student_trt_fp16`) are also measured, since the ONNX export/PTQ pipeline (`config/quant.yaml`) is reach-only. `task=pick_and_place` logs a note and skips straight to writing `teacher`/`student` results.

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
MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collection/collect_data.py --task reach
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
MUJOCO_GL=osmesa SHOW_VIEWER=false uv run python3 scripts/collection/collect_data.py --task reach
```

The Docker image ships with `libegl1` + `libgl1` and sets `MUJOCO_GL=disabled` for tests (physics only). Switch it to `egl` for any container that needs to render frames.

## Porting to a new task

The pipeline above the env/expert layer is fully task-agnostic — dataset recording, ACT training, distillation, evaluation, and measurement need zero code changes. Only two files need to be written, plus a couple of registry/config entries (no more import-swapping — every script dispatches on `--task`).

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

The obs vector must have `qpos` in its first `joint_dim` elements — the eval scripts index `obs[:joint_dim]` to extract joint positions for normalisation. `env.step(action)` must accept exactly `action_dim` values as `ctrl` — if your qpos includes unactuated dims (like pick-and-place's box freejoint), keep them in `joint_dim`/context but exclude them from `action_dim` (see `action_qpos_offset` below).

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

### 3. Register the task

- **`src/utils/tasks.py`** — add `"<your_task>": MyEnvironment` to `ENV_CLASSES`. This is what `eval_act.py`, `eval_distil.py`, and `distillation_measure.py` use to build the right environment for `--task <your_task>`.
- **`scripts/collection/collect_data.py`** — add a `TaskSpec(MyEnvironment, MyExpert, <final_dist_fn>, "<success_verb>")` entry to `TASKS`.
- **`config/collect.yaml`** — add a `collect.tasks.<your_task>` block (same `env`/`expert`/`collection` shape as `reach`/`pick_and_place`), pointing `scene_xml_path` at your MuJoCo XML.
- **`config/train.yaml`** — add a `training.tasks.<your_task>` block (`action_dim`, `joint_dim`, `action_qpos_offset`, `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`).
- **`config/distill.yaml`** — add a `distillation.tasks.<your_task>` block (`teacher`, `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`) once you have a teacher checkpoint to distill.
- **`config/eval.yaml`** — add an `eval.tasks.<your_task>` block (`teacher`, `student`, `measure`).

With those in place, `just collect/train/distill/eval[-distill]/measure <your_task>` all work. Quantization (`export_onnx.py`, `ptq.py`, `config/quant.yaml`) is reach-only and isn't part of this registry — wire it up per-task separately if you need it.

### 4. Swap imports in the eval scripts

`scripts/eval/eval_act.py` / `scripts/eval/eval_distil.py` import `ReachEnvironment` and read `cfg["collect"]["tasks"]["reach"]["env"]` directly — replace those with your new class and config path. Everything else (`train_act.py`, `train_distil.py`, `distillation_measure.py`, all of `src/`) is unchanged.

## Project structure

```
src/
  env/              # MuJoCo reach + pick-and-place environments
  expert/           # IK-based scripted experts (curobo for reach; pick-and-place unimplemented) + abstract base
  renderer/         # off-screen rendering + passive viewer
  dataset/          # HDF5 episode recording + PyTorch dataset/dataloader
  algorithms/       # ACT policy, ImageEmbedding, ChunkingBuffer
  utils/            # logger, config loader, Hugging Face Hub helpers, task→Environment registry
scripts/
  collection/
    collect_data.py           # expert demo collection, --task {reach,pick_and_place}
  training/
    train_act.py             # ACT teacher training loop, --task {reach,pick_and_place}
    train_distil.py          # student distillation loop, --task {reach,pick_and_place}
  eval/
    eval_act.py               # teacher evaluation + video export, --task {reach,pick_and_place}
    eval_distil.py            # student evaluation + video export, --task {reach,pick_and_place}
    distillation_measure.py   # teacher / student / (reach-only) quantized benchmark, --task {reach,pick_and_place}
  quantization/
    export_onnx.py            # export distilled student → fp32 + fp16 ONNX (reach only)
    ptq.py                     # post-training quantization → static + dynamic int8 ONNX (reach only)
    inference.py               # TensorRT engine build + inference
  hub/
    push_data_to_hub.py       # manual dataset push, --task {reach,pick_and_place}
    push_teacher_to_hub.py    # manual teacher checkpoint push, --task {reach,pick_and_place}
    push_student_to_hub.py    # manual student checkpoint push, --task {reach,pick_and_place}
models/
  reach_scene.xml
  pick_and_place_scene.xml
  piper.urdf        # AgileX PiPER kinematics, converted from the robot_descriptions MJCF - curobo IK input
  meshes/            # piper.urdf mesh assets
tests/
config/
  collect.yaml      # per-task (reach, pick_and_place) env, expert, collection
  simulation.yaml   # shared renderer
  train.yaml        # per-task teacher training (dims, checkpoint, hub) + shared architecture/optimizer
  distill.yaml      # per-task student distillation (teacher, checkpoint, hub) + shared architecture/loss weights
  eval.yaml         # per-task evaluation + measurement (teacher, student, measure)
  quant.yaml        # quantization (ptq, reach-only)
justfile
Docker/Dockerfile
.github/workflows/ci.yml
```
