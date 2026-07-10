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
| `config/simulation.yaml` | `env`, `expert`, `renderer` | collection, both eval scripts |
| `config/collection.yaml` | `collection` | `collect_data.py`, `push_data_to_hub.py` |
| `config/train.yaml` | `training` | `train_act.py`, `push_teacher_to_hub.py` |
| `config/distill.yaml` | `distillation` | `train_distil.py`, `push_student_to_hub.py` |
| `config/eval.yaml` | `eval` | `eval_act.py`, `eval_distil.py` |

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
just measure             # compare teacher vs student: success rate, latency, size, VRAM/RAM
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

Runs `scripts/distillation_measure.py`: evaluates both teacher and student over `eval.measure.n_episodes` randomly-seeded episodes and writes `artifacts/distillation_metrics.json` with:

- `success_rate` — fraction of episodes where the EE reached the target
- `mean_convergence_time_s` — average sim time for successful episodes
- `mean_inference_time_ms` — average forward-pass latency
- `model_size_mb` — checkpoint size on disk
- `vram_mb` / `ram_mb` — peak memory usage (VRAM on CUDA, RSS otherwise)
- `joint_mean` / `joint_std` / `ee_pos_mean` / `ee_pos_std` — trajectory statistics across all episodes

## Results

Evaluated over 50 randomly-seeded episodes on CPU (MPS not applicable for VRAM).

| Metric | Teacher | Student | Δ |
|---|---|---|---|
| Success rate | 98% | 86% | −12 pp |
| Mean convergence time | 0.67 s | 0.79 s | +18% |
| Inference latency | 72.0 ms | 21.2 ms | **3.4× faster** |
| Checkpoint size | 107.6 MB | 26.3 MB | **4.1× smaller** |
| Peak RAM | 916.9 MB | 917.2 MB | — |

The student runs **3.4× faster** and is **4.1× smaller** at the cost of a 12 percentage-point drop in success rate and slightly slower convergence on successful episodes. The RAM footprint is identical — both models fit comfortably in CPU memory, so the size gain matters most for edge deployment where storage and load time are the bottleneck.

## Headless rendering (no display)

MuJoCo reads `MUJOCO_GL` **at import time** — it must be set in the shell before the process starts.

`just collect-headless` already sets `MUJOCO_GL=egl`. If you run the script directly:

```bash
MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collect_data.py
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
MUJOCO_GL=osmesa SHOW_VIEWER=false uv run python3 scripts/collect_data.py
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

`scripts/collect_data.py` and `scripts/eval_act.py` / `scripts/eval_distil.py` import `ReachEnvironment` and `ReachExpert` directly — replace those with your new classes. Everything else (`train_act.py`, `train_distil.py`, `distillation_measure.py`, all of `src/`) is unchanged.

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
  distillation_measure.py   # teacher vs student benchmark (success, latency, memory)
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
justfile
Docker/Dockerfile
.github/workflows/ci.yml
```
