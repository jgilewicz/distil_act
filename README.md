# ACT Distillation

Distillation of the ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

**Phase 3** — expert data collected, ACT teacher trained with CVAE + temporal ensembling, distilled into a smaller MobileNetV3-backed student, both evaluated in the MuJoCo reach environment.

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
just push-data           # push the collected dataset to the Hub
just push-teacher        # push the teacher checkpoint to the Hub
just push-student        # push the student checkpoint to the Hub
just test                # run test suite
just lint                # ruff check
just fix                 # ruff check --fix + ruff format
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

## Project structure

```
src/
  env/          # MuJoCo reach environment
  expert/       # IK-based scripted expert (mink)
  renderer/     # off-screen rendering + passive viewer
  dataset/      # HDF5 episode recording + PyTorch dataset/dataloader
  algorithms/   # ACT policy, ImageEmbedding, ChunkingBuffer
  utils/        # logger, config loader, Hugging Face Hub helpers
scripts/
  collect_data.py          # expert demo collection
  train_act.py             # ACT teacher training loop
  train_distil.py          # student distillation loop
  eval_act.py               # teacher evaluation + video export
  eval_distil.py            # student evaluation + video export
  push_data_to_hub.py       # manual dataset push
  push_teacher_to_hub.py    # manual teacher checkpoint push
  push_student_to_hub.py    # manual student checkpoint push
models/
  reach_scene.xml
tests/
config/
  simulation.yaml
  collection.yaml
  train.yaml
  distill.yaml
  eval.yaml
justfile
Docker/Dockerfile
.github/workflows/ci.yml
```
