# ACT Distillation

Distillation of the ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

**Phase 2 complete** — expert data collected, ACT trained with CVAE + temporal ensembling, and evaluated in the MuJoCo reach environment.

## Stack

- **Simulation** — MuJoCo 3 + `low_cost_robot_arm` via `robot_descriptions`
- **IK** — `mink` with `daqp` solver
- **Dataset** — HDF5 (`h5py`), multi-camera frames + joints + timestamps per episode; hosted on [Hugging Face](https://huggingface.co/datasets/jgilewicz/distil_act_reach_env)
- **Policy** — ACT (Action Chunking Transformer) in PyTorch; EfficientNet-B3 image backbone, CVAE encoder, Transformer encoder-decoder
- **Training** — AdamW + linear warmup + cosine decay; KL-weighted ELBO loss; logged to W&B
- **Inference** — `ChunkingBuffer` temporal ensembling over overlapping action chunks

## Quickstart

```bash
uv sync
just collect-headless   # gather expert demos without viewer
just collect            # same with interactive viewer (macOS: mjpython)
just train              # train ACT policy (logs to W&B, saves to artifacts/)
just eval               # run trained policy with viewer (macOS: mjpython)
just test               # run test suite
```

All parameters (episode count, chunk size, model dims, camera names, etc.) are in `config.yaml`.

## Training

Training reads from `data/reach/` (auto-downloaded from HF if absent) and writes:

- `artifacts/act_model_step_<N>.pt` — periodic checkpoints (state dict only)
- `artifacts/act_model_final.pt` — final checkpoint including `norm_mean` / `norm_std` for inference

Set `WANDB_API_KEY` in a `.env` file or shell environment before running.

```bash
cp .env.example .env   # fill in WANDB_API_KEY
just train
```

## Evaluation

```bash
just eval
```

Loads `artifacts/act_model_final.pt`, runs the policy in the MuJoCo reach environment, renders the passive viewer, and writes `artifacts/eval.mp4` (overhead camera).

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
  utils/        # logger, config loader
scripts/
  collect_data.py   # expert demo collection
  train_act.py      # ACT training loop
  eval_act.py       # policy evaluation + video export
  push_to_hub.py    # upload dataset to Hugging Face
models/
  reach_scene.xml
tests/
config.yaml
justfile
Docker/Dockerfile
.github/workflows/ci.yml
```
