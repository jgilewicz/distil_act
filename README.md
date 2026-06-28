# ACT Distillation

Distillation of the ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

## Stack

- **Simulation** — MuJoCo 3 + `low_cost_robot_arm` via `robot_descriptions`
- **IK** — `mink` with `daqp` solver
- **Dataset** — HDF5 (`h5py`), multi-camera frames + joints + timestamps per episode
- **Policy** — ACT (Action Chunking Transformer) in PyTorch; EfficientNet-B3 image backbone, CVAE encoder, Transformer encoder-decoder

## Quickstart

```bash
uv sync
just collect-headless   # gather expert demos without viewer
just collect            # same with interactive viewer (macOS: mjpython)
just train              # run ACT policy forward pass smoke test
just test               # run test suite
```

All parameters (episode count, thresholds, camera names, chunk size, etc.) are in `config.yaml`.

## Project structure

```
src/
  env/          # MuJoCo reach environment
  expert/       # IK-based scripted expert (mink)
  renderer/     # off-screen rendering + passive viewer
  dataset/      # HDF5 episode recording + PyTorch dataset loader
  algorithms/   # ACT policy + ChunkingBuffer
  utils/        # logger, config loader
scripts/
  collect_data.py
  train_act.py
  push_to_hub.py
models/
  reach_scene.xml
tests/
config.yaml
justfile
Docker/Dockerfile
.github/workflows/ci.yml
```
