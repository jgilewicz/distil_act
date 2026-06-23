# ACT Distillation

Distillation of the ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

## Stack

- **Simulation** — MuJoCo 3 + `low_cost_robot_arm` via `robot_descriptions`
- **IK** — `mink` with `daqp` solver
- **Dataset** — HDF5 (`h5py`), frames + joints + timestamps per episode
- **Policy** — PyTorch (student training, upcoming)

## Quickstart

```bash
uv sync
just collect-headless   # gather expert demos without viewer
just collect            # same with interactive viewer (macOS: mjpython)
```

All parameters (episode count, thresholds, camera names, etc.) are in `config.yaml`.

## Project structure

```
src/
  env/          # MuJoCo reach environment
  expert/       # IK-based scripted expert (mink)
  renderer/     # off-screen rendering + passive viewer
  dataset/      # HDF5 episode recording
  algorithms/   # student training (upcoming)
  utils/        # logger, config loader
scripts/
  collect_data.py
models/
  reach_scene.xml
config.yaml
justfile
```
