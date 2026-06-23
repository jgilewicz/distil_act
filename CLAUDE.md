# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACT Distillation — distilling an ACT (Action Chunking Transformer) visuomotor policy from a large simulation-trained teacher to a compressed student deployable on edge hardware.

Current focus: **reach task** — a low-cost robot arm in MuJoCo must reach randomly-sampled target positions. Expert demonstrations are collected via IK, saved to HDF5, and will be used to train a student policy.

## Commands

```bash
uv sync              # install / sync dependencies
just                 # list all available tasks
just collect         # collect demos with viewer (macOS: uses mjpython)
just collect-headless  # collect headless
just lint            # ruff check
just fix             # ruff check --fix + ruff format
just test-expert     # IK convergence smoke test
```

On macOS, anything that calls `mujoco.viewer.launch_passive` must run under `mjpython`. The justfile handles this — `just collect` uses `uv run mjpython`, headless uses `uv run python3`.

All configuration lives in `config.yaml` at the project root. No hardcoded constants in source files.

## Architecture

### Data flow

```
ReachEnvironment → ReachExpert → SceneRenderer → EpisodeRecorder → DatasetManager
  (physics)          (IK)         (rendering)      (HDF5 write)     (file layout)
```

### Key files

**`src/env/env.py` — `ReachEnvironment`**
- Merges `models/reach_scene.xml` with the `low_cost_robot_arm` MJCF from `robot_descriptions` at runtime; attaches an `ego_cam` to `robot_gripper_static_finger`.
- All robot bodies/joints/actuators are prefixed `robot_` after merging.
- `step(action)` returns `(obs, terminated)` — terminated when EE distance < `reach_threshold`.
- Observation vector: `[qpos(6), qvel(6), ee_pos(3), target_pos(3)]`. Target is a mocap body.

**`src/expert/reach_expert.py` — `ReachExpert`**
- Uses `mink` (IK library) with `daqp` solver and `mink.FrameTask` on `robot_gripper_moving_finger` (`frame_type="body"` — robot has no sites).
- Each `compute_action(target_pos)` call syncs mink config from live `data.qpos`, runs up to `max_iters` IK iterations (early-exit at `ik_pos_threshold`), returns a ctrl array for the position actuators.
- No `VelocityLimit` — lets IK jump to solution aggressively for fast expert convergence.

**`src/renderer/renderer.py` — `SceneRenderer`**
- Context manager; `render_step(action)` returns `(obs, terminated, frames)` where `frames` is a `dict[camera_name → BGR ndarray]`.
- Available cameras: `overhead_cam` (scene XML) and `ego_cam` (gripper-mounted).

**`src/dataset/`**
- `EpisodeRecorder`: single HDF5 file with resizable datasets `frames` (uint8), `joints` (float32), `timestamps` (float64).
- `DatasetManager`: manages `root_dir/episodes/episode_N.h5` layout; auto-increments index from existing files; writes `metadata.json`.

**`src/utils/`**
- `logger.py` — `Logger(filename)`: logs `[INFO]`/`[WARNING]`/`[ERROR]` to stdout and file simultaneously.
- `config.py` — `load_config(path)`: loads `config.yaml` via PyYAML, returns dict.

**`src/algorithms/`** — empty, reserved for student training.

### Style rules
- No `sys.path` manipulation — packages are installed via `uv sync` (hatchling src layout).
- No multi-line comments or docstrings — single-line `#` only where the WHY is non-obvious.
- No `print` — use `Logger`.
