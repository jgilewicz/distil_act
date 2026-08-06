# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACT Distillation — distilling an ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

Phase 2 is complete: expert demonstrations collected via IK, ACT trained with CVAE + temporal ensembling, evaluated in the MuJoCo reach environment. Phase 3 adds distillation of the teacher ACT into a smaller student (MobileNetV3 backbone, distillation KL + hard/soft action losses). Phase 4 adds post-training quantization of the distilled student via ONNX Runtime (static QDQ int8 + dynamic weight-only int8), exported to ONNX and benchmarked against the fp32 teacher/student in `distillation_measure.py`.

## Critical invariants

- **macOS viewer** — anything that calls `mujoco.viewer.launch_passive` must run under `mjpython`. The justfile handles this — `just collect` and `just eval` use `uv run mjpython`. `just measure` is headless (no viewer), so it runs under plain `python3` and works cross-platform.
- **Config** — all configuration lives in `config/*.yaml`. `load_config()` (`src/utils/config.py`) merges every YAML file in `config/` into a single dict keyed by top-level section (`env`, `expert`, `renderer`, `collection`, `training`, `distillation`, `eval`, `ptq`); the split into separate files is for navigability, not namespacing. No hardcoded constants in source files.
- **Hub** — each stage's dataset/checkpoint download and upload is config-driven, not a separate manual step: every stage config carries a `hub` block with `repo_id`/`filename` plus `auto_pull` and `auto_push` (via `src/utils/hub.py`). The `push-*` recipes remain for pushing on demand.

### Style rules

- No `sys.path` manipulation — packages are installed via `uv sync` (hatchling src layout).
- No multi-line comments or docstrings — single-line `#` only where the WHY is non-obvious.
- No `print` — use `Logger`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`: lint → test → docker build. The Docker image (`Docker/Dockerfile`) is `python:3.14-slim` with `MUJOCO_GL=disabled` for headless physics.

## Where things live

| Topic | Doc |
|---|---|
| Commands / the four-stage pipeline | [docs/pipeline.md](docs/pipeline.md) |
| Config files + Hugging Face Hub automation | [docs/configuration.md](docs/configuration.md) |
| Teacher training | [docs/training.md](docs/training.md) |
| Student distillation | [docs/distillation.md](docs/distillation.md) |
| Quantization (`ptq.py`) | [docs/quantization.md](docs/quantization.md) |
| Evaluation (`eval_act.py` / `eval_distil.py`) | [docs/evaluation.md](docs/evaluation.md) |
| Measurement + reach-task results | [docs/results.md](docs/results.md) |
| Porting to a new task | [docs/porting.md](docs/porting.md) |
| Headless rendering (`MUJOCO_GL`) | [docs/headless-rendering.md](docs/headless-rendering.md) |
| Data flow, key files, project structure | [docs/architecture.md](docs/architecture.md) |

`docs/architecture.md` is the detailed map of every source file (`src/env`, `src/expert`, `src/renderer`, `src/dataset`, `src/algorithms`, `src/utils`) and every script (`scripts/*`) — read it before changing internals.
