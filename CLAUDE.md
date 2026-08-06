# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACT Distillation — distilling an ACT (Action Chunking Transformer) visuomotor policy from a simulation-trained teacher to a compressed student for edge deployment.

Phase 2 is complete: expert demonstrations collected via IK, ACT trained with CVAE + temporal ensembling, evaluated in the MuJoCo reach environment. Phase 3 adds distillation of the teacher ACT into a smaller student (MobileNetV3 backbone, distillation KL + hard/soft action losses). Phase 4 adds post-training quantization of the distilled student — ONNX Runtime (static QDQ int8 + dynamic weight-only int8) and NVIDIA ModelOpt (int8 QDQ), plus TensorRT fp32/fp16/int8 engines. A second task, **pick** (box pickup — lift the box above a height threshold, no placement), has been added at the env/expert layer alongside reach, with its `curobo`-based `PickExpert` implemented as a four-phase grasp state machine (`approach → grasp → close → lift`). Collection, training, distillation, evaluation, measurement, and ONNX export/PTQ are all `--task {reach,pick}`-parameterized end to end.

## Critical invariants

- **macOS viewer** — anything that calls `mujoco.viewer.launch_passive` must run under `mjpython`. The justfile handles this — `just collect` and `just eval` use `uv run mjpython`. `just measure` is headless (no viewer), so it runs under plain `python3` and works cross-platform.
- **Task parameterization** — every pipeline stage takes `--task <name>` and reads its config from `cfg["<section>"]["tasks"][task]`; script logic never branches on the task name. Adding a task is a config + registration change (`src/utils/tasks.py`'s `ENV_CLASSES`, `collect_data.py`'s `TASKS`), not a new script. See [docs/porting.md](docs/porting.md).
- **Config** — all configuration lives in `config/*.yaml`. `load_config()` (`src/utils/config.py`) merges every YAML file in `config/` into a single dict keyed by top-level section (`collect`, `renderer`, `training`, `distillation`, `eval`, `ptq`); the split into separate files is for navigability, not namespacing, and it raises on duplicate top-level keys. No hardcoded constants in source files.
- **Hub** — each stage's dataset/checkpoint download and upload is config-driven, not a separate manual step: every stage config carries a `hub` block with `repo_id`/`filename` plus `auto_pull` and `auto_push` (via `src/utils/hub.py`, the only place `huggingface_hub` is imported). The `push-*` recipes remain for pushing on demand.

### Style rules

- No `sys.path` manipulation — packages are installed via `uv sync` (hatchling src layout).
- No multi-line comments or docstrings — single-line `#` only where the WHY is non-obvious.
- No `print` — use `Logger`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`: lint → test → docker build. The Docker image (`Docker/Dockerfile`) is `python:3.14-slim` with `MUJOCO_GL=disabled` for headless physics. `ubuntu-latest` runners have no GPU, so `tests/test_expert.py` (curobo IK/grasp planning, which requires CUDA) is skipped via `pytest.mark.skipif(not torch.cuda.is_available())` — it still runs on any machine with a GPU.

## Where things live

| Topic | Doc |
|---|---|
| Commands / the `--task` pipeline | [docs/pipeline.md](docs/pipeline.md) |
| Config layout + per-task block shape | [docs/configuration.md](docs/configuration.md) |
| Hugging Face Hub pull/push mechanics | [docs/hugging-face.md](docs/hugging-face.md) |
| Teacher training | [docs/training.md](docs/training.md) |
| Student distillation | [docs/distillation.md](docs/distillation.md) |
| Quantization (export_onnx / ptq / TensorRT) | [docs/quantization.md](docs/quantization.md) |
| Evaluation (`eval_act` / `eval_distil`) | [docs/evaluation.md](docs/evaluation.md) |
| Measurement + reach-task results | [docs/results.md](docs/results.md) |
| curobo expert patterns (IK vs planning) | [docs/curobo.md](docs/curobo.md) |
| Adding a new task end to end | [docs/porting.md](docs/porting.md) |
| Headless rendering (`MUJOCO_GL`) | [docs/headless-rendering.md](docs/headless-rendering.md) |
| Data flow, key files, project structure | [docs/architecture.md](docs/architecture.md) |

`docs/architecture.md` is the detailed map of every source file (`src/env`, `src/expert`, `src/renderer`, `src/dataset`, `src/algorithms`, `src/utils`) and every script (`scripts/*`) — read it before changing internals.
