# ACT Distillation

A research project exploring in-simulation imitation learning and policy distillation. A scripted expert collects demonstrations in MuJoCo, an ACT teacher is trained via behaviour cloning, and a smaller student is distilled from it for edge deployment — all within a single, config-driven pipeline.

The reach task is a proof of concept. The env/expert layer is deliberately thin and abstract (`Environment` + `Expert` base classes), so the rest of the pipeline — data collection, ACT training, distillation, evaluation, measurement, quantization — carries over to any new task without modification. A second task, **pick** (box pickup — lift the box clear of the table, `src/env/pick_env.py` + `src/expert/pick_expert.py`), follows the same pattern, with `PickExpert` driving a `curobo` `MotionPlanner`-based four-phase grasp (`approach → grasp → close → lift`). The whole pipeline is `--task {reach,pick}`-parameterized and working end to end. A natural next step would be packaging this as a Python library for general in-simulation policy learning and distillation.

## Stack

- **Simulation** — MuJoCo 3 + `piper` (AgileX PiPER, 6-DOF arm + gripper) via `robot_descriptions`
- **IK** — NVIDIA `curobo` against `models/piper.urdf`: position-only tool-frame IK (`InverseKinematics`) for reach, full grasp motion planning (`MotionPlanner`/`plan_grasp`) for pick
- **Dataset** — HDF5 (`h5py`), multi-camera frames + joints + timestamps per episode; hosted on the Hugging Face Hub
- **Teacher policy** — ACT in PyTorch; EfficientNet-B3 image backbone, CVAE encoder, Transformer encoder-decoder
- **Student policy** — same ACT architecture, smaller dims + MobileNetV3-Large backbone; trained with a hard action loss, a soft loss against the teacher's predictions, and a latent-space distillation KL
- **Training** — AdamW + linear warmup + cosine decay; KL-weighted ELBO loss; logged to W&B
- **Quantization** — ONNX Runtime PTQ (static QDQ + dynamic int8) and NVIDIA ModelOpt int8 QDQ, plus TensorRT fp32/fp16/int8 engines; task-parameterized (reach + pick)
- **Inference** — `ChunkingBuffer` temporal ensembling over overlapping action chunks

## Pipeline

Separate stages, run one at a time, in order — each takes `reach` or `pick`:

```
just collect reach   →   just train reach   →   just distill reach   →   just eval reach / just eval-distill reach
```

```bash
uv sync   # install dependencies, then run the stages above
```

See [docs/pipeline.md](docs/pipeline.md) for the full command list and [docs/configuration.md](docs/configuration.md) for settings.

## Headline result

On the reach task, **distillation makes the student 4.1× smaller and ~2× faster** than the teacher (using ~2.4× less VRAM), at some accuracy cost (95% → 80% success). Among the post-training quantization variants, fp16 ONNX gives the best size/accuracy balance, per-channel int8 PTQ is usable but lossy, and the TensorRT path still needs its inference loop optimized. Full numbers and analysis in [docs/results.md](docs/results.md).

## Documentation

- [docs/pipeline.md](docs/pipeline.md) — the `--task`-parameterized stages and full `just` command list
- [docs/configuration.md](docs/configuration.md) — `config/` layout, per-task block shape, merge rules
- [docs/hugging-face.md](docs/hugging-face.md) — Hub pull/push mechanics and repo naming
- [docs/training.md](docs/training.md) — training the ACT teacher
- [docs/distillation.md](docs/distillation.md) — distilling the student
- [docs/quantization.md](docs/quantization.md) — ONNX export + PTQ + TensorRT
- [docs/evaluation.md](docs/evaluation.md) — watching the policies run, video export
- [docs/results.md](docs/results.md) — reach-task benchmark
- [docs/curobo.md](docs/curobo.md) — writing/debugging a curobo-based expert
- [docs/porting.md](docs/porting.md) — adding a new task end to end
- [docs/headless-rendering.md](docs/headless-rendering.md) — running without a display
- [docs/architecture.md](docs/architecture.md) — data flow, key files, project structure
