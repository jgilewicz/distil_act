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
- **Quantization** — ONNX Runtime post-training quantization (static QDQ + dynamic int8), exported to ONNX
- **Inference** — `ChunkingBuffer` temporal ensembling over overlapping action chunks

## Pipeline

Four separate stages, run one at a time, in order:

```
just collect   →   just train   →   just distill   →   just eval / just eval-distill
```

```bash
uv sync   # install dependencies, then run the stages above
```

See [docs/pipeline.md](docs/pipeline.md) for the full command list and [docs/configuration.md](docs/configuration.md) for settings.

## Headline result

On the reach task, **distillation matches the teacher's 82% success rate while being 4.1× smaller and 2× faster** (and using ~2.2× less VRAM) — the headline result of the pipeline. Naive int8 PTQ shrinks the student a further ~3.9× but collapses success to 10–12%. Full numbers and analysis in [docs/results.md](docs/results.md).

## Documentation

- [docs/pipeline.md](docs/pipeline.md) — the four-stage pipeline and full `just` command list
- [docs/configuration.md](docs/configuration.md) — `config/` layout and Hugging Face Hub automation
- [docs/training.md](docs/training.md) — training the ACT teacher
- [docs/distillation.md](docs/distillation.md) — distilling the student
- [docs/quantization.md](docs/quantization.md) — post-training quantization
- [docs/evaluation.md](docs/evaluation.md) — watching the policies run, video export
- [docs/results.md](docs/results.md) — reach-task benchmark
- [docs/porting.md](docs/porting.md) — adapting the pipeline to a new task
- [docs/headless-rendering.md](docs/headless-rendering.md) — running without a display
- [docs/architecture.md](docs/architecture.md) — data flow, key files, project structure
