# ACT Distillation — Embedded Visuomotor Policy

Distillation of the ACT (Action Chunking Transformer) model from a large teacher trained in simulation to a compressed student deployable on edge hardware. A year-long project: from training the teacher in MuJoCo, through CUDA acceleration and quantization, to serving on real hardware.

## Goal

Build a complete pipeline — from simulation to deployment — demonstrating both research competency (RL/IL, visuomotor learning) and engineering competency (CUDA, MLOps, real-time inference). Target profile: ML/Robotics Engineer, Mid/Senior, specializing in Physical AI / Edge AI.

## Status

Phase 1 in progress. See [docs/decisions](docs/decisions/) for the context of key decisions.

## Architecture

```
Teacher (ACT, full, sim)  →  Distillation  →  Student (compressed)  →  Serving (edge)
        MuJoCo                INT8 quant        TensorRT/ONNX        FastAPI/Triton
```

Teacher is trained on full data in simulation (MuJoCo + Gymnasium). Student is distilled from the teacher, quantized to INT8, and eventually run on edge hardware (SO-101 / RPi / Jetson — hardware decision in Phase 2).

## Repository Structure

```
act-distill/
├── src/
│   ├── env/          # MJCF, Gymnasium wrapper, reward definition
│   ├── teacher/       # ACT model, CVAE, training loop
│   ├── student/        # Phase 2 — distillation, quantization
│   ├── cuda/           # custom kernels (e.g., replay buffer sampler)
│   ├── serving/         # Phase 3 — FastAPI / Triton
│   └── common/         # configs, logging, replay buffer
├── tests/             # unit + one smoke e2e test
├── infra/              # Dockerfile.train, Dockerfile.serve
├── scripts/            # train.py, eval.py, export_onnx.py
├── notebooks/          # exploration only, never production logic
├── docs/decisions/      # short ADRs, links to full notes in Notion
├── .github/workflows/   # ci.yml — lint, test, build (CPU)
├── pyproject.toml
└── uv.lock
```

The folders `student/`, `serving/`, and `cuda/` exist from the beginning (with a README describing the plan), but are implemented in their respective phases — to avoid building premature code that will change anyway.

## Technology Stack

| Layer | Technologies |
|---|---|
| Core ML | PyTorch, custom ACT/CVAE implementation (LeRobot only as a data format reference) |
| Simulation & Robotics | MuJoCo, Gymnasium, ROS2 |
| Acceleration | CUDA (custom kernels), TensorRT, INT8 quantization |
| MLOps | Docker, GitHub Actions (CI on CPU), Weights & Biases |
| Serving | FastAPI / Triton Inference Server |
| Package management | uv |

## Roadmap (4 phases, ~1 year)

1. **Phase 1 — Summer.** Teacher ACT in MuJoCo. Fundamentals: PyTorch internals, CUDA basics. Repo + CI from day 1.
2. **Phase 2 — Semester 1 (Oct–Dec).** Distillation to student, INT8 quantization, first CUDA sampler. Hardware decision (SO-101 / RPi / Jetson).
3. **Phase 3 — Semester 1/2 (Jan–Mar).** Serving: TensorRT, FastAPI/Triton, edge-to-cloud split. First transfer to physical hardware, if purchased.
4. **Phase 4 — Semester 2 (Apr–Jun).** End-to-end demo, model hosting on Hugging Face, portfolio/CV finalization.

## Project-Driven Principle

No separate "reading theory" sessions. Theory is introduced on-demand when a block of code requires it. Notes from each blocking question go to Notion, with a link to the commit that triggered it. Short versions (ADRs) go to `docs/decisions/` so that the repo remains understandable without access to private notes.

## Local Development

```bash
# install dependencies
uv sync

# train the teacher
uv run scripts/train.py --config configs/teacher.yaml

# run tests
uv run pytest tests/
```

---

*Project run in parallel with the master's thesis (SC-ERL + DreamerV3, separate research path) and work at xBerry.*
