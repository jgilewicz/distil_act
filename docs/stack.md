# Technology Stack — Summary

A complete list of languages, frameworks, and tools in the ACT Distillation project, grouped by layer. The status indicates in which phase the technology is introduced into the project.

## Languages

| Language | Role in Project | Status |
|---|---|---|
| Python | main language — model, training, environment, serving | phase 1, core |
| C++ | custom CUDA kernels (host-side), potential ROS2 bindings | phase 1–2 |
| CUDA C++ | custom GPU kernels (replay buffer sampler and onwards) | phase 1–2 |
| Bash | automation, CI scripts, Docker entrypoints | phase 1, core |
| YAML | configurations (Hydra/OmegaConf), GitHub Actions | phase 1, core |

## Core ML / RL

| Technology | Role |
|---|---|
| PyTorch | training framework, custom model implementation |
| ACT (Action Chunking Transformer) | teacher architecture — CVAE + transformer encoder-decoder |
| LeRobot | for data format reference only, not as a black box |
| NumPy | data operations, preprocessing |

## Simulation and Robotics

| Technology | Role |
|---|---|
| MuJoCo | physics engine, custom environment (MJCF) |
| Gymnasium | environment API wrapper (step, reset, reward) |
| ROS2 | communication with physical hardware (phase 2–3) |

## Hardware Acceleration

| Technology | Role |
|---|---|
| CUDA | custom GPU kernels, learning fundamentals (vector add → matrix multiply → sampler) |
| TensorRT | model optimization and quantization of the student for inference |
| ONNX | model exchange format between PyTorch and TensorRT |
| INT8 quantization | student compression for edge deployment |

## MLOps and Infrastructure

| Technology | Role |
|---|---|
| Docker | containerization of training and serving (Dockerfile.train, Dockerfile.serve) |
| GitHub Actions | CI — lint, testing, building images (CPU only at start) |
| Weights & Biases | experiment tracking, run comparisons |
| uv | Python dependency management, lockfile |
| pytest | unit and smoke e2e testing |
| pre-commit | linting and formatting prior to commit |

## Serving (Phase 3)

| Technology | Role |
|---|---|
| FastAPI | lightweight inference server, REST/WebSocket |
| Triton Inference Server | alternative/upgrade for production model serving |
| gRPC | transport in edge-to-cloud architecture (optional) |
| Prometheus + Grafana | latency/throughput metrics (optional, time permitting) |

## Target Edge Hardware (decision in Phase 2)

| Option | Characteristics |
|---|---|
| SO-101 | robotic arm, Feetech STS3215 servomotors, budget-friendly |
| Raspberry Pi 4/5 | CPU-only/limited GPU inference, latency testing on physical edge |
| NVIDIA Jetson | more powerful alternative, native CUDA on edge |

## Abbreviations per phase

```
Phase 1:  Python, PyTorch, MuJoCo, Gymnasium, Docker, GitHub Actions, uv, CUDA (fundamentals)
Phase 2:  + ROS2 (deepened), TensorRT, ONNX, INT8 quant, CUDA (sampler)
Phase 3:  + FastAPI/Triton, gRPC, Prometheus/Grafana, physical hardware
Phase 4:  consolidation — Hugging Face model hosting, finalizing portfolio/CV
```
