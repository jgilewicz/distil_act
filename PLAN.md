# Implementation Plan

## 1. Baseline - enironment and SAC Implementation

- MujoCo push env, MJCF XML, 2 cameras, dense reward implementation
- SAC implementation with autotuned entropy
- W&B + Docker + CI (ruff, pytest, smoke tests)
- Sanity check - SAC with qpos

## 2. Perception - RGB encoder + visuomotor SAC

- Frozen backbone (architecture to choose) - projection head per
camera [MLP, concat -> R576]
- RGB vs qpos ablation
- Position prediction from embedding target as loss : efficiency comparsion
between qpos and RGB version

## 3. Assymetric AC - teacher-student

- Privileged critic with unpriviliged actor
- Comparsion between proprio-only / RGB + symmetric / RGB + assymetric

## 4. CUDA upgrade - replay buffer kernel + profiling

- Replay buffer gather kernel - parallel gather transition with random inidces
- torch.profiler

## 5. Model acceleration - ONNX, TensorRT, FastAPI serving

- ONNX model export (encoder)
- TensorRT INT8 vs FP32
- FastAPI/act endpoint - separate Docker serve
