# TODO

## Ideas

- [x] Rewrite IK solver using cuRobo (NVIDIA) instead of `mink`/`daqp` — GPU-parallel,
      likely faster for batched/edge inference. Needs a parity check against the
      current `mink` solver before swapping it into `ReachExpert`/`PickExpert`.
- [ ] INT8 quantization with TensorRT — currently PTQ goes through ONNX Runtime
      (`ptq.py`, static QDQ + dynamic weight-only). Explore TensorRT-native INT8
      (calibration cache, `IInt8Calibrator`) as an alternative/addition to the fp16
      ModelOpt AutoCast path, gated on a parity + task-metric check per CLAUDE.md.
- [ ] Full GPU usage report for TensorRT serving using Nsight tooling (Nsight
      Systems / Nsight Compute) — profile the engine end-to-end (not just
      `trtexec` throughput), attribute time across kernels, report p50/p95/p99,
      thermal/power state, and utilization on target hardware (Orin).

## Tools to learn (physical AI / edge)

- cuRobo — GPU-accelerated motion generation/IK (NVIDIA)
- Nsight Systems — system-wide timeline profiling, CPU/GPU correlation
- Nsight Compute — kernel-level GPU profiling
- TensorRT (C++ API + `trtexec`) — engine build/inference, INT8 calibration
- Isaac Lab / Isaac Sim — GPU-parallel simulation, sim-to-real
- `tegrastats` / `jtop` — Jetson power, thermal, utilization monitoring
- `polygraphy` / `onnx-graphsurgeon` — ONNX graph inspection and debugging
- DeepStream — if video pipeline throughput becomes relevant on edge
