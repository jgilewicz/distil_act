# Results

## Measurement

```bash
just measure
```

Runs `scripts/distillation_measure.py`: evaluates the fp32 teacher and student plus the two quantized variants (`student_ptq`, `student_dyn`) over `eval.measure.n_episodes` randomly-seeded episodes and writes `artifacts/distillation_metrics.json` with, per model:

- `success_rate` — fraction of episodes where the EE reached the target
- `mean_convergence_time_s` — average sim time for successful episodes
- `mean_inference_time_ms` — average forward-pass latency
- `model_size_mb` — checkpoint size on disk
- `vram_mb` / `ram_mb` — peak memory usage (VRAM on CUDA, RSS otherwise)
- `joint_mean` / `joint_std` / `ee_pos_mean` / `ee_pos_std` — trajectory statistics across all episodes

## Reach task results

Numbers below are for the **reach** task. Evaluated over 50 randomly-seeded episodes on an x86 machine. The fp32 teacher and student run on **CUDA**; the int8 variants run on **CPU** via ONNX Runtime — so latency is not directly comparable across those two groups.

| Metric | Teacher (fp32, GPU) | Student (fp32, GPU) | Student PTQ static (int8, CPU) | Student PTQ dynamic (int8, CPU) |
|---|---|---|---|---|
| Success rate | 82% | 82% | 10% | 12% |
| Mean convergence time † | 0.79 s | 0.77 s | 0.41 s | 0.61 s |
| Inference latency | 16.5 ms | 8.4 ms | 194.3 ms | 885.5 ms |
| Model size | 107.6 MB | 26.3 MB | 6.8 MB | 6.9 MB |
| Peak VRAM | 330.9 MB | 149.7 MB | — | — |
| Peak RAM | 2033.8 MB | 2079.5 MB | 2419.6 MB | 2539.6 MB |

† Averaged only over *successful* episodes, so the int8 columns (≈5–6 successes each) are noisy and not meaningfully comparable.

**4 of 7 possible model variants were measured; the project scoped/estimated 4.** The three unmeasured variants are the accuracy-recovery paths named below — per-channel-weight int8, mixed-precision (attention/LayerNorm kept in fp32), and QAT — which were not benchmarked here.

**Distillation is a clear win.** The student matches the teacher's 82% success rate while being **4.1× smaller**, **2× faster**, and using ~2.2× less VRAM. That's the headline result of the pipeline.

**Quantization, on this setup, is not.** Static and dynamic int8 PTQ shrink the student a further ~3.9× (to ~6.8 MB, **15.8× smaller than the teacher**), but success collapses to 10–12% and CPU inference is an order of magnitude slower than the GPU fp32 models. Two things compound here:

- **Accuracy** — per-tensor int8 PTQ is brutal on Transformers; attention logits and LayerNorm activations have a wide dynamic range that per-tensor int8 can't represent, and both static (calibrated) and dynamic (weight-only) variants degrade the same way. Per-channel weights, keeping attention/LayerNorm in fp32, or QAT would be needed to recover it — eager QAT was attempted but its fake-quant op is unsupported by the current ONNX exporters.
- **Latency** — the int8 QDQ graph runs on CPU with unoptimised ONNX Runtime kernels for this model, so it loses badly to the fp32 models on GPU. The only real gain is on-disk size, which matters for storage/load time on edge devices but not for throughput here.

Bottom line: for this ACT policy, **distillation delivers the compression that keeps the task working**, while naive int8 PTQ trades almost all of the task away for a marginal extra size reduction.
