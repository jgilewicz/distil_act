# Results

## Measurement

```bash
just measure reach
just measure pick
```

Runs `scripts/eval/distillation_measure.py --task <task>`: evaluates the fp32 teacher and student over `eval.tasks.<task>.measure.n_episodes` randomly-seeded episodes, and writes `eval.tasks.<task>.measure.output_path` (`artifacts/distillation_metrics_<task>.json`) with, per model:

- `success_rate` — fraction of episodes that reached the task's termination condition (EE-to-target distance for reach, box lifted above a height threshold for pick)
- `mean_convergence_time_s` — average sim time for successful episodes
- `mean_inference_time_ms` — average forward-pass latency
- `model_size_mb` — checkpoint size on disk
- `vram_mb` — peak VRAM on CUDA (`torch.cuda.max_memory_allocated`, reset per variant)
- `ram_delta_mb` — growth in process RSS high-water mark (`ru_maxrss`) attributable to this variant's episodes; not an isolated peak, since all variants share one process
- `joint_mean` / `joint_std` / `ee_pos_mean` / `ee_pos_std` — trajectory statistics across all episodes

Four additional ONNX Runtime variants (`student_ptq`, `student_dyn`, `student_fp16_onnx`, `student_int8_qdq`) and three TensorRT variants (`student_trt_fp32`, `student_trt_fp16`, `student_trt_int8`) are also measured, from that task's `export_onnx.py --task <task>` / `ptq.py --task <task>` outputs (`config/quant.yaml`'s `ptq.tasks.<task>`).

## Reach task results

**Stale — predates fixes to `distillation_measure.py`'s CUDA timing sync, TensorRT engine reuse, and RSS accounting; re-run `just measure reach` to regenerate.**

Numbers below are for the **reach** task. Evaluated over 20 randomly-seeded episodes on an x86 machine. The fp32 teacher and student run on **CUDA** (torch); `student_ptq`/`student_dyn`/`student_fp16_onnx` run on **CPU** via ONNX Runtime; `student_trt_fp32`/`student_trt_fp16` run on **CUDA** via TensorRT — so latency is only directly comparable within each group.

| Metric | Teacher (fp32, GPU) | Student (fp32, GPU) | Student fp16 (ONNX, CPU) | Student PTQ static (int8, CPU) | Student PTQ dynamic (int8, CPU) | Student TRT (fp32, GPU) | Student TRT (fp16, GPU) |
|---|---|---|---|---|---|---|---|
| Success rate | 95% | 80% | 85% | 55% | 30% | 80% | 85% |
| Mean convergence time † | 0.77 s | 0.95 s | 0.95 s | 0.77 s | 0.64 s | 0.94 s | 0.93 s |
| Inference latency | 23.3 ms | 11.7 ms | 61.3 ms | 92.8 ms | 394.9 ms | 192.5 ms | 170.6 ms |
| Model size | 107.6 MB | 26.3 MB | 10.9 MB | 6.7 MB | 6.4 MB | 23.9 MB | 12.6 MB |
| Peak VRAM | 307.0 MB | 125.8 MB | — | — | — | — | — |
| Peak RAM | 2193.2 MB | 2241.5 MB | 2562.4 MB | 2482.0 MB | 2560.2 MB | 3948.7 MB | 5251.3 MB |

† Averaged only over *successful* episodes, so columns with fewer successes (e.g. the 30% `student_dyn` row) are noisier and less comparable.

**Distillation is a solid compression win, at some accuracy cost.** The student is **4.1× smaller** and **~2× faster** than the teacher on GPU, using ~2.4× less VRAM, but success rate drops from 95% to 80% — the CVAE-distilled policy generalizes slightly worse than the teacher it was trained from.

**int8 PTQ still trades accuracy for size, but per-channel calibration narrows the gap.** Static QDQ int8 (`student_ptq`) now reaches 55% success (up from a per-tensor baseline that collapsed to ~10%) at 6.7 MB — **16× smaller than the teacher, 3.9× smaller than the fp32 student** — while dynamic weight-only quantization (`student_dyn`) is both less accurate (30%) and far slower (394.9 ms), since its unoptimised CPU kernels do more work per call than the statically calibrated QDQ graph. Both remain well below the fp32 student's 80% success rate.

**fp16 is the best accuracy/size trade-off of the quantized variants.** The plain fp16 ONNX graph (`student_fp16_onnx`, CPU) matches or beats the fp32 student's success rate (85% vs 80%) at less than half the size (10.9 MB), though at ~5× the latency of the GPU fp32 student since it runs on CPU.

**TensorRT is not a latency win here.** Both TRT engines match or beat the fp32 student on success rate (80% / 85%) and shrink further on disk, but their measured inference latency (170–193 ms) is far higher than either the torch fp32 student (11.7 ms) or the CPU ONNX Runtime variants — most likely dominated by per-call H2D/D2H copy overhead in the raw `cuda-python` inference path (`src/utils/tensorrt.py`) rather than the engine's actual compute time, since this model's per-step batch is tiny. TRT fp16 (12.6 MB) does halve the engine size and cut latency versus TRT fp32 (23.9 MB), consistent with fp16 compute/memory savings once that fixed overhead is factored in.

Bottom line: distillation remains the headline compression step; among the post-training quantization options, **fp16 ONNX gives the best size/accuracy balance without a GPU**, per-channel int8 PTQ is usable but lossy, dynamic int8 is not recommended, and the current TensorRT integration needs its inference path optimized (batched calls, persistent buffers) before its latency numbers are meaningful.
