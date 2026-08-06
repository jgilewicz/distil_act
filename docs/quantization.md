# Quantization

Compressing the distilled student for edge deployment is two separate steps — export to ONNX, then quantize. Settings for both live in `config/quant.yaml` (`ptq.tasks.<task>`, plus shared `calibration_samples`/log files); outputs land in `artifacts/`.

```bash
just export-onnx reach   # distil_act_model_final.pt → fp32 ONNX (torch.onnx.export) + fp16 ONNX (modelopt autocast)
just ptq reach           # fp32 ONNX → static/dynamic int8 (ONNX Runtime) + int8 QDQ (modelopt)
```

## Export

`export_onnx.py --task <task>` loads that task's student checkpoint and exports its inference path (`actions=None`) to fp32 ONNX via `torch.onnx.export(..., dynamo=True)` (input names `images`/`joints`), then converts it to a mixed-precision fp16 graph via `modelopt.onnx.autocast` (`keep_io_types=True`) — the offline replacement for TensorRT's removed `BuilderFlag.FP16`/`INT8` builder flags (TensorRT 11.x now expects precision baked into the ONNX graph ahead of time, as explicit `Cast` nodes). Loading the checkpoint and building the model happens only here, not in `ptq.py`.

## PTQ

`ptq.py --task <task>` only touches that task's already-exported fp32 ONNX (run `export-onnx <task>` first) and writes three int8 models, all benchmarked by `just measure <task>` alongside the fp32/fp16 models and TensorRT engines built from them. It runs `quant_pre_process(skip_symbolic_shape=True)` first (symbolic shape inference chokes on the transformer's `Loop` node), then:

- **static** QDQ (`..._ptq.onnx`, ONNX Runtime) — `quantize_static` + an `ActCalibrationReader` over the validation split.
- **dynamic** weight-only (`..._dyn.onnx`, ONNX Runtime) — `quantize_dynamic`, no calibration. Before dynamic quant it strips `graph.value_info` (the dynamo export's stale shapes trip `quantize_dynamic`'s strict shape inference).
- **int8 QDQ** (`..._int8_qdq.onnx`, `modelopt.onnx.quantization.quantize`) — calibrated on a `.npz` built from the validation split (`calib_npz_path`, via `build_calibration_npz`). `quantize()`'s `calibration_data` argument takes an in-memory array/dict, not a path, so the npz is built then immediately reloaded with `np.load`.

## Normalisation stats

The quantized ONNX models carry no normalisation stats — the measurement loads `norm_mean`/`norm_std`/`action_norm_mean`/`action_norm_std` from the student checkpoint and runs the ONNX graphs on CPU via ONNX Runtime, or as TensorRT engines (`fp32`/`fp16`/`int8`, built once and cached to `engine_*_path`). See [architecture.md](architecture.md) for the TensorRT build/inference path (`src/utils/tensorrt.py`).
