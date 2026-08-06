# Quantization

`just ptq` compresses the distilled student for edge deployment via ONNX Runtime post-training quantization. It starts from `distil_act_model_final.pt`, exports the student's inference path to fp32 ONNX, then writes two int8 models, both benchmarked by `just measure` alongside the fp32 models. Settings live in `config/quant.yaml` (`ptq` section); outputs land in `artifacts/`.

```bash
just ptq     # post-training quantization (CPU, ONNX Runtime)
```

- **static** QDQ (`distil_act_model_ptq.onnx`) — calibrated on the validation split.
- **dynamic** weight-only (`distil_act_model_dyn.onnx`) — no calibration.

The quantized ONNX models carry no normalisation stats — the measurement loads `norm_mean`/`norm_std` from the student checkpoint and runs the ONNX graphs on CPU via ONNX Runtime.
