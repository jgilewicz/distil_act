# Documentation

Full docs for the ACT Distillation pipeline. Start with the [project README](../README.md) for the highlights.

| Doc | Contents |
|---|---|
| [pipeline.md](pipeline.md) | The `--task`-parameterized stages and the full `just` command list |
| [configuration.md](configuration.md) | `config/` layout, the per-task block shape, `load_config()` merge rules |
| [hugging-face.md](hugging-face.md) | Hub pull/push mechanics (`src/utils/hub.py`), repo naming, MCP tools |
| [training.md](training.md) | Training the ACT teacher; per-task dims and normalisation |
| [distillation.md](distillation.md) | Distilling the student from the teacher |
| [quantization.md](quantization.md) | ONNX export + PTQ (ONNX Runtime static/dynamic, ModelOpt int8, TensorRT) |
| [evaluation.md](evaluation.md) | Watching the policies run and exporting videos |
| [results.md](results.md) | Measurement + reach-task benchmark (teacher / student / quantized / TensorRT) |
| [curobo.md](curobo.md) | Writing or debugging a curobo-based `Expert` — IK vs motion planning, gotchas |
| [porting.md](porting.md) | Adding a new task end to end (env + expert + config), the pick checklist |
| [headless-rendering.md](headless-rendering.md) | Running without a display (`MUJOCO_GL` backends) |
| [architecture.md](architecture.md) | Data flow, key files, project structure, style rules |
