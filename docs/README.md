# Documentation

Detailed docs for the ACT Distillation pipeline. Start with the [project README](../README.md) for the highlights.

| Doc | Contents |
|---|---|
| [pipeline.md](pipeline.md) | The four-stage pipeline and the full `just` command list |
| [configuration.md](configuration.md) | `config/` layout and Hugging Face Hub automation |
| [training.md](training.md) | Training the ACT teacher |
| [distillation.md](distillation.md) | Distilling the student from the teacher |
| [quantization.md](quantization.md) | Post-training quantization (static + dynamic int8 ONNX) |
| [evaluation.md](evaluation.md) | Watching the policies run and exporting videos |
| [results.md](results.md) | Reach-task benchmark: success, latency, size, memory |
| [porting.md](porting.md) | Adapting the pipeline to a new task |
| [headless-rendering.md](headless-rendering.md) | Running without a display (`MUJOCO_GL` backends) |
| [architecture.md](architecture.md) | Data flow, key files, project structure, style rules |
