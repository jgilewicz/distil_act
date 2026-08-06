# The pipeline

The pipeline is four separate stages — run them one at a time, in order. Nothing chains automatically into the next stage; each `just` recipe does exactly one job.

```
just collect   →   just train   →   just distill   →   just eval / just eval-distill
 (gather demos)    (train teacher)   (distill student)   (watch it run, save video)
```

```bash
uv sync                  # install / sync dependencies

just collect-headless    # 1. gather expert demonstrations (no viewer)
just train               # 2. train the ACT teacher on the dataset
just distill             # 3. distill the teacher into a smaller student
just eval                # 4a. watch the teacher policy, save a video
just eval-distill        # 4b. watch the distilled student policy, save a video
```

Each stage reads its own settings from `config/`, downloads whatever inputs it needs from the Hugging Face Hub automatically if they're not already on disk, and writes its own logs. See [configuration.md](configuration.md).

## Commands

```bash
just                     # list all available tasks
just collect             # collect demos with viewer (macOS: uses mjpython)
just collect-headless    # collect headless
just train               # train the ACT teacher (logs to W&B, saves to artifacts/)
just distill             # distill the teacher into a smaller student
just eval                # run the trained teacher with viewer (macOS: uses mjpython)
just eval-distill        # run the distilled student with viewer (macOS: uses mjpython)
just ptq                 # post-training quantization (ONNX): static + dynamic int8 student
just measure             # compare teacher, student, and quantized variants: success rate, latency, size, VRAM/RAM
just push-data           # push the collected dataset to the Hub
just push-teacher        # push the teacher checkpoint to the Hub
just push-student        # push the student checkpoint to the Hub
just test                # run test suite
just lint                # ruff check
just fix                 # ruff check --fix + ruff format
just clean               # remove generated logs, dataset, and pycache
```
