# The pipeline

The pipeline is separate stages — run them one at a time, in order. Nothing chains automatically into the next stage; each `just` recipe does exactly one job. Every stage is `--task`-parameterized: pass `reach` or `pick`.

```
just collect reach   →   just train reach   →   just distill reach   →   just eval reach / just eval-distill reach
 (gather demos)          (train teacher)        (distill student)        (watch it run, save video)
```

```bash
uv sync                          # install / sync dependencies

just collect-headless reach      # 1. gather expert demonstrations (no viewer)
just train reach                 # 2. train the ACT teacher on the dataset
just distill reach               # 3. distill the teacher into a smaller student
just eval reach                  # 4a. watch the teacher policy, save a video
just eval-distill reach          # 4b. watch the distilled student policy, save a video
```

`just collect[-headless]`, `just train`, `just distill`, `just eval[-distill]`, `just measure`, `just export-onnx`, and `just ptq` all take `reach` or `pick`. The entire pipeline — collection, teacher training, distillation, evaluation, quantization, and benchmarking — is fully task-agnostic end to end.

Each stage reads its own settings from `config/`, downloads whatever inputs it needs from the Hugging Face Hub automatically if they're not already on disk, and writes its own logs. See [configuration.md](configuration.md).

## Action vs joint dimensions

PiPER has `nq=8` (`joint1`-`joint6` arm + `joint7`/`joint8` gripper) but only `nu=7` actuators — `joint8` mirrors `joint7` via an MJCF equality constraint and has no actuator, so it's excluded from the action slice for both tasks. Pick's recorded qpos additionally mixes the box's free-floating pose (7 dims: pos + quat, unactuated) with PiPER's 8 — the box comes first in the compiled model, so `joint_dim=15` (full state, used as model context) but `action_dim=7` (the robot's actuated joints, sliced out of that same recorded qpos at `action_qpos_offset=7` — see `config/train.yaml`). That's what the model actually predicts and what gets sent to `env.step()` as `ctrl`.

## Commands

```bash
just                              # list all available tasks
just collect <task>               # collect demos with viewer (macOS: uses mjpython); task = reach | pick
just collect-headless <task>      # collect demos headless; task = reach | pick
just train <task>                 # train the ACT teacher (logs to W&B, saves to artifacts/); task = reach | pick
just distill <task>               # distill the teacher into a smaller student; task = reach | pick
just eval <task>                  # run the trained teacher with viewer (macOS: uses mjpython); task = reach | pick
just eval-distill <task>          # run the distilled student with viewer (macOS: uses mjpython); task = reach | pick
just export-onnx <task>           # export the distilled student to ONNX: fp32 + fp16 (modelopt autocast); task = reach | pick
just ptq <task>                   # post-training quantization: ONNX Runtime static/dynamic int8 + modelopt int8 QDQ; task = reach | pick
just measure <task>               # compare teacher, student, quantized, and TensorRT variants: success rate, latency, size, VRAM/RAM
just push-data <task>             # push the collected dataset to the Hub; task = reach | pick
just push-teacher <task>          # push the teacher checkpoint to the Hub; task = reach | pick
just push-student <task>          # push the student checkpoint to the Hub; task = reach | pick
just test                         # run test suite
just lint                         # ruff check
just fix                          # ruff check --fix + ruff format
just clean                        # remove generated logs, dataset, and pycache
```
