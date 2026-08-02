default:
    @just --list

# install / sync all dependencies
sync:
    rm -rf curobo/
    uv sync

# collect expert demonstrations with viewer (macOS requires mjpython); task = reach | pick
collect task:
    uv run scripts/collection/collect_data.py --task {{task}}

# collect expert demonstrations headless (no viewer window); task = reach | pick
collect-headless task:
    MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collection/collect_data.py --task {{task}}

# run test suite
test:
    MUJOCO_GL=disabled uv run pytest tests/ -v

# check code for lint errors
lint:
    uv run ruff check src/ scripts/

# auto-fix lint errors and reformat code
fix:
    uv run ruff check --fix src/ scripts/
    uv run ruff format src/ scripts/

# remove generated logs, dataset, and pycache
clean:
    rm -rf logs/ data/ src/**/__pycache__ scripts/**/__pycache__ artifacts/ .ruff_cache/ .pytest_cache/

# train the ACT policy; task = reach | pick
train task:
    uv run python3 scripts/training/train_act.py --task {{task}}

# distill the ACT policy into a smaller student model; task = reach | pick
distill task:
    uv run python3 scripts/training/train_distil.py --task {{task}}

# evaluate the trained teacher ACT policy with viewer (macOS requires mjpython); task = reach | pick
eval task:
    uv run mjpython scripts/eval/eval_act.py --task {{task}}

# evaluate the distilled student policy with viewer (macOS requires mjpython); task = reach | pick
eval-distill task:
    uv run mjpython scripts/eval/eval_distil.py --task {{task}}

# push collected dataset to Hugging Face Hub (requires huggingface-cli login); task = reach | pick
push-data task:
    uv run python3 scripts/hub/push_data_to_hub.py --task {{task}}

# push trained teacher checkpoint to Hugging Face Hub (requires huggingface-cli login); task = reach | pick
push-teacher task:
    uv run python3 scripts/hub/push_teacher_to_hub.py --task {{task}}

# push distilled student checkpoint to Hugging Face Hub (requires huggingface-cli login); task = reach | pick
push-student task:
    uv run python3 scripts/hub/push_student_to_hub.py --task {{task}}

# compare teacher vs student over 50 episodes each: convergence, success rate, size, VRAM/RAM, inference speed
# headless comparison, so plain python works (no mjpython / viewer needed); task = reach | pick
measure task:
    uv run python3 scripts/eval/distillation_measure.py --task {{task}}

# export the distilled student to ONNX: fp32 (torch.onnx.export) + fp16 (modelopt autocast)
export-onnx:
    uv run python3 scripts/quantization/export_onnx.py

# post-training quantization (ONNX Runtime): static + dynamic int8 of the student, from the exported fp32 ONNX
ptq:
    uv run python3 scripts/quantization/ptq.py
