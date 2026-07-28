default:
    @just --list

# install / sync all dependencies
sync:
    uv sync

# collect reach expert demonstrations with viewer (macOS requires mjpython)
collect:
    uv run scripts/collection/collect_reach_data.py

# collect reach expert demonstrations headless (no viewer window)
collect-headless:
    MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collection/collect_reach_data.py

# collect pick-and-place expert demonstrations with viewer (macOS requires mjpython)
collect-pick-and-place:
    uv run scripts/collection/collect_pick_and_place_data.py

# collect pick-and-place expert demonstrations headless (no viewer window)
collect-pick-and-place-headless:
    MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collection/collect_pick_and_place_data.py

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

# train the ACT policy
train:
    uv run python3 scripts/training/train_act.py

# distill the ACT policy into a smaller student model
distill:
    uv run python3 scripts/training/train_distil.py

# evaluate the trained teacher ACT policy with viewer (macOS requires mjpython)
eval:
    uv run mjpython scripts/eval/eval_act.py

# evaluate the distilled student policy with viewer (macOS requires mjpython)
eval-distill:
    uv run mjpython scripts/eval/eval_distil.py

# push collected dataset to Hugging Face Hub (requires huggingface-cli login)
push-data:
    uv run python3 scripts/hub/push_data_to_hub.py

# push trained teacher checkpoint to Hugging Face Hub (requires huggingface-cli login)
push-teacher:
    uv run python3 scripts/hub/push_teacher_to_hub.py

# push distilled student checkpoint to Hugging Face Hub (requires huggingface-cli login)
push-student:
    uv run python3 scripts/hub/push_student_to_hub.py

# compare teacher vs student over 50 episodes each: convergence, success rate, size, VRAM/RAM, inference speed
# headless comparison, so plain python works (no mjpython / viewer needed)
measure:
    uv run python3 scripts/eval/distillation_measure.py

# export the distilled student to ONNX: fp32 (torch.onnx.export) + fp16 (modelopt autocast)
export-onnx:
    uv run python3 scripts/quantization/export_onnx.py

# post-training quantization (ONNX Runtime): static + dynamic int8 of the student, from the exported fp32 ONNX
ptq:
    uv run python3 scripts/quantization/ptq.py
