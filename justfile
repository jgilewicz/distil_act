default:
    @just --list

# install / sync all dependencies
sync:
    uv sync

# collect expert demonstrations with viewer (macOS requires mjpython)
collect:
    uv run mjpython scripts/collect_data.py

# collect expert demonstrations headless (no viewer window)
collect-headless:
    MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collect_data.py

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
    rm -rf logs/ data/ src/**/__pycache__ scripts/__pycache__

# train the ACT policy
train:
    uv run python3 scripts/train_act.py

# distill the ACT policy into a smaller student model
distill:
    uv run python3 scripts/train_distil.py

# evaluate the trained ACT policy with viewer (macOS requires mjpython)
eval:
    uv run mjpython scripts/eval_act.py

# push collected dataset to Hugging Face Hub (requires huggingface-cli login)
push-data:
    uv run python3 scripts/push_data_to_hub.py

# push trained teacher checkpoint to Hugging Face Hub (requires huggingface-cli login)
push-model:
    uv run python3 scripts/push_model_to_hub.py
