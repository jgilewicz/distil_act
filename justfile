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
    SHOW_VIEWER=false uv run python3 scripts/collect_data.py

# run IK convergence smoke test
test-expert:
    uv run python3 -m expert.reach_expert

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

# push collected dataset to Hugging Face Hub (requires huggingface-cli login)
push-to-hub:
    uv run python3 scripts/push_to_hub.py
