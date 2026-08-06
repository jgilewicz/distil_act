# Configuration

All settings live under `config/`, one YAML file per pipeline stage so you can jump straight to what you're changing instead of scrolling one huge file. `load_config()` (`src/utils/config.py`) reads every `*.yaml` in that directory and merges them into a single dict keyed by top-level section — splitting into files is purely for navigability, not namespacing. It raises on a duplicate top-level key across files, so two files can never both define e.g. `training:`.

| File | Top-level key | Section(s) | Used by |
|---|---|---|---|
| `config/collect.yaml` | `collect` | `collect.tasks.<task>` (each: `env`, `expert`, `collection`) | `collect_data.py`, `push_data_to_hub.py` |
| `config/simulation.yaml` | `renderer` | shared, task-independent | every collection/eval/measure script |
| `config/train.yaml` | `training` | `training.tasks.<task>` (`action_dim`, `joint_dim`, `action_qpos_offset`, `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`) + shared architecture/optimizer keys | `train_act.py`, `push_teacher_to_hub.py` |
| `config/distill.yaml` | `distillation` | `distillation.tasks.<task>` (`teacher`, `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`) + shared student architecture/loss-weight keys | `train_distil.py`, `push_student_to_hub.py` |
| `config/eval.yaml` | `eval` | `eval.tasks.<task>` (`teacher`, `student`, `measure`) | `eval_act.py`, `eval_distil.py`, `distillation_measure.py` |
| `config/quant.yaml` | `ptq` | `ptq.tasks.<task>` (every ONNX/engine artifact path) + shared `calibration_samples`/log files | `export_onnx.py`, `ptq.py` |

Scripts pull from the merged dict regardless of which file a section came from: `cfg["training"]["tasks"]["reach"]`, `cfg["eval"]["tasks"][task]["teacher"]`, `cfg["renderer"]["height"]`.

## The per-task block shape

`collect.yaml`, `train.yaml`, `distill.yaml`, `eval.yaml`, and `quant.yaml` all nest a `tasks.<task_name>` block under their top-level section, and every task's block uses **identical key names** — only the values differ. Keys that don't vary by task (model architecture, optimizer settings, wandb project) sit directly under the top-level section, not under `tasks`. Example from `train.yaml`:

```yaml
training:
  embed_dim: 256        # shared — architecture is the same for every task
  lr: 0.00015           # shared — optimizer is the same for every task
  tasks:
    reach:
      action_dim: 7       # per-task — dataset shape differs
      joint_dim: 8
      action_qpos_offset: 0
      checkpoint_prefix: "act_model"
      hub:
        repo_id: "jgilewicz/distil_act_teacher"
        filename: "act_model_final.pt"
    pick:
      action_dim: 7
      joint_dim: 15
      action_qpos_offset: 7
      checkpoint_prefix: "pick_model"
      hub:
        repo_id: "jgilewicz/distil_act_pick_teacher"
        filename: "pick_model_final.pt"
```

Every task-aware script takes `--task {reach,pick,...}` and does nothing else with that value except index into `cfg["<section>"]["tasks"][args.task]` — the script logic itself never branches on the task name. That's the whole point of the shape: adding a task is a config + registration change, not a script change. `src/utils/tasks.py`'s `ENV_CLASSES` dict maps a task name to its `Environment` subclass and is shared by the collection, eval, and measurement scripts. See [porting.md](porting.md) for the full checklist.

## `hub` blocks

Every config section that touches a checkpoint or dataset carries a `hub` sub-block, but the flag name depends on which side of the transfer it's on:

- **Pull side** (`collect.tasks.<t>.collection.hub`, `distillation.tasks.<t>.teacher`, `eval.tasks.<t>.teacher`/`.student`) uses `auto_pull: bool` — download from `repo_id` before the stage runs if the local file/dir is missing.
- **Push side** (`train.tasks.<t>.hub`, `distillation.tasks.<t>.hub`, `collect.tasks.<t>.collection.hub`) uses `auto_push: bool` — upload once the stage finishes.

Note `collect.tasks.<t>.collection.hub` carries both — a dataset collection run can pull an existing dataset to resume into, or push what it collects.

```yaml
hub:
  repo_id: "jgilewicz/distil_act_teacher"
  filename: "act_model_final.pt"
  auto_push: false   # upload once this stage finishes
```

Leave `auto_push` off (the default for every stage right now) and use `just push-data <task>` / `just push-teacher <task>` / `just push-student <task>` whenever you want to push on demand instead (e.g. re-pushing an existing checkpoint without retraining). Requires `huggingface-cli login` first. See [hugging-face.md](hugging-face.md) for what actually performs the transfer and the repo naming convention.

Logging is likewise automatic and config-driven: every stage writes to the `log_file` path set in its own config section via `Logger` — nothing to wire up per run.

## `quant.yaml` is the odd one out

Its per-task blocks hold *only* artifact paths (every ONNX/engine file the export + PTQ stages produce) — no `hub` block, because quantized artifacts aren't currently pushed to the Hub. `calibration_samples` and the two log file paths are shared across tasks, sitting directly under `ptq:`. Treat it as opt-in per task, not mandatory — if a task doesn't need ONNX export or PTQ, skip it.

## Adding a new task's config

Add a `tasks.<name>` block to each file whose stage the new task needs, copying an existing task's block verbatim and changing only the values: dataset dims (`action_dim`/`joint_dim`/`action_qpos_offset`), every path (`checkpoint_prefix`, `log_file`, artifact paths), and every `repo_id` / `filename`. Reuse of another task's path or repo ID is the most common way to silently corrupt a checkpoint or dataset — grep for the string you're about to write (`rg '<name>_final.pt'`, `rg 'jgilewicz/<repo>'`) before committing to it, since nothing enforces uniqueness at load time.
