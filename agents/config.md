# Config system

All settings live under `config/`, one YAML file per pipeline stage.
`load_config()` (`src/utils/config.py`) reads every `*.yaml` in that
directory and merges them into a single dict keyed by top-level section —
splitting into files is purely for navigability, not namespacing. It raises
on a duplicate top-level key across files, so two files can never both
define e.g. `training:`.

| File | Top-level key | Used by |
|---|---|---|
| `collect.yaml` | `collect` | `collect_data.py`, `push_data_to_hub.py` |
| `simulation.yaml` | `renderer` | every collection/eval/measure script (task-independent) |
| `train.yaml` | `training` | `train_act.py`, `push_teacher_to_hub.py` |
| `distill.yaml` | `distillation` | `train_distil.py`, `push_student_to_hub.py` |
| `eval.yaml` | `eval` | `eval_act.py`, `eval_distil.py`, `distillation_measure.py` |
| `quant.yaml` | `ptq` | `export_onnx.py`, `ptq.py` |

Scripts pull from the merged dict regardless of which file a section came
from: `cfg["training"]["tasks"]["reach"]`, `cfg["eval"]["tasks"][task]["teacher"]`,
`cfg["renderer"]["height"]`.

## The per-task block shape

`collect.yaml`, `train.yaml`, `distill.yaml`, `eval.yaml`, and `quant.yaml`
all nest a `tasks.<task_name>` block under their top-level section, and every
task's block uses **identical key names** — only the values differ. Keys
that don't vary by task (model architecture, optimizer settings, wandb
project) sit directly under the top-level section, not under `tasks`. Example
from `train.yaml`:

```yaml
training:
  embed_dim: 256        # shared — architecture is the same for every task
  lr: 0.00015            # shared — optimizer is the same for every task
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

Every task-aware script takes `--task {reach,pick,...}` and does nothing
else with that value except index into `cfg["<section>"]["tasks"][args.task]`
— the script logic itself never branches on the task name. That's the whole
point of the shape: adding a task is a config + registration change, not a
script change. See [`new_task_pipeline.md`](new_task_pipeline.md) for the
full checklist.

## `hub` blocks

Every config section that touches a checkpoint or dataset carries a `hub`
sub-block, but the flag name depends on which side of the transfer it's on:

- **Pull side** (`collect.tasks.<t>.collection.hub`,
  `distillation.tasks.<t>.teacher`, `eval.tasks.<t>.teacher`/`.student`) uses
  `auto_pull: bool` — download from `repo_id` before the stage runs if the
  local file/dir is missing.
- **Push side** (`train.tasks.<t>.hub`, `distillation.tasks.<t>.hub`,
  `collect.tasks.<t>.collection.hub`) uses `auto_push: bool` — upload once
  the stage finishes.

Note `collect.tasks.<t>.collection.hub` carries both — a dataset collection
run can pull an existing dataset to resume into, or push what it collects.
See [`hugging_face.md`](hugging_face.md) for what actually performs the
transfer.

## `quant.yaml` is the odd one out

Its per-task blocks hold *only* artifact paths (every ONNX/engine file the
export + PTQ stages produce) — no `hub` block, because quantized artifacts
aren't currently pushed to the Hub. `calibration_samples` and the two log
file paths are shared across tasks, sitting directly under `ptq:`.

## Adding a new task's config

Add a `tasks.<name>` block to each file whose stage the new task needs,
copying an existing task's block verbatim and changing only the values:
dataset dims (`action_dim`/`joint_dim`/`action_qpos_offset`), every path
(`checkpoint_prefix`, `log_file`, artifact paths), and every `repo_id` /
`filename`. Reuse of another task's path or repo ID is the most common way
to silently corrupt a checkpoint or dataset — grep for the string you're
about to write (`rg '<name>_final.pt'`, `rg 'jgilewicz/<repo>'`) before
committing to it, since nothing enforces uniqueness at load time.

`quant.yaml` is reach + pick today but was reach-only until pick's
quantization support was added — treat it as opt-in per task, not
mandatory. If the new task doesn't need ONNX export or PTQ, skip it.
