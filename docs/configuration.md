# Configuration

All settings live under `config/`, split into one file per pipeline stage so you can jump straight to what you're changing instead of scrolling one huge file:

| File | Section(s) | Used by |
|---|---|---|
| `config/simulation.yaml` | `env`, `expert`, `renderer` | collection, both eval scripts |
| `config/collection.yaml` | `collection` | `collect_data.py`, `push_data_to_hub.py` |
| `config/train.yaml` | `training` | `train_act.py`, `push_teacher_to_hub.py` |
| `config/distill.yaml` | `distillation` | `train_distil.py`, `push_student_to_hub.py` |
| `config/eval.yaml` | `eval` | `eval_act.py`, `eval_distil.py`, `distillation_measure.py` |
| `config/quant.yaml` | `ptq` | `ptq.py` |

`load_config()` merges every `*.yaml` in `config/` into a single dict, so scripts just do `cfg["training"]`, `cfg["eval"]["teacher"]`, etc. regardless of which file the section lives in.

## Hugging Face Hub automation

Every stage that touches a checkpoint or dataset has a `hub` block in its config:

```yaml
hub:
  repo_id: "jgilewicz/distil_act_teacher"
  filename: "act_model_final.pt"
  auto_push: false   # upload once this stage finishes
```

(pull-side blocks — `collection.hub`, `distillation.teacher`, `eval.teacher`/`eval.student` — use `auto_pull` instead of `auto_push`.)

- **`auto_pull: true`** (default) — if the local dataset/checkpoint is missing, it's downloaded from `repo_id` before the stage runs. Nothing to do manually.
- **`auto_push: true`** — once the stage finishes (dataset collected, training/distillation complete), the result is uploaded to `repo_id` automatically.

Leave `auto_push` off and use `just push-data` / `just push-teacher` / `just push-student` whenever you want to push on demand instead (e.g. re-pushing an existing checkpoint without retraining). Requires `huggingface-cli login` first.

Logging is likewise automatic and config-driven: every stage writes to the `log_file` path set in its own config section via `Logger` — nothing to wire up per run.
