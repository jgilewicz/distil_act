# Hugging Face Hub

Two separate mechanisms touch the Hub. Don't confuse them.

1. **Pipeline code** (`src/utils/hub.py`) — what the scripts themselves do
   at runtime to pull inputs and push outputs. Runs as part of `just`
   recipes, driven entirely by `config/*.yaml`.
2. **MCP Hugging Face tools** (`mcp__claude_ai_Hugging_Face__*`) — what *you*,
   the agent, can use in-session to search or inspect the Hub without
   touching the repo's Python. Useful for finding a reference checkpoint,
   checking a repo's file list, or confirming a `repo_id` exists before
   wiring it into config. Not used by the pipeline itself.

## Pipeline code: `src/utils/hub.py`

Four functions, all thin wrappers over `huggingface_hub`:

| Function | Direction | repo_type | Called from |
|---|---|---|---|
| `ensure_checkpoint(local_path, repo_id, filename)` | pull | model | any stage with `auto_pull: true` on a checkpoint |
| `ensure_dataset(dataset_dir, repo_id)` | pull | dataset | any stage with `auto_pull: true` on a dataset |
| `push_checkpoint(local_path, repo_id, filename, logger)` | push | model | any stage with `auto_push: true`, or `push_teacher_to_hub.py`/`push_student_to_hub.py` |
| `push_dataset(dataset_dir, repo_id, logger)` | push | dataset | `collect_data.py` (`auto_push: true`), or `push_data_to_hub.py` |

Both `ensure_*` functions are no-ops if the local path already exists — they
never overwrite a local file/dir with the Hub version. Both `push_*`
functions call `create_repo(..., exist_ok=True)` first, so pushing to a
`repo_id` that doesn't exist yet creates it.

This is the **only** place `huggingface_hub` is imported. If you need Hub
I/O anywhere else, add a function here rather than importing
`huggingface_hub` directly in a script.

## Config wiring (see [`config.md`](config.md) for the block shape)

`auto_pull`/`auto_push` are read by each script, not by `hub.py` — `hub.py`
functions are unconditional, the caller decides whether to call them:

```python
if tcfg["hub"]["auto_pull"] and not Path(checkpoint_path).exists():
    ensure_checkpoint(checkpoint_path, tcfg["hub"]["repo_id"], tcfg["hub"]["filename"])
```

Leaving `auto_push: false` (the default for every stage right now) means
nothing is uploaded automatically — use the dedicated push recipes instead:

```bash
just push-data <task>       # scripts/hub/push_data_to_hub.py
just push-teacher <task>    # scripts/hub/push_teacher_to_hub.py
just push-student <task>    # scripts/hub/push_student_to_hub.py
```

These require `huggingface-cli login` first (the same auth `huggingface_hub`
uses at runtime — check with the MCP `hf_whoami` tool if you're unsure
whether the environment is already authenticated).

## Repo naming convention

One repo per artifact type per task, under the `jgilewicz/` namespace:

| Task | Dataset | Teacher | Student |
|---|---|---|---|
| reach | `jgilewicz/act_reach_env` | `jgilewicz/distil_act_teacher` | `jgilewicz/distil_act_student` |
| pick | `jgilewicz/pick_data` | `jgilewicz/distil_act_pick_teacher` | `jgilewicz/distil_act_pick_student` |

Note reach's repo IDs don't carry `_reach_`/`_reach` suffixes (they predate
the pick task) — a new task should follow pick's convention
(`distil_act_<task>_teacher`/`_student`, `<task>_data`) rather than reach's,
to keep repo names unambiguous as more tasks are added. Quantized ONNX/engine
artifacts (`config/quant.yaml`) are not currently pushed to the Hub — they're
treated as local build outputs, consistent with the root `CLAUDE.md` rule
that engines/ONNX files aren't committed to git either.

## MCP tools: searching/inspecting the Hub

Available in-session (see the MCP server instructions for URI conventions):
`hub_repo_search`, `hub_repo_details`, `hf_whoami`, `hf_fs` (read Hub repo
filesystems), plus job/space/sandbox tools not relevant to this pipeline.
Use these to check a repo exists or inspect its contents *before* writing a
`repo_id` into config — `ensure_checkpoint`/`ensure_dataset` will fail loudly
at pipeline runtime on a typo'd or nonexistent repo, so it's cheaper to
verify up front than to debug a failed `just train` run.
