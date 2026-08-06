# agents/

Runbooks for agents operating on this repo — how-to, not what-is. For
architecture, key files, and data flow, `CLAUDE.md` at the repo root is
already loaded into every session; don't duplicate it here. These files exist
for the procedural knowledge that CLAUDE.md doesn't spell out step-by-step.

| File | Read this when you need to... |
|---|---|
| [`new_task_pipeline.md`](new_task_pipeline.md) | Add a brand new task (new env + expert) and run it through collect → train → distill → eval → measure → quantize, end to end. |
| [`curobo.md`](curobo.md) | Write or debug a curobo-based `Expert` — IK vs motion planning, collision config, common gotchas. |
| [`config.md`](config.md) | Understand or extend `config/*.yaml` — the per-task block shape, `load_config()` merge rules, what breaks if you get it wrong. |
| [`hugging_face.md`](hugging_face.md) | Push/pull datasets or checkpoints, wire up `auto_pull`/`auto_push`, or search the Hub for reference models/datasets. |

## Ground rules that apply across all of the above

- Every pipeline stage is `--task <name>`-parameterized and reads its config
  from `cfg["<section>"]["tasks"][task]`. Adding a task means adding a config
  block and registering two classes — not writing a new script.
- Nothing chains automatically. `just collect`, `just train`, `just distill`,
  `just eval[-distill]`, `just measure`, `just export-onnx`, `just ptq` are
  independent steps; run them in order by hand.
- Config is the only place constants live. If you're tempted to hardcode a
  path, dimension, or repo ID in a script, it belongs in `config/` instead.
- Verify claims against the current files before acting on them — these docs
  describe patterns, and patterns drift. `git grep` the thing you're about to
  rely on.
