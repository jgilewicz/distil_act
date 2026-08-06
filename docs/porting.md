# Adding a new task, end to end

The pipeline above the env/expert layer is fully task-agnostic — the training loop, ACT architecture, dataset loader, `ChunkingBuffer`, and quantization scripts never branch on task name. This is the checklist `pick` followed on top of `reach`; everything past step 3 is config-only.

## 1. Physics: scene + `Environment`

- Add a scene XML under `models/` (mirrors `models/reach_scene.xml` / `models/pick_scene.xml` — static geometry the robot interacts with; the robot itself is merged in at runtime by `build_model`, not baked into the scene XML).
- Add `src/env/<task>_env.py` with an `Environment` subclass (`src/env/base.py`): implement `reset() -> np.ndarray` and `step(action) -> (obs, terminated)`. Call `build_model(scene_xml_path, gripper_kp, gripper_kv)` — don't reimplement the PiPER merge/`ego_cam` attachment logic.
- The obs vector must have `qpos` in its first `joint_dim` elements — the eval scripts index `obs[:joint_dim]` to extract joint positions for normalisation. Decide the **recorded qpos layout** — this determines `joint_dim`/`action_dim`/`action_qpos_offset` in step 4. If the scene adds a free body before the robot in the compiled model (like pick's box), the robot's actuated slice starts at an offset, not at 0.
- Decide the termination condition and put its threshold in config, not a literal (see `reach_threshold`, `lift_height`).

## 2. Expert

- Add `src/expert/<task>_expert.py` with an `Expert` subclass (`src/expert/base.py`): implement `compute_action(obs) -> np.ndarray` (full `env.model.nu`-length ctrl vector) and, if the expert holds per-episode state (a planned trajectory, a phase index), override `reset()`.
- If this is a curobo-based expert, read [curobo.md](curobo.md) first — it covers the IK-vs-motion-planning choice, the shared robot collision cache, and the joint-ordering gotcha that bites almost every new expert.
- The expert only needs to produce good-enough demonstrations — it is discarded after data collection.

## 3. Register the task

Two registries, both required:

- `src/utils/tasks.py` — add `"<task>": <Task>Environment` to `ENV_CLASSES`. Shared by `eval_act.py`, `eval_distil.py`, `distillation_measure.py`, and `collect_data.py`.
- `scripts/collection/collect_data.py`'s `TASKS` dict — add a `TaskSpec`: `env_cls`, `expert_cls`, a `final_dist(env, obs) -> float` function (used for episode-collection logging, not training), and a `success_verb` string for log messages. Follow `_reach_final_dist`/`_pick_final_dist` as examples — this is the one place per-task Python logic is expected to live.

## 4. Config

Add a `tasks.<task>` block to every file whose stage the task needs — see [configuration.md](configuration.md) for the exact shape and the `hub` sub-block convention. In order of the pipeline:

1. `collect.yaml` — `env` (constructor kwargs for your `Environment`), `expert` (constructor kwargs for your `Expert`), `collection` (`n_episodes`, `n_steps`, `dataset_dir`, `log_file`, `render_cameras`, `hub`).
2. `train.yaml` — `action_dim`, `joint_dim`, `action_qpos_offset` (work these out from your qpos layout — see step 1), `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`.
3. `distill.yaml` — `teacher` (points at the checkpoint from step 2), `log_file`, `checkpoint_dir`, `checkpoint_prefix`, `hub`.
4. `eval.yaml` — `teacher`, `student`, `measure` (`n_episodes`, `log_file`, `output_path`).
5. `quant.yaml` — **only if** the task needs ONNX export / PTQ. Every artifact path (`fp32_path` through `engine_int8_path`), no `hub` block.

Pick a `checkpoint_prefix` and Hub `repo_id`s that don't collide with any existing task — see [hugging-face.md](hugging-face.md) for the naming convention to follow.

## 5. Tests

Add a test module mirroring `tests/test_expert.py` for the new expert. If it's curobo-based, gate it the same way (`pytest.mark.skipif(not torch.cuda.is_available(), ...)`). Assert action shape unconditionally and, for anything IK/planning-based, run a convergence test that fails loudly if the expert stops solving.

## 6. Run it

Nothing chains automatically — run each stage by hand, in order, and check its log file before moving to the next:

```bash
just collect-headless <task>   # gather expert demonstrations (no viewer)
just train <task>              # train the ACT teacher
just distill <task>            # distill the teacher into a smaller student
just eval <task>               # watch the teacher, save a video
just eval-distill <task>       # watch the student, save a video
just measure <task>            # success rate / latency / size comparison
just export-onnx <task>        # only if quant.yaml has this task
just ptq <task>                # only if quant.yaml has this task
```

`auto_pull: true` on any stage means you can skip straight to a later stage if the upstream artifact already exists on the Hub under the `repo_id` you configured — it'll be downloaded instead of requiring a local run first.

## 7. Push checkpoints/datasets

If `auto_push` is off (the default), push on demand once you're happy with a result:

```bash
just push-data <task>
just push-teacher <task>
just push-student <task>
```

## What you should *not* need to touch

`src/dataset/`, `src/algorithms/`, `train_act.py`/`train_distil.py` internals, `eval_act.py`/`eval_distil.py` internals, `distillation_measure.py` internals, `export_onnx.py`, `ptq.py`. All of these read `action_dim`/`joint_dim`/`action_qpos_offset` and task-specific paths out of config and are otherwise identical across tasks. If you find yourself editing one of these to special-case a task name, stop — the fix almost certainly belongs in config or in the `Environment`/`Expert` subclass instead.
