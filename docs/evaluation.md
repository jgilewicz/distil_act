# Evaluation

```bash
just eval reach              # teacher, reach
just eval-distill reach      # student, reach
just eval pick               # teacher, pick
```

Each loads its checkpoint (auto-pulled per `eval.tasks.<task>.teacher`/`.student.auto_pull`), builds the task's `Environment` (`src/utils/tasks.py`'s `ENV_CLASSES[task]`), runs the policy, renders the passive viewer, and writes a video to `eval.tasks.<task>.teacher.video_path` / `.student.video_path` (overhead camera).

The policy is queried every `chunk_size // 5` physics steps; `ChunkingBuffer` handles temporal ensembling for intermediate steps. The model's predicted action is denormalised with the task's `action_norm_mean`/`action_norm_std` and applied directly as `env.step()`'s `ctrl` — for pick this is already robot-only (7-dim, matching `nu`; see [training.md](training.md)), so no extra slicing is needed at eval time.
