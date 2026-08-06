# Evaluation

```bash
just eval             # teacher
just eval-distill     # student
```

Each loads its checkpoint (auto-pulled per `eval.teacher`/`eval.student.auto_pull`), runs the policy in the MuJoCo reach environment, renders the passive viewer, and writes a video to `eval.teacher.video_path` / `eval.student.video_path` (overhead camera).

The policy is queried every `chunk_size // 5` physics steps; `ChunkingBuffer` handles temporal ensembling for intermediate steps.
