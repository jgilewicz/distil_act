# Training

`just train <task>` reads its dataset from `collect.tasks.<task>.collection.dataset_dir` (auto-pulled from the Hub if absent, per `collect.tasks.<task>.collection.hub.auto_pull`) and writes into `training.tasks.<task>.checkpoint_dir`, using that task's `checkpoint_prefix`:

- `<prefix>_step_<N>.pt` — periodic checkpoints (state dict only)
- `<prefix>_final.pt` — final checkpoint including normalisation stats for inference (`act_model_final.pt` for reach, `pick_model_final.pt` for pick)

## Per-task dimensions and normalisation

`action_dim`/`joint_dim`/`action_qpos_offset` are per-task. PiPER has `nq=8` (`joint1`-`joint6` arm + `joint7`/`joint8` gripper) but only `nu=7` actuators — `joint8` mirrors `joint7` via an equality constraint and has no actuator, so it's excluded from the action slice:

- **reach** — recorded qpos is just PiPER's 8 (`joint_dim=8`, `action_dim=7`, `offset=0`).
- **pick** — box's 7-dof freejoint followed by PiPER's 8 (`joint_dim=15`, `action_dim=7`, `offset=7`).

The model conditions on full state but only ever predicts/controls the robot's actuated joints. `EpisodeDataset` (`src/dataset/dataloader.py`) slices the recorded-qpos "actions" target to `[offset:offset+action_dim]` per task; `qpos` (the context input) stays the full `joint_dim`-length vector. Checkpoints save two independent normalisation pairs — `norm_mean`/`norm_std` (qpos, for the input) and `action_norm_mean`/`action_norm_std` (actions, for denormalising the model's output before it's sent to `env.step()`).

Set `WANDB_API_KEY` in a `.env` file or shell environment before running.

```bash
cp .env.example .env   # fill in WANDB_API_KEY
just train reach
just train pick
```
