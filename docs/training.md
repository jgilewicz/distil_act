# Training

Training reads its dataset from `collection.dataset_dir` (auto-pulled from the Hub if absent, per `collection.hub.auto_pull`) and writes into `training.checkpoint_dir`:

- `act_model_step_<N>.pt` — periodic checkpoints (state dict only)
- `act_model_final.pt` — final checkpoint including `norm_mean` / `norm_std` for inference

Set `WANDB_API_KEY` in a `.env` file or shell environment before running.

```bash
cp .env.example .env   # fill in WANDB_API_KEY
just train
```
