# Distillation

`just distill <task>` loads that task's teacher checkpoint (`distillation.tasks.<task>.teacher.checkpoint`, auto-pulled from `.teacher.repo_id` if missing) and trains a smaller student — same ACT architecture at reduced `embed_dim`/`latent_dim`/`num_layers` with a MobileNetV3-Large backbone instead of EfficientNet-B3, and the same task-specific `action_dim`/`joint_dim` as the teacher (from `training.tasks.<task>`). The student's latent is projected up to `teacher_latent_dim` and matched against the teacher's CVAE posterior (`distillation_kl`), alongside a hard action loss and a soft loss against the teacher's predictions.

Loss = `alpha * hard_loss + (1-alpha) * soft_loss + beta * prior_kl + gamma * distill_kl`; `distill_kl` matches the student's latent (projected to `teacher_latent_dim`) against the teacher's CVAE posterior at `temperature`.

Writes `<checkpoint_prefix>_step_<N>.pt` / `<checkpoint_prefix>_final.pt` into `distillation.tasks.<task>.checkpoint_dir` (`distil_act_model_*` for reach, `pick_distil_model_*` for pick), same layout and normalisation-stats contract as the teacher.
