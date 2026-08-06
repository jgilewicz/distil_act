# Distillation

`just distill` loads the teacher checkpoint (`distillation.teacher.checkpoint`, auto-pulled from `distillation.teacher.repo_id` if missing) and trains a smaller student — same ACT architecture at reduced `embed_dim`/`latent_dim`/`num_layers` with a MobileNetV3-Large backbone instead of EfficientNet-B3. The student's latent is projected up to `teacher_latent_dim` and matched against the teacher's CVAE posterior (`distillation_kl`), alongside a hard action loss and a soft loss against the teacher's predictions.

Writes `distil_act_model_step_<N>.pt` / `distil_act_model_final.pt` into `distillation.checkpoint_dir`, same layout as the teacher.
