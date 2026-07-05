import itertools
import os

import torch
import wandb
from dotenv import load_dotenv

from algorithms.act_policy import ACT
from dataset.dataloader import make_dataloader
from utils.config import load_config
from utils.hub import ensure_checkpoint
from utils.logger import Logger

load_dotenv()
logger = Logger("logs/training.log")


def distillation_kl(mu_s, logvar_s, mu_t, logvar_t, temperature):
    logvar_t_soft = logvar_t + 2 * torch.log(torch.tensor(temperature))
    var_s = logvar_s.exp()
    var_t_soft = logvar_t_soft.exp()
    return 0.5 * torch.mean(
        logvar_t_soft - logvar_s + (var_s + (mu_s - mu_t).pow(2)) / var_t_soft - 1
    )


def training_step(
    student: ACT,
    teacher: ACT,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    beta: float,
    alpha: float,
    gamma: float,
    temperature: float,
    step: int,
    log_interval: int,
    device: torch.device,
):
    images = batch["images"].float().to(device)
    qpos = batch["qpos"].float().to(device)
    actions = batch["actions"].float().to(device)

    with torch.no_grad():
        teacher_pred, teacher_mu, teacher_logvar = teacher(images, qpos, actions)

    pred, mu, logvar = student(images, qpos, actions)

    hard_loss = torch.nn.functional.mse_loss(pred, actions, reduction="mean")
    soft_loss = torch.nn.functional.mse_loss(pred, teacher_pred, reduction="mean")

    prior_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    distill_kl = distillation_kl(mu, logvar, teacher_mu, teacher_logvar, temperature)

    total_loss = (
        alpha * hard_loss
        + (1 - alpha) * soft_loss
        + beta * prior_kl
        + gamma * distill_kl
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % log_interval == 0:
        logger.info(
            f"Step: {step}, Hard: {hard_loss.item()}, Soft: {soft_loss.item()}, KL: {prior_kl.item()}, Total: {total_loss.item()}"
        )
        wandb.log(
            {
                "loss/hard": hard_loss.item(),
                "loss/soft": soft_loss.item(),
                "loss/kl": prior_kl.item(),
                "loss/distill": distill_kl.item(),
                "loss/total": total_loss.item(),
                "step": step,
            }
        )


def train():
    cfg = load_config("config.yaml")

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        config={
            **cfg["training"],
            "wandb": cfg["wandb"],
        },
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    t = cfg["training"]
    ensure_checkpoint(cfg["eval"]["checkpoint"], t["hf_repo_id"], t["hf_filename"])
    checkpoint = torch.load(
        cfg["eval"]["checkpoint"], map_location=device, weights_only=True
    )

    act = ACT(
        action_dim=t["action_dim"],
        embed_dim=t["embed_dim"],
        latent_dim=t["latent_dim"],
        joint_dim=t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=t["nhead"],
        num_layers=t["num_layers"],
        num_cameras=t["num_cameras"],
    )
    act.load_state_dict(checkpoint["model"])
    act = act.to(device)
    act.eval()

    distil_act = ACT(
        action_dim=t["action_dim"],
        embed_dim=cfg["distillation"]["embed_dim"],
        latent_dim=cfg["distillation"]["latent_dim"],
        joint_dim=t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=cfg["distillation"]["nhead"],
        num_layers=cfg["distillation"]["num_layers"],
        num_cameras=t["num_cameras"],
    )
    distil_act = distil_act.to(device)

    beta = cfg["training"]["beta"]
    alpha = cfg["distillation"]["alpha"]
    gamma = cfg["distillation"]["gamma"]
    temperature = cfg["distillation"]["temperature"]
    max_steps = cfg["training"]["max_steps"]
    lr = cfg["training"]["lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    log_interval = cfg["training"]["log_interval"]
    save_interval = cfg["training"]["save_interval"]

    train_loader = make_dataloader(cfg)

    optim = torch.optim.AdamW(distil_act.parameters(), lr=lr)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max_steps - warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    os.makedirs("artifacts", exist_ok=True)

    for step, batch in enumerate(
        itertools.islice(itertools.cycle(train_loader), max_steps), start=1
    ):
        training_step(
            distil_act,
            act,
            optim,
            batch,
            beta,
            alpha,
            gamma,
            temperature,
            step,
            log_interval,
            device,
        )
        scheduler.step()
        wandb.log({"lr": scheduler.get_last_lr()[0], "step": step})

        if step % save_interval == 0:
            try:
                torch.save(
                    distil_act.state_dict(),
                    f"artifacts/distil_act_model_step_{step}.pt",
                )
                logger.info(f"Saved model at step {step}")
            except RuntimeError as e:
                logger.warning(f"Checkpoint save failed at step {step}: {e}")

    try:
        torch.save(
            {
                "model": distil_act.state_dict(),
                "norm_mean": train_loader.dataset.mean,
                "norm_std": train_loader.dataset.std,
            },
            "artifacts/distil_act_model_final.pt",
        )
    except RuntimeError as e:
        logger.warning(f"Final model save failed: {e}")

    artifact = wandb.Artifact("distil_act_model", type="model")
    artifact.add_file("artifacts/distil_act_model_final.pt")
    wandb.log_artifact(artifact)

    wandb.finish()

    logger.info("Training completed and final model saved.")


if __name__ == "__main__":
    train()
