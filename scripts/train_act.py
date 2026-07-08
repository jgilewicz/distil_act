import os

import torch
import wandb
from dotenv import load_dotenv

from algorithms.act_policy import ACT
from dataset.dataloader import make_dataloader
from utils.config import load_config
from utils.hub import push_checkpoint
from utils.logger import Logger

load_dotenv()


def training_step(
    act: ACT,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    beta: float,
    step: int,
    log_interval: int,
    device: torch.device,
    logger: Logger,
):
    images = batch["images"].float().to(device)
    qpos = batch["qpos"].float().to(device)
    actions = batch["actions"].float().to(device)

    pred, mu, logvar = act(images, qpos, actions)
    loss = torch.nn.functional.mse_loss(pred, actions, reduction="mean")

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = loss + beta * kl_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % log_interval == 0:
        logger.info(
            f"Step: {step}, Loss: {loss.item()}, KL Loss: {kl_loss.item()}, Total Loss: {total_loss.item()}"
        )
        wandb.log(
            {
                "loss/mse": loss.item(),
                "loss/kl": kl_loss.item(),
                "loss/total": total_loss.item(),
                "step": step,
            }
        )


def train():
    cfg = load_config()
    t = cfg["training"]
    logger = Logger(t["log_file"])

    wandb.init(project=t["wandb"]["project"], entity=t["wandb"]["entity"], config=t)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

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
    act = act.to(device)

    beta = t["beta"]
    max_steps = t["max_steps"]
    lr = t["lr"]
    warmup_steps = t["warmup_steps"]
    log_interval = t["log_interval"]
    save_interval = t["save_interval"]
    checkpoint_dir = t["checkpoint_dir"]

    train_loader = make_dataloader(cfg)

    optim = torch.optim.AdamW(act.parameters(), lr=lr)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max_steps - warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    step = 0
    while step < max_steps:
        for batch in train_loader:
            step += 1
            if step > max_steps:
                break

            training_step(act, optim, batch, beta, step, log_interval, device, logger)
            scheduler.step()
            wandb.log({"lr": scheduler.get_last_lr()[0], "step": step})

            if step % save_interval == 0:
                try:
                    torch.save(
                        act.state_dict(), f"{checkpoint_dir}/act_model_step_{step}.pt"
                    )
                    logger.info(f"Saved model at step {step}")
                except RuntimeError as e:
                    logger.warning(f"Checkpoint save failed at step {step}: {e}")

    final_path = f"{checkpoint_dir}/act_model_final.pt"
    try:
        torch.save(
            {
                "model": act.state_dict(),
                "norm_mean": train_loader.dataset.mean,
                "norm_std": train_loader.dataset.std,
            },
            final_path,
        )
    except RuntimeError as e:
        logger.warning(f"Final model save failed: {e}")

    artifact = wandb.Artifact("act_model", type="model")
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)

    wandb.finish()

    logger.info("Training completed and final model saved.")

    if t["hub"]["auto_push"]:
        push_checkpoint(final_path, t["hub"]["repo_id"], t["hub"]["filename"], logger)


if __name__ == "__main__":
    train()
