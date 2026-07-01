import itertools
import os

import torch
import wandb
from dotenv import load_dotenv

from algorithms.act_policy import ACT
from dataset.dataloader import make_dataloader
from utils.config import load_config
from utils.logger import Logger

load_dotenv()
logger = Logger("logs/training.log")


def training_step(
    act: ACT,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    beta: float,
    device: torch.device,
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

    logger.info(
        f"Loss: {loss.item()}, KL Loss: {kl_loss.item()}, Total Loss: {total_loss.item()}"
    )

    wandb.log(
        {
            "loss/mse": loss.item(),
            "loss/kl": kl_loss.item(),
            "loss/total": total_loss.item(),
        }
    )

    return total_loss


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

    beta = cfg["training"]["beta"]
    max_steps = cfg["training"]["max_steps"]
    lr = cfg["training"]["lr"]

    train_loader = make_dataloader(cfg)

    optim = torch.optim.AdamW(act.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps)

    os.makedirs("artifacts", exist_ok=True)

    for step, batch in enumerate(
        itertools.islice(itertools.cycle(train_loader), max_steps), start=1
    ):
        training_step(act, optim, batch, beta, device)
        scheduler.step()
        wandb.log({"lr": scheduler.get_last_lr()[0], "step": step})

        if step % 20 == 0:
            torch.save(act.state_dict(), f"artifacts/act_model_step_{step}.pt")
            logger.info(f"Saved model at step {step}")

    torch.save(
        {
            "model": act.state_dict(),
            "norm_mean": train_loader.dataset.mean,
            "norm_std": train_loader.dataset.std,
        },
        "artifacts/act_model_final.pt",
    )

    artifact = wandb.Artifact("act_model", type="model")
    artifact.add_file("artifacts/act_model_final.pt")
    wandb.log_artifact(artifact)

    wandb.finish()

    logger.info("Training completed and final model saved.")


if __name__ == "__main__":
    train()
