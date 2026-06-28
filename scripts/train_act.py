from dataset.dataloader import make_dataloader
from utils.config import load_config
from algorithms.act_policy import ACT
from algorithms.chunking_buffer import ChunkingBuffer
import torch


def train():
    cfg = load_config("config.yaml")
    act = ACT(action_dim=6, embed_dim=256, latent_dim=128, joint_dim=6)
    chunk_buffer = ChunkingBuffer(
        chunk_size=cfg["training"]["chunk_size"], action_size=6
    )

    train_loader = make_dataloader(cfg)
    batch = next(iter(train_loader))

    images = batch["images"].float()
    qpos = batch["qpos"].float()
    actions = batch["actions"].float()

    print(f"Images shape: {images.shape}")
    print(f"QPOS shape: {qpos.shape}")
    print(f"Actions shape: {actions.shape}")

    pred, mu, logvar = act(images, qpos, actions)

    print(f"Pred actions shape: {pred.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    assert pred.shape == actions.shape, f"Expected {actions.shape}, got {pred.shape}"
    assert mu.shape == (images.shape[0], 128), (
        f"Expected ({images.shape[0]}, 128), got {mu.shape}"
    )
    print("ACT OK")

    print("\n--- ChunkingBuffer test ---")
    chunk_buffer.reset()

    single_image = images[0:1]
    single_qpos = qpos[0:1]

    for t in range(5):
        with torch.no_grad():
            pred_single = act(single_image, single_qpos)  # inferencja bez actions
        chunk_buffer.add(pred_single.squeeze(0), t)
        action = chunk_buffer.get_action(t)
        print(
            f"t={t} | buffer size: {len(chunk_buffer.buffer)} | action shape: {action.shape} | action[:3]: {action[:3].numpy().round(3)}"
        )

    print("\nt=2 z pełnym buforem (średnia ważona z 3 predykcji):")
    chunk_buffer.reset()
    for t in range(3):
        with torch.no_grad():
            pred_single = act(single_image, single_qpos)
        chunk_buffer.add(pred_single.squeeze(0), t)

    action_t2 = chunk_buffer.get_action(t=2)
    print(f"action shape: {action_t2.shape}")
    print(f"buffer size po evict: {len(chunk_buffer.buffer)}")
    print("ChunkingBuffer OK")


if __name__ == "__main__":
    train()
