import math
from pathlib import Path

import h5py
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset


def _ensure_data(dataset_dir: str, repo_id: str) -> None:
    if not Path(dataset_dir).exists():
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dataset_dir)


class ReachDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train") -> None:
        dataset_dir = cfg["collection"]["dataset_dir"]
        repo_id = cfg["collection"]["hf_repo_id"]
        chunk_size = cfg["training"]["chunk_size"]
        val_ratio = cfg["training"]["val_ratio"]

        _ensure_data(dataset_dir, repo_id)

        episodes = sorted(
            Path(dataset_dir).glob("episodes/episode_*.h5"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        n_train = math.ceil(len(episodes) * (1 - val_ratio))
        episodes = episodes[:n_train] if split == "train" else episodes[n_train:]

        self._chunk_size = chunk_size
        self._index = []
        for ep in episodes:
            with h5py.File(ep, "r") as f:
                n_steps = f["timestamps"].shape[0]

            for t in range(n_steps - chunk_size):
                self._index.append((str(ep), t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        path, t = self._index[idx]
        with h5py.File(path, "r") as f:
            image = f["frames"][t]
            qpos = f["joints"][t]
            actions = f["joints"][t : t + self._chunk_size]

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "qpos": torch.from_numpy(qpos.copy()),
            "actions": torch.from_numpy(actions.copy()),
        }


def make_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    ds = ReachDataset(cfg, split=split)
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
