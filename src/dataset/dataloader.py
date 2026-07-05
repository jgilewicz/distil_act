import math
from pathlib import Path

import h5py
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
import numpy as np


def _ensure_data(dataset_dir: str, repo_id: str) -> None:
    if not Path(dataset_dir).exists():
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dataset_dir)


class EpisodeDataset(Dataset):
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

        all_joints = []

        for ep in episodes:
            with h5py.File(ep, "r") as f:
                n_steps = f["timestamps"].shape[0]
                all_joints.append(f["joints"][:])

            for t in range(n_steps - chunk_size):
                self._index.append((str(ep), t))

        all_joints = np.concatenate(all_joints, axis=0)
        self.mean = torch.from_numpy(np.mean(all_joints, axis=0)).float()
        self.std = torch.from_numpy(np.std(all_joints, axis=0)).float()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        path, t = self._index[idx]
        with h5py.File(path, "r") as f:
            images = f["frames"][t]
            qpos = f["joints"][t]
            actions = f["joints"][t : t + self._chunk_size]

        return {
            "images": torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0,
            "qpos": self._normalise(torch.from_numpy(qpos.copy())),
            "actions": self._normalise(torch.from_numpy(actions.copy())),
        }

    def _normalise(self, qpos: torch.Tensor) -> torch.Tensor:
        return (qpos - self.mean) / self.std


def make_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    ds = EpisodeDataset(cfg, split=split)
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
