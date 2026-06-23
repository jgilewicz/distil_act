import os
import json
from dataset.episode_recorder import EpisodeRecorder


class DatasetManager:
    def __init__(
        self, root_dir: str, img_shape: tuple = (480, 640, 3), joint_dim: int = 6
    ) -> None:
        self.root_dir = root_dir
        self.episodes_dir = os.path.join(root_dir, "episodes")
        self.img_shape = img_shape
        self.joint_dim = joint_dim

        os.makedirs(self.episodes_dir, exist_ok=True)
        self.current_episode_idx = self._count_existing_episodes()

    def _count_existing_episodes(self) -> int:
        if not os.path.exists(self.episodes_dir):
            return 0
        files = [
            f
            for f in os.listdir(self.episodes_dir)
            if f.endswith(".h5") and f.startswith("episode_")
        ]
        return len(files)

    def save_metadata(self, metadata: dict) -> None:
        meta_path = os.path.join(self.root_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def create_new_episode(self) -> EpisodeRecorder:
        episode_filename = f"episode_{self.current_episode_idx}.h5"
        episode_path = os.path.join(self.episodes_dir, episode_filename)
        self.current_episode_idx += 1
        return EpisodeRecorder(episode_path, self.img_shape, self.joint_dim)
