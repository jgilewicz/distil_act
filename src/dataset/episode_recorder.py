import h5py
import numpy as np


class EpisodeRecorder:
    def __init__(self, file_path: str, img_shape: tuple, joint_dim: int) -> None:
        self.path = file_path
        self.file = h5py.File(file_path, "w")

        self.frames = self.file.create_dataset(
            name="frames",
            shape=(0, *img_shape),
            maxshape=(None, *img_shape),
            dtype="uint8",
            chunks=(1, *img_shape),
            compression="gzip",
        )

        self.joints = self.file.create_dataset(
            name="joints",
            shape=(0, joint_dim),
            maxshape=(None, joint_dim),
            dtype="float32",
            chunks=(1, joint_dim),
        )

        self.timestamps = self.file.create_dataset(
            name="timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",
            chunks=(1,),
        )

    def append_step(
        self, frames: np.ndarray, joint_pos: np.ndarray, timestamp: float
    ) -> None:
        current_len = self.frames.shape[0]
        new_len = current_len + 1

        self.frames.resize(new_len, axis=0)
        self.joints.resize(new_len, axis=0)
        self.timestamps.resize(new_len, axis=0)

        self.frames[current_len] = frames
        self.joints[current_len] = joint_pos
        self.timestamps[current_len] = timestamp

    def close(self) -> None:
        self.file.close()
