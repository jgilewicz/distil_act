import json
import numpy as np
import h5py
from dataset.dataset_manager import DatasetManager


def test_episode_recorder_writes_hdf5(tmp_path):
    manager = DatasetManager(root_dir=str(tmp_path), img_shape=(2, 48, 64, 3), joint_dim=6)
    recorder = manager.create_new_episode()
    frames = np.zeros((2, 48, 64, 3), dtype=np.uint8)
    joints = np.zeros(6, dtype=np.float32)
    recorder.append_step(frames, joints, 0.0)
    recorder.append_step(frames, joints, 0.1)
    recorder.close()

    ep_path = tmp_path / "episodes" / "episode_0.h5"
    assert ep_path.exists()
    with h5py.File(ep_path, "r") as f:
        assert f["frames"].shape == (2, 2, 48, 64, 3)
        assert f["joints"].shape == (2, 6)
        assert f["timestamps"].shape == (2,)


def test_dataset_manager_auto_increments(tmp_path):
    m1 = DatasetManager(root_dir=str(tmp_path), img_shape=(2, 4, 4, 3), joint_dim=6)
    r = m1.create_new_episode()
    r.close()

    m2 = DatasetManager(root_dir=str(tmp_path), img_shape=(2, 4, 4, 3), joint_dim=6)
    r2 = m2.create_new_episode()
    r2.close()

    files = list((tmp_path / "episodes").iterdir())
    assert len(files) == 2


def test_save_metadata(tmp_path):
    manager = DatasetManager(root_dir=str(tmp_path))
    manager.save_metadata({"n_episodes": 5, "success_rate": 0.9})
    meta_path = tmp_path / "metadata.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["n_episodes"] == 5
