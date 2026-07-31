from pathlib import Path
from utils.config import load_config

REPO_ROOT = Path(__file__).parent.parent


def test_config_loads():
    cfg = load_config(str(REPO_ROOT / "config"))
    for key in ("collect", "renderer", "training", "distillation", "eval"):
        assert key in cfg
    for task in ("reach", "pick_and_place"):
        collect_cfg = cfg["collect"]["tasks"][task]
        for section in ("env", "expert", "collection"):
            assert section in collect_cfg

        train_cfg = cfg["training"]["tasks"][task]
        for key in ("action_dim", "joint_dim", "action_qpos_offset"):
            assert key in train_cfg
        assert train_cfg["action_dim"] <= train_cfg["joint_dim"]

        assert task in cfg["distillation"]["tasks"]
        assert task in cfg["eval"]["tasks"]
