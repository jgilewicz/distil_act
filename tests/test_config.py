from pathlib import Path
from utils.config import load_config

REPO_ROOT = Path(__file__).parent.parent


def test_config_loads():
    cfg = load_config(str(REPO_ROOT / "config"))
    for key in ("collect", "renderer", "training", "distillation", "eval"):
        assert key in cfg
    for task in ("reach", "pick_and_place"):
        task_cfg = cfg["collect"]["tasks"][task]
        for section in ("env", "expert", "collection"):
            assert section in task_cfg
