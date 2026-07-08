from pathlib import Path
from utils.config import load_config

REPO_ROOT = Path(__file__).parent.parent


def test_config_loads():
    cfg = load_config(str(REPO_ROOT / "config"))
    for key in ("env", "expert", "renderer", "collection", "training", "distillation", "eval"):
        assert key in cfg
