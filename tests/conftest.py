from pathlib import Path
import pytest
from utils.config import load_config

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config():
    cfg = load_config(str(REPO_ROOT / "config.yaml"))
    cfg["env"]["scene_xml_path"] = str(REPO_ROOT / cfg["env"]["scene_xml_path"])
    return cfg


@pytest.fixture
def env(config):
    from env.env import ReachEnvironment
    return ReachEnvironment(
        scene_xml_path=config["env"]["scene_xml_path"],
        target_x_range=tuple(config["env"]["target_x_range"]),
        target_y_range=tuple(config["env"]["target_y_range"]),
        target_z_range=tuple(config["env"]["target_z_range"]),
        reach_threshold=config["env"]["reach_threshold"],
    )
