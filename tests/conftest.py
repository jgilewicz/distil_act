from pathlib import Path
import pytest
from utils.config import load_config

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config():
    cfg = load_config(str(REPO_ROOT / "config"))
    reach_env = cfg["collect"]["tasks"]["reach"]["env"]
    reach_env["scene_xml_path"] = str(REPO_ROOT / reach_env["scene_xml_path"])
    return cfg


@pytest.fixture
def env(config):
    from env.reach_env import ReachEnvironment

    reach_env = config["collect"]["tasks"]["reach"]["env"]
    return ReachEnvironment(
        scene_xml_path=reach_env["scene_xml_path"],
        target_x_range=tuple(reach_env["target_x_range"]),
        target_y_range=tuple(reach_env["target_y_range"]),
        target_z_range=tuple(reach_env["target_z_range"]),
        reach_threshold=reach_env["reach_threshold"],
    )
