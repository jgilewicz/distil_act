from pathlib import Path

import yaml


def load_config(config_dir: str = "config") -> dict:
    cfg: dict = {}
    for path in sorted(Path(config_dir).glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        overlap = cfg.keys() & data.keys()
        if overlap:
            raise ValueError(f"Duplicate config key(s) {overlap} found in {path}")
        cfg.update(data)
    return cfg
