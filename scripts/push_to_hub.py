from huggingface_hub import HfApi

from utils.config import load_config
from utils.logger import Logger

CONFIG_PATH = "config.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    log = Logger("logs/push_to_hub.log")
    repo_id = cfg["collection"]["hf_repo_id"]
    dataset_dir = cfg["collection"]["dataset_dir"]

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    log.info(f"Pushing {dataset_dir} → {repo_id}")
    api.upload_large_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=8,
    )
    log.info("Upload complete")


if __name__ == "__main__":
    main()
