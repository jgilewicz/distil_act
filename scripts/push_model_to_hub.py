from huggingface_hub import HfApi

from utils.config import load_config
from utils.logger import Logger

logger = Logger("logs/push_model_to_hub.log")


def main() -> None:
    cfg = load_config("config.yaml")
    t = cfg["training"]
    repo_id = t["hf_repo_id"]
    filename = t["hf_filename"]
    local_path = cfg["eval"]["checkpoint"]

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    logger.info(f"Pushing {local_path} → {repo_id}/{filename}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info("Upload complete")


if __name__ == "__main__":
    main()
