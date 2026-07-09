from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from utils.logger import Logger


def ensure_checkpoint(local_path: str, repo_id: str, filename: str) -> None:
    if not Path(local_path).exists():
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(Path(local_path).parent),
        )


def ensure_dataset(dataset_dir: str, repo_id: str) -> None:
    if not Path(dataset_dir).exists():
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dataset_dir)


def push_checkpoint(
    local_path: str, repo_id: str, filename: str, logger: Logger
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    logger.info(f"Pushing {local_path} -> {repo_id}/{filename}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info("Upload complete")


def push_dataset(dataset_dir: str, repo_id: str, logger: Logger) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    logger.info(f"Pushing {dataset_dir} -> {repo_id}")
    api.upload_large_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=8,
    )
    logger.info("Upload complete")
