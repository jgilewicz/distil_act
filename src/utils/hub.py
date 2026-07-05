from pathlib import Path

from huggingface_hub import hf_hub_download


def ensure_checkpoint(local_path: str, repo_id: str, filename: str) -> None:
    if not Path(local_path).exists():
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(Path(local_path).parent),
        )
