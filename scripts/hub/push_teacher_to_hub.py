import argparse

from utils.config import load_config
from utils.hub import push_checkpoint
from utils.logger import Logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("reach", "pick"), required=True)
    args = parser.parse_args()

    cfg = load_config()
    tt = cfg["training"]["tasks"][args.task]
    log = Logger("logs/push_teacher_to_hub.log")
    push_checkpoint(
        f"{tt['checkpoint_dir']}/{tt['checkpoint_prefix']}_final.pt",
        tt["hub"]["repo_id"],
        tt["hub"]["filename"],
        log,
    )


if __name__ == "__main__":
    main()
