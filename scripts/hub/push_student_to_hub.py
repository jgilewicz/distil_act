import argparse

from utils.config import load_config
from utils.hub import push_checkpoint
from utils.logger import Logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("reach", "pick"), required=True)
    args = parser.parse_args()

    cfg = load_config()
    dt = cfg["distillation"]["tasks"][args.task]
    log = Logger("logs/push_student_to_hub.log")
    push_checkpoint(
        f"{dt['checkpoint_dir']}/{dt['checkpoint_prefix']}_final.pt",
        dt["hub"]["repo_id"],
        dt["hub"]["filename"],
        log,
    )


if __name__ == "__main__":
    main()
