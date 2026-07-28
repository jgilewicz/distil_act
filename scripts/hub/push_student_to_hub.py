from utils.config import load_config
from utils.hub import push_checkpoint
from utils.logger import Logger


def main() -> None:
    cfg = load_config()
    hub = cfg["distillation"]["hub"]
    log = Logger("logs/push_student_to_hub.log")
    push_checkpoint(
        cfg["distillation"]["checkpoint_dir"] + "/distil_act_model_final.pt",
        hub["repo_id"],
        hub["filename"],
        log,
    )


if __name__ == "__main__":
    main()
