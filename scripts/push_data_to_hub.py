from utils.config import load_config
from utils.hub import push_dataset
from utils.logger import Logger


def main() -> None:
    cfg = load_config()
    col = cfg["collection"]
    log = Logger("logs/push_data_to_hub.log")
    push_dataset(col["dataset_dir"], col["hub"]["repo_id"], log)


if __name__ == "__main__":
    main()
