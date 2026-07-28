from dataset.dataloader import EpisodeDataset
from utils.config import load_config
from utils.tensorrt import TensorRtRuntime, build_engine

if __name__ == "__main__":
    config = load_config()
    engine_bytes = build_engine(
        config["ptq"]["fp32_path"], config["ptq"]["engine_fp32_path"]
    )

    calib_dataset = EpisodeDataset(config, split="val")
    example_inputs = {
        "images": calib_dataset[0]["images"].unsqueeze(0).numpy(),
        "joints": calib_dataset[0]["qpos"].unsqueeze(0).numpy(),
    }

    with TensorRtRuntime(engine_bytes) as runtime:
        output = runtime.infer(example_inputs)

    print(output)
