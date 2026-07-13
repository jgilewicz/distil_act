import torch
from dotenv import load_dotenv
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from algorithms.act_policy import ACT
from dataset.dataloader import EpisodeDataset
from utils.config import load_config
from utils.hub import ensure_checkpoint
from utils.logger import Logger

load_dotenv()


class ActCalibrationReader(CalibrationDataReader):
    def __init__(self, dataset: EpisodeDataset, n_samples: int):
        self._samples = iter(
            {
                "images": dataset[i]["images"].unsqueeze(0).numpy(),
                "joints": dataset[i]["qpos"].unsqueeze(0).numpy(),
            }
            for i in range(min(n_samples, len(dataset)))
        )

    def get_next(self):
        return next(self._samples, None)


def main():
    cfg = load_config()
    ev = cfg["eval"]["student"]
    ptq = cfg["ptq"]
    logger = Logger(ptq["log_file"])

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    if ev["auto_pull"]:
        ensure_checkpoint(ev["checkpoint"], ev["repo_id"], ev["filename"])

    checkpoint = torch.load(ev["checkpoint"], map_location=device, weights_only=True)

    t = cfg["training"]
    d = cfg["distillation"]
    act = ACT(
        action_dim=t["action_dim"],
        embed_dim=d["embed_dim"],
        latent_dim=d["latent_dim"],
        joint_dim=t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=d["nhead"],
        num_layers=d["num_layers"],
        num_cameras=d["num_cameras"],
        teacher_latent_dim=t["latent_dim"],
        distil_act=True,
    )
    act.load_state_dict(checkpoint["model"])
    act = act.to(device)
    act.eval()
    logger.info("Model loaded")

    calib_dataset = EpisodeDataset(cfg, split="val")
    example_inputs = (
        calib_dataset[0]["images"].unsqueeze(0),
        calib_dataset[0]["qpos"].unsqueeze(0),
    )

    onnx_program = torch.onnx.export(act, example_inputs, dynamo=True)
    onnx_program.save(ptq["fp32_path"])
    logger.info(f"Exported fp32 ONNX to {ptq['fp32_path']}")

    quant_pre_process(ptq["fp32_path"], ptq["fp32_path"], skip_symbolic_shape=True)

    reader = ActCalibrationReader(calib_dataset, ptq["calibration_samples"])

    quantize_static(
        ptq["fp32_path"],
        ptq["output_path"],
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Saved static-quantized ONNX to {ptq['output_path']}")

    quantize_dynamic(
        ptq["fp32_path"],
        ptq["dyn_path"],
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Saved dynamic-quantized ONNX to {ptq['dyn_path']}")


if __name__ == "__main__":
    main()
