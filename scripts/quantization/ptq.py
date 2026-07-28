import onnx
from dotenv import load_dotenv
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from dataset.dataloader import EpisodeDataset
from utils.config import load_config
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
    ptq = cfg["ptq"]
    logger = Logger(ptq["log_file"])

    calib_dataset = EpisodeDataset(cfg, split="val")

    quant_pre_process(
        ptq["fp32_path"], ptq["fp32_preprocessed_path"], skip_symbolic_shape=True
    )

    reader = ActCalibrationReader(calib_dataset, ptq["calibration_samples"])

    quantize_static(
        ptq["fp32_preprocessed_path"],
        ptq["output_path"],
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Saved static-quantized ONNX to {ptq['output_path']}")

    fp32_model = onnx.load(ptq["fp32_preprocessed_path"])
    del fp32_model.graph.value_info[:]
    onnx.save(fp32_model, ptq["fp32_preprocessed_path"])

    quantize_dynamic(
        ptq["fp32_preprocessed_path"],
        ptq["dyn_path"],
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Saved dynamic-quantized ONNX to {ptq['dyn_path']}")


if __name__ == "__main__":
    main()
