import onnx
import torch
from dotenv import load_dotenv
import modelopt.onnx.autocast as autocast

from algorithms.act_policy import ACT
from dataset.dataloader import EpisodeDataset
from utils.config import load_config
from utils.hub import ensure_checkpoint
from utils.logger import Logger

load_dotenv()

device = torch.device("cuda")


def main():
    cfg = load_config()
    ev = cfg["eval"]["student"]
    ptq = cfg["ptq"]
    logger = Logger(ptq["export_log_file"])

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
        calib_dataset[0]["images"].unsqueeze(0).to(device),
        calib_dataset[0]["qpos"].unsqueeze(0).to(device),
    )

    onnx_program = torch.onnx.export(act, example_inputs, dynamo=True)
    onnx_program.save(ptq["fp32_path"])
    logger.info(f"Exported fp32 ONNX to {ptq['fp32_path']}")

    converted_model = autocast.convert_to_mixed_precision(
        onnx_path=ptq["fp32_path"],
        low_precision_type="fp16",
        keep_io_types=True,
    )
    onnx.save(converted_model, ptq["fp16_path"])
    logger.info(f"Exported fp16 ONNX to {ptq['fp16_path']}")


if __name__ == "__main__":
    main()
