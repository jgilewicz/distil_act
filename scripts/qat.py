import torch
from dotenv import load_dotenv

from algorithms.act_policy import ACT
from dataset.dataloader import make_dataloader
from utils.config import load_config
from utils.hub import ensure_checkpoint
from utils.logger import Logger
import torch.nn.functional as F

load_dotenv()


def apply_qat_qconfig(model: ACT, backend: str) -> None:
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = None

    for name, module in model.named_modules():
        if "encoder_cvae" in name:
            continue

        if type(module) is torch.nn.Linear:
            module.qconfig = qconfig
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig


def qat_step(student, teacher, optimizer, batch, alpha, device):
    images = batch["images"].float().to(device)
    qpos = batch["qpos"].float().to(device)
    actions = batch["actions"].float().to(device)

    with torch.no_grad():
        teacher_pred = teacher(images, qpos, actions)[0]

    student_pred = student(images, qpos)

    loss = alpha * F.mse_loss(student_pred, actions) + (1 - alpha) * F.mse_loss(
        student_pred, teacher_pred
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def export_qat_to_onnx(model: ACT, example_inputs, path: str) -> None:
    model.eval()
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    onnx_program.save(path)


def main():
    cfg = load_config()
    qat = cfg["qat"]
    logger = Logger(qat["log_file"])

    # Fake-quant fine-tuning runs on CUDA when available; the export is moved to CPU below.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    t = cfg["training"]
    d = cfg["distillation"]

    teacher_cfg = d["teacher"]
    student_ev = cfg["eval"]["student"]
    if teacher_cfg["auto_pull"]:
        ensure_checkpoint(
            teacher_cfg["checkpoint"], teacher_cfg["repo_id"], teacher_cfg["filename"]
        )
    if student_ev["auto_pull"]:
        ensure_checkpoint(
            student_ev["checkpoint"], student_ev["repo_id"], student_ev["filename"]
        )

    teacher_ckpt = torch.load(
        teacher_cfg["checkpoint"], map_location=device, weights_only=True
    )
    student_ckpt = torch.load(
        student_ev["checkpoint"], map_location=device, weights_only=True
    )

    teacher = ACT(
        action_dim=t["action_dim"],
        embed_dim=t["embed_dim"],
        latent_dim=t["latent_dim"],
        joint_dim=t["joint_dim"],
        action_query_len=t["chunk_size"],
        nhead=t["nhead"],
        num_layers=t["num_layers"],
        num_cameras=t["num_cameras"],
    )
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher = teacher.to(device).eval()

    student = ACT(
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
    student.load_state_dict(student_ckpt["model"])
    student = student.to(device)
    logger.info("Teacher and student loaded")

    apply_qat_qconfig(student, qat["backend"])
    student.train()
    student.image_embedding.eval()
    torch.ao.quantization.prepare_qat(student, inplace=True)

    train_loader = make_dataloader(cfg)
    optimizer = torch.optim.AdamW(student.parameters(), lr=qat["lr"])

    step = 0
    while step < qat["max_steps"]:
        for batch in train_loader:
            step += 1
            if step > qat["max_steps"]:
                break

            loss = qat_step(student, teacher, optimizer, batch, qat["alpha"], device)

            if step % qat["log_interval"] == 0:
                logger.info(f"QAT step loss : {loss:.4f}")

    # Move to CPU for a clean ONNX export (example inputs are CPU tensors).
    student = student.to("cpu")
    example_inputs = (
        train_loader.dataset[0]["images"].unsqueeze(0),
        train_loader.dataset[0]["qpos"].unsqueeze(0),
    )
    export_qat_to_onnx(student, example_inputs, qat["output_path"])
    logger.info(f"Saved QAT model to {qat['output_path']}")


if __name__ == "__main__":
    main()
