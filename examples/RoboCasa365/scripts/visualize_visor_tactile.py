#!/usr/bin/env python3
"""Offline VISOR tactile (gripper force + contact) prediction validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import gr00t.model  # noqa: F401
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from transformers import AutoModel, AutoProcessor

DEFAULT_DATASET = (
    "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/"
    "robocasa365-datasets/pretrain/atomic/PickPlaceToasterToCounter/20250819/lerobot"
)

# Fixed probe steps (same as prior visor_visualize_pickplace30k runs).
DEFAULT_PROBES = [
    (0, 55, "no_contact"),
    (0, 89, "pre_contact"),
    (0, 109, "contact_peak"),
    (1, 36, "no_contact"),
    (1, 51, "pre_contact"),
    (1, 71, "contact_peak"),
]


def load_modality_config(path: Path) -> None:
    sys.path.insert(0, str(path.parent))
    importlib.import_module(path.stem)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(tactile_pred: torch.Tensor, tactile_gt: torch.Tensor) -> dict:
    """Metrics on horizon steps with valid GT (first 40 steps)."""
    pred = tactile_pred.detach().float().cpu()
    gt = tactile_gt.detach().float().cpu()
    force_pred = pred[..., :2].clamp_min(0.0)
    force_gt = gt[..., :2].clamp_min(0.0)
    contact_prob = torch.sigmoid(pred[..., 2])
    contact_gt = (gt[..., 2] > 0.5).float()
    contact_pred = (contact_prob > 0.5).float()

    contact_acc = float((contact_pred == contact_gt).float().mean())
    tp = float(((contact_pred > 0.5) & (contact_gt > 0.5)).sum())
    fn = float(((contact_pred <= 0.5) & (contact_gt > 0.5)).sum())
    contact_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    force_mask = contact_gt > 0.5
    if force_mask.any():
        force_mae = float((force_pred - force_gt).abs()[force_mask.unsqueeze(-1).expand_as(force_pred)].mean())
        fp = force_pred[force_mask].numpy()
        fg = force_gt[force_mask].numpy()
        force_corr = _safe_corr(fp.reshape(-1), fg.reshape(-1))
    else:
        force_mae = float((force_pred - force_gt).abs().mean())
        force_corr = float("nan")

    return {
        "contact_acc": contact_acc,
        "contact_recall": contact_recall,
        "force_mae": force_mae,
        "force_corr": force_corr,
        "contact_gt_rate": float(contact_gt.mean()),
        "contact_pred_rate": float(contact_pred.mean()),
    }


def plot_sample(
    *,
    out_path: Path,
    ep: int,
    step: int,
    tag: str,
    metrics: dict,
    gt: np.ndarray,
    pred_clean: np.ndarray,
    pred_flow: dict[float, np.ndarray],
    eye_image: np.ndarray | None,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])
    title = (
        f"ep={ep} step={step} tag={tag} | contact_gt_rate={metrics['contact_gt_rate']:.2f} "
        f"acc={metrics['contact_acc']:.2f} force_corr={metrics['force_corr']:.2f}"
    )
    fig.suptitle(title, fontsize=12)

    if eye_image is not None:
        ax_img = fig.add_subplot(gs[0, :])
        ax_img.imshow(eye_image)
        ax_img.axis("off")
        ax_img.set_title("robot0_eye_in_hand (current frame)")

    steps = np.arange(gt.shape[0])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_l.plot(steps, gt[:, 0], label="GT left", linewidth=2)
    ax_l.plot(steps, pred_clean[:, 0], "--", label="pred left (clean WWM)", linewidth=2)
    ax_l.set_title("Left force (N)")
    ax_l.legend()
    ax_l.grid(True, alpha=0.3)

    ax_r = fig.add_subplot(gs[1, 1])
    ax_r.plot(steps, gt[:, 1], label="GT right", linewidth=2)
    ax_r.plot(steps, pred_clean[:, 1], "--", label="pred right (clean WWM)", linewidth=2)
    ax_r.set_title("Right force (N)")
    ax_r.legend()
    ax_r.grid(True, alpha=0.3)

    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.plot(steps, gt[:, 2], label="GT contact", linewidth=2)
    ax_c.plot(steps, 1 / (1 + np.exp(-pred_clean[:, 2])), "--", label="pred contact (clean WWM)", linewidth=2)
    ax_c.set_title("Contact probability")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.plot(steps, gt[:, 2], label="GT contact", linewidth=2)
    for t_val, arr in pred_flow.items():
        prob = 1 / (1 + np.exp(-arr[:, 2]))
        ax_f.plot(steps, prob, "--", label=f"WWM contact @ t={t_val:.1f}")
    ax_f.set_title("WWM contact vs flow time")
    ax_f.set_ylim(-0.05, 1.05)
    ax_f.legend(fontsize=8)
    ax_f.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def predict_tactile(
    model,
    processor,
    batch_inputs: dict,
    flow_time: float,
    *,
    use_clean_action: bool,
) -> torch.Tensor:
    device = model.device
    dtype = model.dtype
    inputs = {k: v for k, v in batch_inputs.items()}
    if "vlm_content" in inputs:
        prep = processor.collator([{"vlm_content": inputs.pop("vlm_content")}])["inputs"]
        inputs.update(prep)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device=device, dtype=dtype if v.is_floating_point() else None)
        elif isinstance(v, dict):
            inputs[k] = {
                sk: sv.to(device=device, dtype=dtype if sv.is_floating_point() else None)
                for sk, sv in v.items()
            }

    backbone_inputs, action_inputs = model.prepare_input(inputs)
    backbone_outputs = model.backbone(backbone_inputs)
    backbone_outputs = model.action_head.process_backbone_output(backbone_outputs)

    vl_embeds = backbone_outputs.backbone_features
    vision_context = model.action_head.visor.pool_vision_context(
        vl_embeds, backbone_outputs.image_mask
    )
    proprio = action_inputs.state.reshape(action_inputs.state.shape[0], -1)
    actions = action_inputs.action
    t = torch.full((actions.shape[0], 1, 1), flow_time, device=device, dtype=actions.dtype)
    tactile_pred = model.action_head.visor.wwm(
        actions,
        t,
        vision_context,
        proprio,
        use_clean_action=use_clean_action,
    )
    return tactile_pred[0]


def build_batch(processor, loader, ep: int, step: int, embodiment_tag: EmbodimentTag):
    traj = loader[ep]
    modality_configs = deepcopy(loader.modality_configs)
    data = extract_step_data(traj, step, modality_configs, embodiment_tag)
    messages = [{"type": MessageType.EPISODE_STEP, "content": data}]
    sample = processor(messages)
    collated = processor.collator([sample])["inputs"]
    return collated, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument(
        "--modality-config-path",
        default=str(PROJECT_ROOT / "examples" / "RoboCasa365" / "robocasa365_config_4frame.py"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embodiment-tag", default="ROBOCASA_PANDA_OMRON")
    parser.add_argument("--video-backend", default="opencv")
    args = parser.parse_args()

    load_modality_config(Path(args.modality_config_path))
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    print(f"[i] VISOR visualize checkpoint={ckpt}")
    print(f"[i] output={out_dir} device={args.device}")

    model = AutoModel.from_pretrained(ckpt)
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)

    processor_dir = ckpt / "processor" if (ckpt / "processor").is_dir() else ckpt
    processor = AutoProcessor.from_pretrained(processor_dir)
    processor.eval()

    modality = processor.get_modality_configs()[embodiment_tag.value]
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
    )

    gate = float(model.action_head.visor.gate.detach().cpu())
    use_visor = bool(getattr(model.config, "use_visor", False))
    print(f"[i] Dataset: {args.dataset_path}")
    print(f"[i] use_visor={use_visor}")
    print(f"[i] gate (scalar)={gate:.6f}")

    results = []
    for ep, step, tag in DEFAULT_PROBES:
        if ep >= len(loader):
            print(f"[w] skip ep={ep}: out of range")
            continue
        ep_len = loader.get_episode_length(ep)
        if step >= ep_len:
            print(f"[w] skip ep={ep} step={step}: out of range (len={ep_len})")
            continue

        batch, data = build_batch(processor, loader, ep, step, embodiment_tag)
        tactile_gt = batch["tactile_gt"][0].float()

        pred_clean = predict_tactile(model, processor, batch, 1.0, use_clean_action=True)
        pred_flow = {}
        for t_val in (0.0, 0.5, 1.0):
            pred_flow[t_val] = predict_tactile(
                model, processor, batch, t_val, use_clean_action=False
            ).float().cpu().numpy()

        metrics = compute_metrics(pred_clean, tactile_gt)
        gate_mod_norm = float(
            (model.action_head.visor.gate * model.action_head.visor.gate_proj(
                pred_clean.mean(dim=0, keepdim=True)
            )).norm().detach().cpu()
        )
        loss, loss_stats = model.action_head.visor.compute_tactile_loss(
            pred_clean.unsqueeze(0),
            tactile_gt.unsqueeze(0).to(device=pred_clean.device, dtype=pred_clean.dtype),
        )

        eye_image = None
        if "robot0_eye_in_hand" in data.images:
            imgs = data.images["robot0_eye_in_hand"]
            eye_image = imgs[-1] if isinstance(imgs, list) else imgs

        plot_path = out_dir / f"ep{ep:03d}_step{step:04d}_{tag}_tactile.png"
        plot_sample(
            out_path=plot_path,
            ep=ep,
            step=step,
            tag=tag,
            metrics=metrics,
            gt=tactile_gt.numpy(),
            pred_clean=pred_clean.float().cpu().numpy(),
            pred_flow=pred_flow,
            eye_image=eye_image,
        )

        row = {
            "episode": ep,
            "step": step,
            "tag": tag,
            "gate": gate,
            "gate_mod_norm": gate_mod_norm,
            "tactile_loss": float(loss.detach().cpu()),
            "lambda_eff": float(loss_stats["lambda_eff"].detach().cpu()),
            **metrics,
            "plot": str(plot_path),
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    txt_lines = [
        f"Checkpoint: {ckpt}",
        f"Dataset: {args.dataset_path}",
        f"use_visor={use_visor}",
        f"gate (scalar)={gate:.6f}",
        "",
        "Per-sample metrics:",
    ]
    for r in results:
        txt_lines.append(
            f"  {r['tag']} ep={r['episode']} step={r['step']}: "
            f"contact_acc={r['contact_acc']:.3f} contact_recall={r['contact_recall']:.3f} "
            f"force_corr={r['force_corr']:.3f} force_mae={r['force_mae']:.3f} "
            f"gate_mod={r['gate_mod_norm']:.4f}"
        )
    txt_lines.append(f"\nSaved plots under: {out_dir}")
    (out_dir / "summary.txt").write_text("\n".join(txt_lines))
    print("\n".join(txt_lines[-8:]))


if __name__ == "__main__":
    main()
