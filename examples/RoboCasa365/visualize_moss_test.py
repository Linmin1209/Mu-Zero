#!/usr/bin/env python3
"""Small visualization test for STSS/MOSS motion module.

Two modes:
  synthetic  - fast demo on shifted synthetic features (no checkpoint)
  checkpoint - run finetuned motion ckpt on RoboCasa365 train set or eval MP4

Example:
  cd Isaac-GR00T
  .venv/bin/python examples/RoboCasa365/visualize_moss_test.py --mode synthetic

  # RoboCasa365 training dataset (4-frame history + 3 views, same as finetune)
  .venv/bin/python examples/RoboCasa365/visualize_moss_test.py \\
    --mode checkpoint \\
    --dataset-root /path/to/robocasa365-datasets \\
    --task PickPlaceToasterToCounter --episode 0 --step 100

  .venv/bin/python examples/RoboCasa365/visualize_moss_test.py \\
    --mode checkpoint \\
    --checkpoint output/rc365_PickPlaceToasterToCounter_30k_b64_4frame_motion/checkpoint-30000 \\
    --video output/robocasa365_eval/.../videos/xxx_s1.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gr00t.model.modules.motion import STSSTransformation  # noqa: E402
from gr00t.model.modules.qwen3_motion import (  # noqa: E402
    _visual_forward_with_motion,
    apply_moss,
)

DEFAULT_DATASET_ROOT = Path(
    "/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/robocasa365-datasets"
)
VIDEO_DELTA_INDICES = [-6, -4, -2, 0]
VIEW_KEYS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
DEFAULT_DISPLAY_VIEW = "robot0_eye_in_hand"


def resolve_display_view_index(view_keys: list[str], display_view: str) -> int:
    if display_view not in view_keys:
        raise ValueError(
            f"display_view={display_view!r} not in view_keys={view_keys}. "
            f"Choose one of: {view_keys}"
        )
    return view_keys.index(display_view)


@dataclass
class MossCapture:
    stss_corr: torch.Tensor | None = None
    moss_delta: torch.Tensor | None = None
    hidden_before: torch.Tensor | None = None
    grid_thw: torch.Tensor | None = None


def stss_to_motion_map(stss: torch.Tensor) -> torch.Tensor:
    """Convert STSS correlation volume to per-patch motion energy.

    Args:
        stss: (b, t, h, w, 1, l, u, v) cosine correlations in a local window.
    Returns:
        (b, t, h, w) motion energy in [0, 2] (higher = more motion).
    """
    center_u = stss.shape[-2] // 2
    center_v = stss.shape[-1] // 2
    center = stss[..., center_u, center_v].mean(dim=-1).squeeze(-1)
    motion_from_center = 1.0 - center
    max_off_center = stss.squeeze(4).amax(dim=(-2, -1)).mean(dim=-1)
    return 0.5 * motion_from_center + 0.5 * (1.0 - max_off_center)


def upsample_map(patch_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(patch_map).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def save_frame_grid(
    out_path: Path,
    frames: list[np.ndarray],
    heatmaps: list[np.ndarray],
    titles: list[str],
    heatmap_label: str,
) -> None:
    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(n):
        axes[0, i].imshow(frames[i])
        axes[0, i].set_title(titles[i])
        axes[0, i].axis("off")

        im = axes[1, i].imshow(heatmaps[i], cmap="inferno", vmin=0.0, vmax=np.percentile(heatmaps[i], 99))
        axes[1, i].set_title(f"{heatmap_label} t={i}")
        axes[1, i].axis("off")
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_synthetic_rgb_frames(
    num_frames: int = 4,
    height: int = 128,
    width: int = 128,
    shift: int = 8,
) -> np.ndarray:
    """RGB frames with a moving white square on gray background."""
    frames = []
    for t in range(num_frames):
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        y0 = height // 3
        x0 = width // 8 + t * shift
        y1, x1 = y0 + 24, x0 + 24
        img[y0:y1, x0:x1] = 230
        frames.append(img)
    return np.stack(frames, axis=0)


def make_synthetic_feature_video(
    num_frames: int,
    height: int,
    width: int,
    dim: int,
    shift: int,
    device: torch.device,
) -> torch.Tensor:
    """Feature tensor (T*H*W, C) with a localized bump that shifts over time."""
    feats = torch.randn(num_frames, height, width, dim, device=device) * 0.05
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    for t in range(num_frames):
        cx = width // 4 + t * shift
        cy = height // 2
        blob = torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2)))
        feats[t] += blob.unsqueeze(-1) * 2.0
    return feats.reshape(num_frames * height * width, dim)


def run_synthetic_demo(out_dir: Path, device: str) -> Path:
    """Show STSS responds to motion vs static using STSSTransformation."""
    dev = torch.device(device)
    num_frames, h, w = 4, 16, 16
    dim = 64
    shift = 2

    stss = STSSTransformation(window=(5, 9, 9)).to(dev)
    stss.eval()

    moving = make_synthetic_feature_video(num_frames, h, w, dim, shift=shift, device=dev)
    static = make_synthetic_feature_video(num_frames, h, w, dim, shift=0, device=dev)
    grid = torch.tensor([[num_frames, h, w]], dtype=torch.long, device=dev)

    with torch.no_grad():
        corr_moving = stss(moving, grid)
        corr_static = stss(static, grid)

    motion_moving = stss_to_motion_map(corr_moving).squeeze(0).numpy()
    motion_static = stss_to_motion_map(corr_static).squeeze(0).numpy()

    rgb = make_synthetic_rgb_frames(num_frames=num_frames, shift=shift * 4)
    moving_maps = [upsample_map(motion_moving[t], (rgb.shape[1], rgb.shape[2])) for t in range(num_frames)]
    static_maps = [upsample_map(motion_static[t], (rgb.shape[1], rgb.shape[2])) for t in range(num_frames)]

    out_moving = out_dir / "synthetic_moving_stss.png"
    out_static = out_dir / "synthetic_static_stss.png"
    save_frame_grid(
        out_moving,
        [rgb[t] for t in range(num_frames)],
        moving_maps,
        [f"moving frame {t}" for t in range(num_frames)],
        "STSS motion",
    )
    save_frame_grid(
        out_static,
        [rgb[t] for t in range(num_frames)],
        static_maps,
        [f"static frame {t}" for t in range(num_frames)],
        "STSS motion",
    )

    summary = out_dir / "synthetic_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "Synthetic STSS sanity check (STSSTransformation only)",
                f"moving STSS mean: {motion_moving.mean():.4f}, max: {motion_moving.max():.4f}",
                f"static STSS mean: {motion_static.mean():.4f}, max: {motion_static.max():.4f}",
                "",
                "Expect: moving > static, especially around the moving blob.",
                f"Saved: {out_moving}",
                f"Saved: {out_static}",
            ]
        )
    )
    print(summary.read_text())
    return out_moving


def read_video_frames(video_path: Path, num_frames: int = 4) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < num_frames:
        raise RuntimeError(f"Expected {num_frames} frames, got {len(frames)} from {video_path}")
    return np.stack(frames, axis=0)


def load_training_dataset_frames(
    dataset_root: Path,
    task: str,
    episode: int,
    step: int,
    split: str = "pretrain",
    category: str = "atomic",
    view_keys: list[str] | None = None,
    delta_indices: list[int] | None = None,
    video_backend: str = "opencv",
) -> tuple[dict[str, np.ndarray], str, Path, list[int]]:
    """Load 4-frame video history from RoboCasa365 LeRobot training data."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ModalityConfig
    from gr00t.experiment.robocasa365_datasets import resolve_robocasa365_dataset_paths

    view_keys = view_keys or VIEW_KEYS
    delta_indices = delta_indices or VIDEO_DELTA_INDICES
    min_step = -min(delta_indices)
    if step < min_step:
        raise ValueError(f"--step must be >= {min_step} for delta_indices={delta_indices}")

    dataset_paths = resolve_robocasa365_dataset_paths(
        root=dataset_root,
        split=split,
        category=category,
        tasks=task,
    )
    dataset_path = Path(dataset_paths[0])

    modality_configs = {
        "video": ModalityConfig(delta_indices=delta_indices, modality_keys=view_keys),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }
    loader = LeRobotEpisodeLoader(
        dataset_path,
        modality_configs,
        video_backend=video_backend,
    )
    if episode < 0 or episode >= len(loader):
        raise IndexError(f"episode {episode} out of range [0, {len(loader) - 1}]")

    episode_data = loader[episode]
    if step >= len(episode_data):
        raise IndexError(f"step {step} out of range [0, {len(episode_data) - 1}] for episode {episode}")

    step_data = extract_step_data(
        episode_data,
        step,
        modality_configs,
        EmbodimentTag.ROBOCASA_PANDA_OMRON,
    )
    frames_by_view: dict[str, np.ndarray] = {}
    for view in view_keys:
        raw_frames = step_data.images[view]
        frames_by_view[view] = np.stack([np.asarray(f) for f in raw_frames], axis=0)

    frame_indices = [step + delta for delta in delta_indices]
    return frames_by_view, step_data.text, dataset_path, frame_indices


def build_pil_frames_multiview(
    frames_by_view: dict[str, np.ndarray],
    view_keys: list[str],
) -> list:
    """Match GR00T processor order: for each timestep, iterate all views."""
    from PIL import Image

    num_frames = next(iter(frames_by_view.values())).shape[0]
    pil_frames = []
    for t in range(num_frames):
        for view in view_keys:
            pil_frames.append(Image.fromarray(frames_by_view[view][t]))
    return pil_frames


def load_visual_from_checkpoint(checkpoint: Path, device: str):
    from transformers import AutoConfig, AutoModel

    import gr00t.model.gr00t_n1d7.setup  # noqa: F401  # register Gr00tN1d7

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    if not getattr(config, "use_motion", False):
        raise ValueError(f"Checkpoint {checkpoint} has use_motion=False")

    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    visual = model.backbone.model.visual
    visual.to(device)
    return model, visual, config


def run_checkpoint_demo(
    checkpoint: Path,
    out_dir: Path,
    device: str,
    num_frames: int,
    video: Path | None = None,
    frames_by_view: dict[str, np.ndarray] | None = None,
    view_keys: list[str] | None = None,
    display_view: str = DEFAULT_DISPLAY_VIEW,
    language: str = "Visualize MOSS motion features.",
    source_desc: str = "",
) -> Path:
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import build_processor

    import gr00t.model.gr00t_n1d7.setup  # noqa: F401

    dev = torch.device(device)
    _, visual, config = load_visual_from_checkpoint(checkpoint, device)

    loading_kwargs = {"trust_remote_code": True, "local_files_only": True}
    processor = build_processor(config.model_name, loading_kwargs)

    view_keys = view_keys or VIEW_KEYS
    display_view_idx = resolve_display_view_index(view_keys, display_view)
    if frames_by_view is not None:
        num_frames_eff = next(iter(frames_by_view.values())).shape[0]
        num_views = len(view_keys)
        rgb_frames = frames_by_view[display_view]
        pil_frames = build_pil_frames_multiview(frames_by_view, view_keys)
        text = language
    elif video is not None:
        rgb_frames = read_video_frames(video, num_frames=num_frames)
        pil_frames = [__import__("PIL").Image.fromarray(f) for f in rgb_frames]
        num_views = 1
        num_frames_eff = num_frames
        text = language
    else:
        raise ValueError("Provide either video path or frames_by_view")

    from PIL import Image
    if not isinstance(pil_frames[0], Image.Image):
        pil_frames = [Image.fromarray(np.asarray(f)) for f in pil_frames]

    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_frames],
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[prompt], images=pil_frames, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(dev, dtype=torch.bfloat16)
    grid_thw = inputs["image_grid_thw"].to(dev)

    capture = MossCapture()
    stss = visual.motion_block.stss_encoders[0].stss_transformation

    def _stss_hook(_module, _inp, out):
        capture.stss_corr = out.detach().float().cpu()

    stss_handle = stss.register_forward_hook(_stss_hook)

    orig_apply = apply_moss

    def apply_moss_with_capture(
        visual_obj,
        hidden_states,
        grid_thw_in,
        num_frames,
        num_views,
    ):
        capture.hidden_before = hidden_states.detach().float().cpu()
        capture.grid_thw = grid_thw_in.detach().cpu()
        hidden_before = hidden_states
        moss_delta = orig_apply(
            visual_obj,
            hidden_states,
            grid_thw_in,
            num_frames=num_frames,
            num_views=num_views,
        )
        capture.moss_delta = (moss_delta - hidden_before).detach().float().cpu()
        return moss_delta

    import gr00t.model.modules.qwen3_motion as qwen3_motion

    qwen3_motion.apply_moss = apply_moss_with_capture
    try:
        with torch.no_grad():
            _ = _visual_forward_with_motion(
                visual,
                pixel_values,
                grid_thw,
                num_frames=num_frames_eff,
                num_views=num_views,
            )
    finally:
        qwen3_motion.apply_moss = orig_apply
        stss_handle.remove()

    if capture.stss_corr is None or capture.moss_delta is None:
        raise RuntimeError("Failed to capture MOSS/STSS tensors")

    t, h, w = capture.grid_thw[0].tolist()
    num_patches = int(h * w)
    stss_all = stss_to_motion_map(capture.stss_corr).numpy()
    if stss_all.shape[0] == num_views:
        stss_motion = stss_all[display_view_idx]
    else:
        stss_motion = stss_all[0]

    hidden_dim = capture.moss_delta.shape[-1]
    delta_5d = capture.moss_delta.reshape(1, num_views, num_frames_eff, num_patches, hidden_dim)

    target_hw = (rgb_frames.shape[1], rgb_frames.shape[2])
    stss_maps = [
        upsample_map(stss_motion[t_i], target_hw) for t_i in range(num_frames_eff)
    ]
    moss_maps = [
        upsample_map(
            delta_5d[0, display_view_idx, t_i].norm(dim=-1).numpy().reshape(int(h), int(w)),
            target_hw,
        )
        for t_i in range(num_frames_eff)
    ]
    delta_norm_all = delta_5d[0, display_view_idx].norm(dim=-1).numpy()

    out_stss = out_dir / "checkpoint_stss_motion.png"
    out_moss = out_dir / "checkpoint_moss_delta.png"
    if frames_by_view is not None:
        out_stss = out_dir / "dataset_stss_motion.png"
        out_moss = out_dir / "dataset_moss_delta.png"
    save_frame_grid(
        out_stss,
        [rgb_frames[t_i] for t_i in range(num_frames_eff)],
        stss_maps,
        [f"delta={VIDEO_DELTA_INDICES[t_i]}" for t_i in range(num_frames_eff)],
        "STSS motion",
    )
    save_frame_grid(
        out_moss,
        [rgb_frames[t_i] for t_i in range(num_frames_eff)],
        moss_maps,
        [f"frame {t_i}" for t_i in range(num_frames_eff)],
        "MOSS |delta|",
    )

    summary = out_dir / "checkpoint_summary.txt"
    if frames_by_view is not None:
        summary = out_dir / "dataset_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"Checkpoint: {checkpoint}",
                f"Source: {source_desc}",
                f"display_view={display_view}",
                f"num_frames={num_frames_eff}, num_views={num_views}, delta_indices={VIDEO_DELTA_INDICES}",
                f"grid_thw: T={t}, H={h}, W={w}, patches={num_patches}",
                f"STSS motion mean: {stss_motion.mean():.4f}, max: {stss_motion.max():.4f}",
                f"MOSS delta norm mean: {delta_norm_all.mean():.4f}, max: {delta_norm_all.max():.4f}",
                f"Saved: {out_stss}",
                f"Saved: {out_moss}",
            ]
        )
    )
    print(summary.read_text())
    return out_stss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize STSS/MOSS motion module")
    parser.add_argument("--mode", choices=["synthetic", "checkpoint"], default="synthetic")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "output/rc365_PickPlaceToasterToCounter_30k_b64_4frame_motion/checkpoint-30000",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="MP4 path for checkpoint mode (omit when using --dataset-root)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="RoboCasa365 dataset root; when set, load training frames instead of --video",
    )
    parser.add_argument("--task", default="PickPlaceToasterToCounter")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--category", default="atomic")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--video-backend", default="opencv")
    parser.add_argument(
        "--display-view",
        default=DEFAULT_DISPLAY_VIEW,
        choices=VIEW_KEYS,
        help="Camera used for displayed RGB frames and heatmap overlay",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output/moss_visualize_test",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-frames", type=int, default=4)
    return parser.parse_args()


def default_video_path() -> Path:
    candidates = sorted(
        (
            PROJECT_ROOT
            / "output/robocasa365_eval/checkpoint-30000_atomic_seen_pretrain_exp20260609_014208/PickPlaceToasterToCounter/videos"
        ).glob("*.mp4")
    )
    if not candidates:
        raise FileNotFoundError("No default eval videos found; pass --video explicitly.")
    return candidates[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "synthetic":
        run_synthetic_demo(args.output_dir, args.device)
        return

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.dataset_root is not None:
        frames_by_view, language, dataset_path, frame_indices = load_training_dataset_frames(
            dataset_root=args.dataset_root,
            task=args.task,
            episode=args.episode,
            step=args.step,
            split=args.split,
            category=args.category,
            video_backend=args.video_backend,
        )
        source_desc = (
            f"{dataset_path} episode={args.episode} step={args.step} "
            f"frame_indices={frame_indices} task={args.task}"
        )
        run_checkpoint_demo(
            checkpoint=args.checkpoint,
            out_dir=args.output_dir,
            device=args.device,
            num_frames=args.num_frames,
            frames_by_view=frames_by_view,
            display_view=args.display_view,
            language=language,
            source_desc=source_desc,
        )
        return

    video = args.video or default_video_path()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}; pass --dataset-root to use training data")

    run_checkpoint_demo(
        checkpoint=args.checkpoint,
        out_dir=args.output_dir,
        device=args.device,
        num_frames=args.num_frames,
        video=video,
        display_view=args.display_view,
        source_desc=str(video),
    )


if __name__ == "__main__":
    main()
