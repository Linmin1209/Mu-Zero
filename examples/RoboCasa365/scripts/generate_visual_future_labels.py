#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate L1 pool-encoded visual_future labels for VISOR v4.1/v4.2.

Reads LeRobot videos only (no sim replay). For each parquet row *i* writes:
  - ``visual_future.manip``: frozen ResNet18 pool of eye_in_hand frame *i* → 256D
  - ``visual_future.nav``: mean pool of left/right agentview frames *i* → 256D

At training step *t*, ``visual_future`` delta_indices ``[0,5,...,35]`` stack eight
waypoint rows into shape ``(8, 256)`` for manip/nav supervision.

Example:
  python examples/RoboCasa365/scripts/generate_visual_future_labels.py \\
    --dataset /path/to/lerobot \\
    --output-dataset /path/to/lerobot_visual \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gr00t.utils.video_utils import get_frames_by_indices  # noqa: E402

VISUAL_DIM = 256
EYE_IN_HAND_KEY = "observation.images.robot0_eye_in_hand"
AGENTVIEW_LEFT_KEY = "observation.images.robot0_agentview_left"
AGENTVIEW_RIGHT_KEY = "observation.images.robot0_agentview_right"

VISUAL_FUTURE_FEATURE_TEMPLATE: dict[str, dict[str, Any]] = {
    "visual_future.manip": {
        "dtype": "float32",
        "shape": [VISUAL_DIM],
        "names": [f"latent_{i}" for i in range(VISUAL_DIM)],
    },
    "visual_future.nav": {
        "dtype": "float32",
        "shape": [VISUAL_DIM],
        "names": [f"latent_{i}" for i in range(VISUAL_DIM)],
    },
}


class FrozenPoolEncoder(nn.Module):
    """ResNet18 global-pool features projected to ``visual_dim`` (fixed weights)."""

    def __init__(self, visual_dim: int = VISUAL_DIM, device: torch.device | str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        torch.manual_seed(0)
        self.proj = nn.Linear(512, visual_dim).to(self.device)
        self.proj.eval()
        for param in self.proj.parameters():
            param.requires_grad = False
        self.preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2).float() / 255.0),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @torch.no_grad()
    def encode(self, frames_nhwc: np.ndarray, batch_size: int = 64) -> np.ndarray:
        if frames_nhwc.size == 0:
            return np.zeros((0, VISUAL_DIM), dtype=np.float32)
        out: list[np.ndarray] = []
        tensor = torch.from_numpy(frames_nhwc).to(self.device)
        for start in range(0, tensor.shape[0], batch_size):
            chunk = tensor[start : start + batch_size]
            x = self.preprocess(chunk)
            feats = self.backbone(x).flatten(1)
            vecs = self.proj(feats)
            out.append(vecs.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


def _patch_info_json(info_path: Path) -> None:
    info = json.loads(info_path.read_text())
    features = dict(info.get("features", {}))
    features.update(VISUAL_FUTURE_FEATURE_TEMPLATE)
    info["features"] = features
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def _patch_modality_json(modality_path: Path) -> None:
    modality = json.loads(modality_path.read_text())
    visual_future = modality.setdefault("visual_future", {})
    visual_future["manip"] = {"original_key": "visual_future.manip"}
    visual_future["nav"] = {"original_key": "visual_future.nav"}
    modality_path.write_text(json.dumps(modality, indent=4) + "\n")


def _prepare_output_dataset(
    input_dataset: Path,
    output_dataset: Path,
    overwrite: bool,
) -> None:
    if output_dataset.resolve() == input_dataset.resolve():
        return
    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset exists: {output_dataset}. Pass --overwrite to replace."
            )
        shutil.rmtree(output_dataset)
    shutil.copytree(
        input_dataset,
        output_dataset,
        ignore=shutil.ignore_patterns("data"),
    )
    (output_dataset / "data").mkdir(parents=True, exist_ok=True)
    _patch_info_json(output_dataset / "meta" / "info.json")
    _patch_modality_json(output_dataset / "meta" / "modality.json")


def _load_info(dataset: Path) -> dict[str, Any]:
    return json.loads((dataset / "meta" / "info.json").read_text())


def _episode_parquet_path(dataset: Path, episode_index: int, info: dict[str, Any]) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return dataset / rel


def _video_path(dataset: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    rel = info["video_path"].format(
        episode_chunk=chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    return dataset / rel


def _label_episode(
    *,
    input_dataset: Path,
    output_dataset: Path,
    episode_index: int,
    encoder: FrozenPoolEncoder,
    encode_batch_size: int,
) -> dict[str, Any]:
    info = _load_info(input_dataset)
    src_parquet = _episode_parquet_path(input_dataset, episode_index, info)
    dst_parquet = _episode_parquet_path(output_dataset, episode_index, info)
    df = pd.read_parquet(src_parquet)
    num_frames = len(df)

    eye_path = _video_path(input_dataset, info, episode_index, EYE_IN_HAND_KEY)
    left_path = _video_path(input_dataset, info, episode_index, AGENTVIEW_LEFT_KEY)
    right_path = _video_path(input_dataset, info, episode_index, AGENTVIEW_RIGHT_KEY)
    indices = np.arange(num_frames, dtype=np.int64)

    eye_frames = get_frames_by_indices(str(eye_path), indices)
    left_frames = get_frames_by_indices(str(left_path), indices)
    right_frames = get_frames_by_indices(str(right_path), indices)

    manip_vecs = encoder.encode(eye_frames, batch_size=encode_batch_size)
    left_vecs = encoder.encode(left_frames, batch_size=encode_batch_size)
    right_vecs = encoder.encode(right_frames, batch_size=encode_batch_size)
    nav_vecs = 0.5 * (left_vecs + right_vecs)

    df["visual_future.manip"] = [row.tolist() for row in manip_vecs]
    df["visual_future.nav"] = [row.tolist() for row in nav_vecs]

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False)
    return {
        "episode_index": episode_index,
        "num_frames": num_frames,
        "parquet_path": str(dst_parquet),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Defaults to --dataset (in-place update).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dataset = args.dataset.resolve()
    output_dataset = (args.output_dataset or args.dataset).resolve()
    _prepare_output_dataset(input_dataset, output_dataset, overwrite=args.overwrite)

    info = _load_info(input_dataset)
    total_episodes = int(info["total_episodes"])
    max_episodes = args.max_episodes or total_episodes
    episode_indices = list(range(min(max_episodes, total_episodes)))

    device = args.device if torch.cuda.is_available() else "cpu"
    encoder = FrozenPoolEncoder(visual_dim=VISUAL_DIM, device=device)

    results: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        result = _label_episode(
            input_dataset=input_dataset,
            output_dataset=output_dataset,
            episode_index=episode_index,
            encoder=encoder,
            encode_batch_size=args.encode_batch_size,
        )
        results.append(result)
        print(
            f"[ok] episode {episode_index}: {result['num_frames']} frames -> {result['parquet_path']}"
        )

    summary_path = output_dataset / "meta" / "visual_future_label_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "visual_dim": VISUAL_DIM,
                "encoder": "resnet18_imagenet_pool_256",
                "eye_in_hand_key": EYE_IN_HAND_KEY,
                "nav_keys": [AGENTVIEW_LEFT_KEY, AGENTVIEW_RIGHT_KEY],
                "episodes": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] wrote {len(results)} episodes; summary: {summary_path}")


if __name__ == "__main__":
    main()
