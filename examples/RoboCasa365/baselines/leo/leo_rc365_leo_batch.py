#!/usr/bin/env python3
"""Collate RoboCasa365 manifest samples into LEO LeoAgent data_dict batches."""

from __future__ import annotations

from typing import Any

from leo_rc365_sanitize import sanitize_leo_batch_3d
import torch
import torch.nn.functional as F

DEFAULT_MAX_OBJ_LEN = 60
LEO_OBJ_FTS_PAD = 1.0
LEO_OBJ_LOCS_PAD = 0.0
PIX_MEAN = torch.tensor([0.485, 0.456, 0.406])
PIX_STD = torch.tensor([0.229, 0.224, 0.225])

LEO_ROLE_PROMPT = (
    "You are an AI visual assistant situated in a 3D scene. "
    "You can perceive (1) an ego-view image (accessible when necessary) and "
    "(2) the objects (including yourself) in the scene (always accessible). "
    "You should properly respond to the USER's instruction according to the given visual information. "
)
LEO_OBJECTS_PROMPT = "Objects (including you) in the scene:"


def get_leo_prompts(instruction: str, situation: str = "") -> dict[str, str]:
    situation_prompt = situation or "You are in a kitchen manipulation scene."
    return {
        "prompt_before_obj": LEO_ROLE_PROMPT + situation_prompt,
        "prompt_middle_1": "Ego-view images (left, right, eye-in-hand):",
        "prompt_middle_2": LEO_OBJECTS_PROMPT,
        "prompt_after_obj": f"USER: {instruction} ASSISTANT:",
    }


def _preprocess_view_chw(img_chw: torch.Tensor) -> torch.Tensor:
    """Dataset images are float01 CHW; LEO expects ImageNet-normalized CHW."""
    if img_chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(img_chw.shape)}")
    x = img_chw.float().unsqueeze(0)
    if x.shape[-1] != 224 or x.shape[-2] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - PIX_MEAN.view(1, 3, 1, 1)) / PIX_STD.view(1, 3, 1, 1)
    return x.squeeze(0)


def _leo_rgb_to_minus1_1(obj_fts: torch.Tensor) -> torch.Tensor:
    """LEO preprocess: RGB in [0, 1] -> [-1, 1] (see data/eai.py)."""
    if obj_fts.numel() == 0:
        return obj_fts
    rgb = obj_fts[..., 3:6]
    if float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0:
        out = obj_fts.clone()
        out[..., 3:6] = out[..., 3:6] * 2.0 - 1.0
        return out
    return obj_fts


def leo_pad_objects(
    obj_fts: torch.Tensor,
    obj_locs: torch.Tensor,
    max_obj_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match LEO LeoObjPadDatasetWrapper padding conventions."""
    n_obj = int(obj_fts.shape[0])
    num_points = int(obj_fts.shape[1]) if obj_fts.ndim == 3 else 1024
    feat_dim = int(obj_fts.shape[2]) if obj_fts.ndim == 3 else 6

    if n_obj == 0:
        obj_fts = torch.empty(0, num_points, feat_dim, dtype=torch.float32)
        obj_locs = torch.empty(0, 6, dtype=torch.float32)
    else:
        obj_fts = _leo_rgb_to_minus1_1(obj_fts.float())
        obj_locs = obj_locs.float()

    if n_obj < max_obj_len:
        pad_n = max_obj_len - n_obj
        obj_fts = torch.cat(
            [
                obj_fts,
                torch.full((pad_n, num_points, feat_dim), LEO_OBJ_FTS_PAD, dtype=obj_fts.dtype),
            ],
            dim=0,
        )
        obj_locs = torch.cat(
            [
                obj_locs,
                torch.full((pad_n, 6), LEO_OBJ_LOCS_PAD, dtype=obj_locs.dtype),
            ],
            dim=0,
        )
    else:
        obj_fts = obj_fts[:max_obj_len]
        obj_locs = obj_locs[:max_obj_len]
        n_obj = max_obj_len

    # LEO: True = valid object, False = padded slot.
    obj_masks = torch.arange(max_obj_len) < n_obj
    return obj_fts, obj_locs, obj_masks


def sample_to_leo_dict(sample: dict[str, Any], *, use_3d: bool) -> dict[str, Any]:
    language = str(sample.get("language") or sample.get("task") or "manipulation")
    task = str(sample.get("task") or "RoboCasa365")
    situation = (
        f"You are controlling a Panda mobile manipulator in a RoboCasa365 kitchen. "
        f"Current task: {task}."
    )
    prompts = get_leo_prompts(instruction=language, situation=situation)

    images = sample["images"]
    if images.ndim != 4:
        raise ValueError(f"Expected images (V,C,H,W), got {tuple(images.shape)}")
    view_imgs = [_preprocess_view_chw(images[i]) for i in range(images.shape[0])]
    img_fts = torch.stack(view_imgs, dim=0)

    state = sample["state"].float()
    action = sample["action"].float()

    if use_3d and sample.get("has_3d") and "obj_fts" in sample:
        obj_fts = sample["obj_fts"].float()
        obj_locs = sample["obj_locs"].float()
        if obj_fts.ndim == 2:
            obj_fts = obj_fts.unsqueeze(0)
        if obj_locs.ndim == 1:
            obj_locs = obj_locs.unsqueeze(0)
        obj_fts = _leo_rgb_to_minus1_1(obj_fts)
        anchor_locs = sample.get("anchor_locs", torch.zeros(3)).float()
        anchor_orientation = sample.get(
            "anchor_orientation", torch.tensor([0.0, 0.0, 0.0, 1.0])
        ).float()
    else:
        num_points = int(sample.get("num_points", 1024))
        obj_fts = torch.zeros(0, num_points, 6)
        obj_locs = torch.zeros(0, 6)
        anchor_locs = torch.zeros(3)
        anchor_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])

    return {
        "source": "robocasa365",
        **prompts,
        "obj_fts": obj_fts,
        "obj_locs": obj_locs,
        "anchor_locs": anchor_locs,
        "anchor_orientation": anchor_orientation,
        "img_fts": img_fts,
        "img_masks": torch.tensor([1.0], dtype=torch.float32),
        "state": state,
        "action": action,
        "task": task,
    }


def _pad_obj_batch(
    obj_fts_list: list[torch.Tensor],
    obj_locs_list: list[torch.Tensor],
    max_obj_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_fts, padded_locs, padded_masks = [], [], []
    for fts, locs in zip(obj_fts_list, obj_locs_list):
        fts, locs, masks = leo_pad_objects(fts, locs, max_obj_len=max_obj_len)
        padded_fts.append(fts)
        padded_locs.append(locs)
        padded_masks.append(masks)
    return (
        torch.stack(padded_fts, dim=0),
        torch.stack(padded_locs, dim=0),
        torch.stack(padded_masks, dim=0),
    )


def collate_leo_batch(
    samples: list[dict[str, Any]],
    *,
    use_3d: bool,
    max_obj_len: int = DEFAULT_MAX_OBJ_LEN,
) -> dict[str, Any]:
    leo_samples = [sample_to_leo_dict(s, use_3d=use_3d) for s in samples]

    obj_fts, obj_locs, obj_masks = _pad_obj_batch(
        [s["obj_fts"] for s in leo_samples],
        [s["obj_locs"] for s in leo_samples],
        max_obj_len=max_obj_len,
    )

    batch: dict[str, Any] = {
        "source": [s["source"] for s in leo_samples],
        "prompt_before_obj": [s["prompt_before_obj"] for s in leo_samples],
        "prompt_middle_1": [s["prompt_middle_1"] for s in leo_samples],
        "prompt_middle_2": [s["prompt_middle_2"] for s in leo_samples],
        "prompt_after_obj": [s["prompt_after_obj"] for s in leo_samples],
        "obj_fts": obj_fts,
        "obj_locs": obj_locs,
        "obj_masks": obj_masks,
        "anchor_locs": torch.stack([s["anchor_locs"] for s in leo_samples], dim=0),
        "anchor_orientation": torch.stack([s["anchor_orientation"] for s in leo_samples], dim=0),
        "img_fts": torch.stack([s["img_fts"] for s in leo_samples], dim=0),
        "img_masks": torch.stack([s["img_masks"] for s in leo_samples], dim=0),
        "state": torch.stack([s["state"] for s in leo_samples], dim=0),
        "action": torch.stack([s["action"] for s in leo_samples], dim=0),
        "task": [s["task"] for s in leo_samples],
    }
    return sanitize_leo_batch_3d(batch)
