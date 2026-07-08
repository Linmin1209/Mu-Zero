# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa365 LeRobot dataset for FLUX.1-Fill-dev LoRA (current → future frame pairs)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from gr00t.data.flux_inpainting import DEFAULT_FLUX_PROMPT
from gr00t.data.flux_lora import mask_image_from_mode, prepare_mask_and_masked_image, snap_dim
from gr00t.utils.video_utils import get_frames_by_indices


@dataclass(frozen=True)
class FluxFrameSample:
    episode_index: int
    current_index: int
    future_index: int
    prompt: str


class RoboCasa365FluxFillDataset(Dataset):
    """(I_t, I_{t+k}) pairs from LeRobot video for Fill LoRA training."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        video_key: str = "observation.images.robot0_eye_in_hand",
        future_delta: int = 5,
        resolution: int = 256,
        mask_mode: str = "keep_reference",
        default_prompt: str = DEFAULT_FLUX_PROMPT,
        max_episodes: int | None = None,
        frame_stride: int = 1,
        center_crop: bool = False,
        random_flip: bool = False,
        seed: int = 42,
    ):
        self.dataset_path = Path(dataset_path)
        self.video_key = video_key
        self.future_delta = int(future_delta)
        self.resolution = int(resolution)
        self.mask_mode = mask_mode
        self.default_prompt = default_prompt
        self.center_crop = center_crop
        self.random_flip = bool(random_flip)
        self.rng = random.Random(seed)

        info = json.loads((self.dataset_path / "meta" / "info.json").read_text())
        self.info = info
        self.chunk_size = int(info["chunks_size"])
        self.video_path_pattern = info["video_path"]

        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]

        self.samples: list[FluxFrameSample] = []
        stride = max(1, int(frame_stride))
        delta = max(1, self.future_delta)
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            prompt = episode["tasks"][0] if episode.get("tasks") else self.default_prompt
            for current_index in range(0, length - delta, stride):
                self.samples.append(
                    FluxFrameSample(
                        episode_index=episode_index,
                        current_index=current_index,
                        future_index=current_index + delta,
                        prompt=prompt,
                    )
                )

        if not self.samples:
            raise ValueError(f"No training samples found in {self.dataset_path}")

        self.train_resize = transforms.Resize(
            self.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.train_crop = (
            transforms.CenterCrop(self.resolution)
            if center_crop
            else transforms.RandomCrop(self.resolution)
        )
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.resize_and_crop = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution)
                if center_crop
                else transforms.RandomCrop(self.resolution),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _video_path(self, episode_index: int) -> Path:
        chunk_idx = episode_index // self.chunk_size
        rel = self.video_path_pattern.format(
            episode_chunk=chunk_idx,
            video_key=self.video_key,
            episode_index=episode_index,
        )
        return self.dataset_path / rel

    @staticmethod
    def _frame_to_pil(frame: np.ndarray) -> Image.Image:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr, mode="RGB")

    def _transform_pair(
        self,
        current: Image.Image,
        future: Image.Image,
    ) -> tuple[Image.Image, torch.Tensor]:
        current = current.convert("RGB")
        future = future.convert("RGB")

        current = self.train_resize(current)
        future = self.train_resize(future)

        if self.random_flip and self.rng.random() < 0.5:
            current = self.train_flip(current)
            future = self.train_flip(future)

        if self.center_crop:
            current = self.train_crop(current)
            future = self.train_crop(future)
        else:
            y1, x1, h, w = self.train_crop.get_params(current, (self.resolution, self.resolution))
            current = crop(current, y1, x1, h, w)
            future = crop(future, y1, x1, h, w)

        future_tensor = self.to_tensor(future)
        return current, future_tensor

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        video_path = self._video_path(sample.episode_index)
        frames = get_frames_by_indices(
            str(video_path),
            [sample.current_index, sample.future_index],
            decoder_kwargs={"device": "cpu", "dimension_order": "NHWC", "num_ffmpeg_threads": 0},
        )
        current_pil = self._frame_to_pil(frames[0])
        future_pil = self._frame_to_pil(frames[1])

        width = snap_dim(current_pil.width)
        height = snap_dim(current_pil.height)
        if current_pil.size != (width, height):
            current_pil = current_pil.resize((width, height), Image.Resampling.BILINEAR)
            future_pil = future_pil.resize((width, height), Image.Resampling.BILINEAR)

        current_pil, future_tensor = self._transform_pair(current_pil, future_pil)
        mask_pil = mask_image_from_mode(current_pil.size, self.mask_mode)
        mask_tensor, masked_image = prepare_mask_and_masked_image(current_pil, mask_pil)

        return {
            "pixel_values": future_tensor,
            "masks": mask_tensor.squeeze(0),
            "masked_images": masked_image.squeeze(0),
            "prompts": sample.prompt,
        }


def collate_flux_fill_batch(examples: list[dict]) -> dict:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    masks = torch.stack([example["masks"] for example in examples])
    masked_images = torch.stack([example["masked_images"] for example in examples])
    prompts = [example["prompts"] for example in examples]
    return {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "masks": masks,
        "masked_images": masked_images,
        "prompts": prompts,
    }
