#!/usr/bin/env python3
"""NaN/Inf checks and safe defaults for LEO 3D inputs (RoboCasa365)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# LEO obj_locs: xyz (m) + whd (m). Keep whd above eps for pairwise geom divisions.
LOC_XYZ_CLIP = 20.0
LOC_WHD_MIN = 1e-2
LOC_WHD_MAX = 20.0
FT_CLIP = 10.0
DEFAULT_QUAT_WXYZ = (0.0, 0.0, 0.0, 1.0)


def _finite_or_default(arr: np.ndarray, default: float = 0.0) -> tuple[np.ndarray, int]:
    out = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(out)
    n_bad = int(bad.sum())
    if n_bad:
        out = out.copy()
        out[bad] = default
    return out, n_bad


def sanitize_3d_numpy(
    obj_fts: np.ndarray,
    obj_locs: np.ndarray,
    *,
    anchor_locs: np.ndarray | None = None,
    anchor_orientation: np.ndarray | None = None,
    loc_whd_min: float = LOC_WHD_MIN,
) -> dict[str, Any]:
    """Clean raw npz arrays before torch conversion."""
    obj_fts = np.asarray(obj_fts, dtype=np.float32)
    if obj_fts.ndim == 2:
        obj_fts = obj_fts[None, ...]
    obj_locs = np.asarray(obj_locs, dtype=np.float32)
    if obj_locs.ndim == 1:
        obj_locs = obj_locs[None, ...]

    stats = {"n_bad_fts": 0, "n_bad_locs": 0, "n_bad_anchor": 0, "n_invalid_obj": 0}

    cleaned_fts = []
    cleaned_locs = []
    valid_masks = []
    for i in range(obj_fts.shape[0]):
        fts, n1 = _finite_or_default(obj_fts[i])
        locs, n2 = _finite_or_default(obj_locs[i] if i < obj_locs.shape[0] else np.zeros(6, np.float32))
        stats["n_bad_fts"] += n1
        stats["n_bad_locs"] += n2

        fts = np.clip(fts, -FT_CLIP, FT_CLIP)
        locs = np.clip(locs, -LOC_XYZ_CLIP, LOC_XYZ_CLIP)
        locs[3:6] = np.clip(locs[3:6], loc_whd_min, LOC_WHD_MAX)

        obj_valid = bool(np.isfinite(fts).all() and np.isfinite(locs).all())
        if not obj_valid:
            stats["n_invalid_obj"] += 1
            fts = np.zeros_like(fts)
            locs = np.zeros_like(locs)

        cleaned_fts.append(fts)
        cleaned_locs.append(locs)
        valid_masks.append(obj_valid)

    out_anchor_locs = None
    if anchor_locs is not None:
        out_anchor_locs, n = _finite_or_default(anchor_locs)
        stats["n_bad_anchor"] += n
        out_anchor_locs = np.clip(out_anchor_locs, -LOC_XYZ_CLIP, LOC_XYZ_CLIP)

    out_anchor_orient = None
    if anchor_orientation is not None:
        out_anchor_orient, n = _finite_or_default(anchor_orientation)
        stats["n_bad_anchor"] += n
        if out_anchor_orient.size >= 4:
            norm = float(np.linalg.norm(out_anchor_orient[:4]))
            if norm < 1e-6:
                out_anchor_orient = np.array(DEFAULT_QUAT_WXYZ, dtype=np.float32)
            else:
                out_anchor_orient = out_anchor_orient.copy()
                out_anchor_orient[:4] /= norm

    return {
        "obj_fts": np.stack(cleaned_fts, axis=0),
        "obj_locs": np.stack(cleaned_locs, axis=0),
        "obj_masks": np.asarray(valid_masks, dtype=bool),
        "anchor_locs": out_anchor_locs,
        "anchor_orientation": out_anchor_orient,
        "stats": stats,
    }


def sanitize_3d_tensors(
    obj_fts: torch.Tensor,
    obj_locs: torch.Tensor,
    obj_masks: torch.Tensor,
    anchor_locs: torch.Tensor,
    anchor_orientation: torch.Tensor,
    *,
    loc_whd_min: float = LOC_WHD_MIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """In-place-safe cleaning for batched LEO 3D tensors (B, N, ...)."""
    stats = {"n_bad_fts": 0, "n_bad_locs": 0, "n_bad_anchor": 0, "n_zeroed_obj": 0}

    obj_fts = obj_fts.float().clone()
    obj_locs = obj_locs.float().clone()
    obj_masks = obj_masks.bool().clone()
    anchor_locs = anchor_locs.float().clone()
    anchor_orientation = anchor_orientation.float().clone()

    for tensor, key in (
        (obj_fts, "n_bad_fts"),
        (obj_locs, "n_bad_locs"),
        (anchor_locs, "n_bad_anchor"),
        (anchor_orientation, "n_bad_anchor"),
    ):
        bad = ~torch.isfinite(tensor)
        n_bad = int(bad.sum().item())
        if n_bad:
            stats[key] += n_bad
            tensor[bad] = 0.0

    obj_fts.clamp_(-FT_CLIP, FT_CLIP)
    obj_locs.clamp_(-LOC_XYZ_CLIP, LOC_XYZ_CLIP)
    if obj_locs.shape[-1] >= 6:
        obj_locs[..., 3:6] = obj_locs[..., 3:6].clamp(min=loc_whd_min, max=LOC_WHD_MAX)
    anchor_locs.clamp_(-LOC_XYZ_CLIP, LOC_XYZ_CLIP)

    if anchor_orientation.shape[-1] >= 4:
        quat = anchor_orientation[..., :4]
        norms = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
        anchor_orientation[..., :4] = quat / norms
        bad_norm = norms.squeeze(-1) < 1e-6
        if bad_norm.any():
            stats["n_bad_anchor"] += int(bad_norm.sum().item())
            anchor_orientation[bad_norm] = torch.tensor(
                DEFAULT_QUAT_WXYZ, device=anchor_orientation.device, dtype=anchor_orientation.dtype
            )

    # Per-object validity: if still non-finite or all-zero locs with mask on, drop object.
    if obj_fts.ndim == 4:
        fts_ok = torch.isfinite(obj_fts).flatten(-2).all(dim=-1)
        locs_ok = torch.isfinite(obj_locs).all(dim=-1)
        per_obj_bad = ~fts_ok | ~locs_ok
        if per_obj_bad.any():
            stats["n_zeroed_obj"] += int(per_obj_bad.sum().item())
            obj_masks = obj_masks & ~per_obj_bad
            obj_fts[per_obj_bad] = 0.0
            obj_locs[per_obj_bad] = 0.0

    obj_locs, obj_masks, anchor_locs = _prepare_3d_geometry(
        obj_locs, obj_masks, anchor_locs, loc_whd_min=loc_whd_min
    )

    return obj_fts, obj_locs, obj_masks, anchor_locs, anchor_orientation, stats


def _prepare_3d_geometry(
    obj_locs: torch.Tensor,
    obj_masks: torch.Tensor,
    anchor_locs: torch.Tensor,
    *,
    loc_whd_min: float = LOC_WHD_MIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Avoid degenerate pairwise geometry (all centers identical -> max_dists=0 -> NaN)."""
    obj_locs = obj_locs.clone()
    obj_masks = obj_masks.clone()
    anchor_locs = anchor_locs.clone()

    # LEO anchor at origin collides with zeroed no-3d objects; lift anchor slightly.
    flat_anchor = anchor_locs.reshape(-1, anchor_locs.shape[-1])
    if flat_anchor.shape[-1] >= 3:
        near_origin = flat_anchor[:, :3].abs().sum(dim=-1) < 1e-4
        if near_origin.any():
            flat_anchor[near_origin, 2] = 1.0
        anchor_locs = flat_anchor.reshape(anchor_locs.shape)

    if obj_locs.ndim != 3:
        return obj_locs, obj_masks, anchor_locs

    bsz, num_obj, _ = obj_locs.shape
    for b in range(bsz):
        anchor_xyz = anchor_locs[b, :3] if anchor_locs[b].numel() >= 3 else torch.zeros(3, device=obj_locs.device)

        invalid = ~obj_masks[b]
        inv_idx = invalid.nonzero(as_tuple=True)[0]
        for j, idx in enumerate(inv_idx):
            obj_locs[b, idx, 0] = anchor_xyz[0] + 5.0 + float(j) * 2.0
            obj_locs[b, idx, 1] = anchor_xyz[1] + 0.3 * float(j)
            obj_locs[b, idx, 2] = anchor_xyz[2] + 0.5 * float(j)
            obj_locs[b, idx, 3:6] = loc_whd_min

        valid_idx = obj_masks[b].nonzero(as_tuple=True)[0]
        for j, idx in enumerate(valid_idx):
            xyz = obj_locs[b, idx, :3]
            if torch.linalg.norm(xyz - anchor_xyz) < 1e-3:
                obj_locs[b, idx, :3] = anchor_xyz + torch.tensor(
                    [0.2 + 0.05 * j, 0.1, 0.15], device=obj_locs.device, dtype=obj_locs.dtype
                )
            obj_locs[b, idx, 3:6] = obj_locs[b, idx, 3:6].clamp(min=loc_whd_min, max=LOC_WHD_MAX)

    return obj_locs, obj_masks, anchor_locs


def apply_leo_numeric_patches() -> None:
    """Patch LEO pairwise geom + spatial-attn to tolerate edge-case batches."""
    import einops
    import model.utils as leo_utils
    import model.transformers as leo_transformers

    if getattr(leo_utils, "_rc365_patched", False):
        return

    _orig_pairwise = leo_utils.calc_pairwise_locs

    def calc_pairwise_locs_safe(obj_centers, obj_whls, eps=1e-10, **kwargs):
        safe_eps = max(float(eps), 1e-6)
        out = _orig_pairwise(obj_centers, obj_whls, eps=safe_eps, **kwargs)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    leo_utils.calc_pairwise_locs = calc_pairwise_locs_safe  # type: ignore[assignment]

    _orig_attn = leo_transformers.MultiHeadAttentionSpatial.forward

    def spatial_attn_safe(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        pairwise_locs = torch.nan_to_num(pairwise_locs, nan=0.0, posinf=1.0, neginf=-1.0)
        try:
            return _orig_attn(self, q, k, v, pairwise_locs, key_padding_mask, txt_embeds)
        except AssertionError:
            # Degenerate softmax row: fall back to uniform attention over unmasked keys.
            import numpy as np
            import torch.nn.functional as F

            residual = q
            head = self.n_head
            qh = einops.rearrange(self.w_qs(q), "b l (head k) -> head b l k", head=head)
            kh = einops.rearrange(self.w_ks(k), "b t (head k) -> head b t k", head=head)
            vh = einops.rearrange(self.w_vs(v), "b t (head v) -> head b t v", head=head)
            attn = torch.einsum("hblk,hbtk->hblt", qh, kh) / np.sqrt(qh.shape[-1])
            if key_padding_mask is not None:
                mask = einops.repeat(key_padding_mask, "b t -> h b l t", h=head, l=qh.size(2))
                attn = attn.masked_fill(mask, -1e4)
            fused = torch.softmax(attn, dim=-1)
            fused = torch.nan_to_num(fused, nan=0.0)
            fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            output = torch.einsum("hblt,hbtv->hblv", fused, vh)
            output = einops.rearrange(output, "head b l v -> b l (head v)")
            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)
            return output, fused

    leo_transformers.MultiHeadAttentionSpatial.forward = spatial_attn_safe  # type: ignore[assignment]
    leo_utils._rc365_patched = True


def sanitize_leo_batch_3d(batch: dict[str, Any]) -> dict[str, Any]:
    """Sanitize 3D keys on a collated LEO batch dict."""
    if "obj_fts" not in batch:
        return batch
    out = dict(batch)
    fts, locs, masks, aloc, aorient, _stats = sanitize_3d_tensors(
        batch["obj_fts"],
        batch["obj_locs"],
        batch["obj_masks"],
        batch["anchor_locs"],
        batch["anchor_orientation"],
    )
    out["obj_fts"] = fts
    out["obj_locs"] = locs
    out["obj_masks"] = masks
    out["anchor_locs"] = aloc
    out["anchor_orientation"] = aorient
    return out
