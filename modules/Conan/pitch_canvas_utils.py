from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from modules.Conan.common import first_present
from modules.Conan.common_utils import resolve_content_padding_mask

VALID_STYLE_TO_PITCH_RESIDUAL_MODES = (
    "auto",
    "source_aligned",
    "post_rhythm",
)


def normalize_style_to_pitch_residual_mode(mode, default: str = "auto") -> str:
    normalized_default = str(default or "auto").strip().lower() or "auto"
    normalized = str(mode or normalized_default).strip().lower() or normalized_default
    alias_map = {
        "default": normalized_default,
        "source": "source_aligned",
        "sourcealigned": "source_aligned",
        "pre_rhythm": "source_aligned",
        "content": "source_aligned",
        "post": "post_rhythm",
        "after_rhythm": "post_rhythm",
        "post_projector": "post_rhythm",
        "joint": "post_rhythm",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in VALID_STYLE_TO_PITCH_RESIDUAL_MODES:
        return normalized_default
    return normalized


def _coerce_mask_like(
    mask,
    target,
    *,
    dtype=None,
    bool_output: bool = False,
    clamp: Optional[Tuple[float, float]] = None,
):
    if not isinstance(mask, torch.Tensor) or not isinstance(target, torch.Tensor):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if target.dim() == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)
    if mask.dim() != 2 or target.dim() != 2 or tuple(mask.shape) != tuple(target.shape):
        return None
    resolved = mask.to(device=target.device)
    if bool_output:
        return resolved.bool()
    resolved = resolved.to(dtype=dtype if dtype is not None else target.dtype)
    if clamp is not None:
        lo, hi = clamp
        resolved = resolved.clamp(min=lo, max=hi)
    return resolved


def project_source_sequence_to_pitch_canvas(
    sequence,
    runtime_state: Optional[Mapping[str, Any]],
    *,
    content_padding_idx: int,
    target_shape=None,
    mode: str = "auto",
):
    meta = {
        "canvas": "source_aligned",
        "mask": None,
        "blank_mask": None,
        "valid_mask": None,
    }
    if not isinstance(sequence, torch.Tensor) or sequence.dim() != 2:
        return sequence, meta

    runtime_state = runtime_state or {}
    normalized_target_shape = tuple(target_shape.shape) if isinstance(target_shape, torch.Tensor) else (
        tuple(target_shape) if target_shape is not None else None
    )
    mode = normalize_style_to_pitch_residual_mode(mode)
    if mode == "source_aligned":
        meta["mask"] = resolve_content_padding_mask(
            runtime_state.get("content"),
            content_padding_idx,
            target=sequence,
        )
        return sequence, meta

    if normalized_target_shape is not None and tuple(sequence.shape) == normalized_target_shape and mode == "auto":
        meta["mask"] = resolve_content_padding_mask(
            runtime_state.get("content"),
            content_padding_idx,
            target=sequence,
        )
        return sequence, meta

    source_frame_index = first_present(
        runtime_state,
        "rhythm_source_frame_index",
        "source_frame_index",
        "frame_source_index",
        "retimed_source_frame_index",
    )
    unit_index = first_present(
        runtime_state,
        "rhythm_frame_unit_index",
        "rhythm_slot_unit_index",
        "frame_unit_index",
        "slot_unit_index",
    )
    frame_valid_mask = first_present(
        runtime_state,
        "rhythm_frame_valid_mask",
        "frame_valid_mask",
        "slot_mask",
    )
    blank_mask = first_present(
        runtime_state,
        "rhythm_frame_blank_mask",
        "frame_blank_mask",
        "slot_is_blank",
    )

    projected = None
    canvas = None
    if isinstance(source_frame_index, torch.Tensor) and source_frame_index.dim() == 2:
        gather_index = source_frame_index.long().clamp(min=0, max=max(0, sequence.size(1) - 1))
        projected = torch.gather(sequence, 1, gather_index)
        canvas = "post_rhythm_source_frame_index"
    elif isinstance(unit_index, torch.Tensor) and unit_index.dim() == 2:
        gather_index = unit_index.long().clamp(min=0, max=max(0, sequence.size(1) - 1))
        projected = torch.gather(sequence, 1, gather_index)
        canvas = "post_rhythm_unit_index"

    if isinstance(projected, torch.Tensor):
        valid_mask = _coerce_mask_like(
            frame_valid_mask,
            projected,
            dtype=projected.dtype,
            clamp=(0.0, 1.0),
        )
        if isinstance(valid_mask, torch.Tensor):
            projected = projected * valid_mask
        resolved_blank_mask = _coerce_mask_like(blank_mask, projected, bool_output=True)
        if isinstance(resolved_blank_mask, torch.Tensor):
            projected = projected.masked_fill(resolved_blank_mask, 0.0)
        if normalized_target_shape is None or tuple(projected.shape) == normalized_target_shape:
            invalid_mask = None
            if isinstance(valid_mask, torch.Tensor):
                invalid_mask = valid_mask <= 0
            if isinstance(resolved_blank_mask, torch.Tensor):
                invalid_mask = (
                    resolved_blank_mask
                    if invalid_mask is None
                    else (invalid_mask | resolved_blank_mask)
                )
            meta.update(
                {
                    "canvas": canvas or "post_rhythm",
                    "mask": invalid_mask,
                    "blank_mask": resolved_blank_mask,
                    "valid_mask": valid_mask,
                }
            )
            return projected, meta

    meta["mask"] = resolve_content_padding_mask(
        runtime_state.get("content"),
        content_padding_idx,
        target=sequence,
    )
    return sequence, meta


def smooth_sequence_2d(sequence, smooth_factor: float, *, valid_mask=None, kernel_size: int = 3):
    if not isinstance(sequence, torch.Tensor) or sequence.dim() != 2:
        return sequence
    smooth_factor = float(smooth_factor)
    if smooth_factor <= 0.0 or sequence.size(1) <= 1:
        return sequence
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2

    resolved_valid_mask = _coerce_mask_like(
        valid_mask,
        sequence,
        dtype=sequence.dtype,
        clamp=(0.0, 1.0),
    )
    if isinstance(resolved_valid_mask, torch.Tensor):
        pooled_value = F.avg_pool1d(
            (sequence * resolved_valid_mask).unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ).squeeze(1)
        pooled_denom = F.avg_pool1d(
            resolved_valid_mask.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ).squeeze(1).clamp_min(1.0e-6)
        pooled = pooled_value / pooled_denom
        smoothed = (1.0 - smooth_factor) * sequence + smooth_factor * pooled
        return smoothed * resolved_valid_mask + sequence * (1.0 - resolved_valid_mask)

    pooled = F.avg_pool1d(
        sequence.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    ).squeeze(1)
    return (1.0 - smooth_factor) * sequence + smooth_factor * pooled


__all__ = [
    "VALID_STYLE_TO_PITCH_RESIDUAL_MODES",
    "normalize_style_to_pitch_residual_mode",
    "project_source_sequence_to_pitch_canvas",
    "smooth_sequence_2d",
]
