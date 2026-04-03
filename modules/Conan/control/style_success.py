from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from modules.Conan.control.common import summary_vector


STYLE_SUCCESS_PAIR_WEIGHT = 0.50
STYLE_SUCCESS_RANK_WEIGHT = 1.00
STYLE_SUCCESS_RANK_TEMPERATURE = 0.20
STYLE_SUCCESS_SELF_REF_SCALE = 0.35


def mean_optional_vectors(*values: Any):
    total = None
    count = 0
    for value in values:
        value = summary_vector(value)
        if not isinstance(value, torch.Tensor) or value.dim() != 2:
            continue
        if total is None:
            total = value
            count = 1
            continue
        if tuple(value.shape) != tuple(total.shape):
            continue
        total = total + value
        count += 1
    if total is None or count <= 0:
        return None
    return total / float(count)


def normalized_summary_batch(value: Any):
    value = summary_vector(value)
    if not isinstance(value, torch.Tensor) or value.dim() != 2:
        return None
    return F.normalize(value.float(), dim=-1, eps=1e-6)


def style_success_target_global_summary(output: Mapping[str, Any]):
    if not isinstance(output, Mapping):
        return None
    source = str(
        output.get(
            "global_style_summary_runtime_source",
            output.get("global_style_summary_source", ""),
        )
        or ""
    ).strip().lower()
    if source == "fallback_timbre_anchor":
        return None
    return summary_vector(
        output.get("global_style_summary_runtime", output.get("global_style_summary"))
    )


def style_success_negative_mask(sample: Mapping[str, Any], *, batch_size: int, device):
    if not isinstance(sample, Mapping) or batch_size <= 1:
        return None
    negative_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    for key in ("emotion_ids", "accent_ids"):
        labels = sample.get(key)
        if not isinstance(labels, torch.Tensor) or labels.dim() != 1 or labels.numel() != batch_size:
            continue
        labels = labels.to(device=device)
        valid = labels >= 0
        if valid.any():
            valid_pairs = valid.unsqueeze(1) & valid.unsqueeze(0)
            negative_mask = negative_mask | (valid_pairs & (labels.unsqueeze(1) != labels.unsqueeze(0)))
    # Continuous weak labels are intentionally excluded here: on the shipped mainline
    # path, missing arousal/valence are canonicalized to 0.0, so they are not reliable
    # weak negatives without an explicit missing-value contract.
    negative_mask = negative_mask & (~eye)
    return negative_mask if negative_mask.any() else None


def style_success_supervision_scale(
    output: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    default_self_ref_scale: float = STYLE_SUCCESS_SELF_REF_SCALE,
):
    if bool(output.get("reference_curriculum_use_self_ref", False)):
        return max(
            float(config.get("style_success_self_ref_scale", default_self_ref_scale)),
            0.0,
        )
    return 1.0


__all__ = [
    "STYLE_SUCCESS_PAIR_WEIGHT",
    "STYLE_SUCCESS_RANK_WEIGHT",
    "STYLE_SUCCESS_RANK_TEMPERATURE",
    "STYLE_SUCCESS_SELF_REF_SCALE",
    "mean_optional_vectors",
    "normalized_summary_batch",
    "style_success_negative_mask",
    "style_success_supervision_scale",
    "style_success_target_global_summary",
]
