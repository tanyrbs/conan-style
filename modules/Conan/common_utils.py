from __future__ import annotations

from typing import Optional

import torch

from modules.Conan.common import first_present


def expand_sequence_like(
    value: Optional[torch.Tensor],
    target_len: int,
    *,
    device=None,
    dtype=None,
    mean_fallback: bool = True,
):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 2:
        value = value.unsqueeze(1)
    if value.dim() != 3:
        return None
    if value.size(1) == target_len:
        expanded = value
    elif value.size(1) == 1:
        expanded = value.expand(-1, target_len, -1)
    elif mean_fallback and value.size(1) > 0:
        expanded = value.mean(dim=1, keepdim=True).expand(-1, target_len, -1)
    else:
        return None
    return expanded.to(
        device=device if device is not None else expanded.device,
        dtype=dtype if dtype is not None else expanded.dtype,
    )


def resolve_content_padding_mask(content, padding_idx: int, *, target=None):
    if not isinstance(content, torch.Tensor) or content.dim() != 2:
        return None
    target_device = content.device
    if target is not None:
        if isinstance(target, torch.Tensor):
            target_shape = tuple(target.shape)
            target_device = target.device
        else:
            target_shape = tuple(target)
        if tuple(content.shape) != target_shape:
            return None
    return content.eq(int(padding_idx)).to(device=target_device)


__all__ = [
    "expand_sequence_like",
    "first_present",
    "resolve_content_padding_mask",
]
