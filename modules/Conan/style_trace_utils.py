from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


def _is_style_trace(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() == 3


def _normalize_trace_mask(mask: Any, sequence: Optional[torch.Tensor]):
    if not isinstance(mask, torch.Tensor) or not _is_style_trace(sequence):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 2 or tuple(mask.shape) != tuple(sequence.shape[:2]):
        return None
    return mask.bool().to(device=sequence.device)


def combine_style_traces(
    fast_trace: Optional[torch.Tensor],
    slow_trace: Optional[torch.Tensor],
    *,
    fast_scale: float = 1.0,
    slow_scale: float = 1.0,
):
    has_fast = _is_style_trace(fast_trace)
    has_slow = _is_style_trace(slow_trace)
    if has_fast and has_slow and tuple(fast_trace.shape) == tuple(slow_trace.shape):
        return fast_trace * float(fast_scale) + slow_trace * float(slow_scale)
    if has_fast:
        return fast_trace * float(fast_scale)
    if has_slow:
        return slow_trace * float(slow_scale)
    return None


def resolve_combined_style_trace(
    source: Optional[Mapping[str, Any]],
    *,
    fast_scale: float = 1.0,
    slow_scale: float = 1.0,
    fast_key: str = "style_trace",
    slow_key: str = "slow_style_trace",
    fast_mask_key: str = "style_trace_mask",
    slow_mask_key: str = "slow_style_trace_mask",
):
    if not isinstance(source, Mapping):
        return None, None
    fast_trace = source.get(fast_key)
    slow_trace = source.get(slow_key)
    combined = combine_style_traces(
        fast_trace,
        slow_trace,
        fast_scale=fast_scale,
        slow_scale=slow_scale,
    )
    if not _is_style_trace(combined):
        return None, None
    fast_mask = _normalize_trace_mask(source.get(fast_mask_key), fast_trace if _is_style_trace(fast_trace) else combined)
    if isinstance(fast_mask, torch.Tensor) and tuple(fast_mask.shape) == tuple(combined.shape[:2]):
        return combined, fast_mask
    slow_mask = _normalize_trace_mask(source.get(slow_mask_key), slow_trace if _is_style_trace(slow_trace) else combined)
    if isinstance(slow_mask, torch.Tensor) and tuple(slow_mask.shape) == tuple(combined.shape[:2]):
        return combined, slow_mask
    return combined, None


__all__ = [
    "combine_style_traces",
    "resolve_combined_style_trace",
]
