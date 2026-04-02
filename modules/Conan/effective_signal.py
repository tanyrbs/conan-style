from __future__ import annotations

from typing import Any, Optional

import torch


def tensor_signal_max_abs(
    value: Any,
    *,
    mask: Optional[torch.Tensor] = None,
):
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        return None
    signal = value
    if (
        isinstance(mask, torch.Tensor)
        and mask.dim() == 3
        and signal.dim() == 3
        and tuple(mask.shape[:2]) == tuple(signal.shape[:2])
    ):
        signal = signal * mask.to(device=signal.device, dtype=signal.dtype)
    return signal.detach().abs().amax()


def tensor_has_effective_signal(
    value: Any,
    *,
    eps: float = 1e-8,
    mask: Optional[torch.Tensor] = None,
) -> bool:
    max_abs = tensor_signal_max_abs(value, mask=mask)
    if max_abs is None:
        return False
    return bool(torch.isfinite(max_abs) and (max_abs > float(eps)))


def maybe_effective_sequence(
    sequence: Any,
    *,
    eps: float = 1e-8,
):
    if not isinstance(sequence, torch.Tensor):
        return None
    if sequence.dim() != 3 or sequence.size(1) <= 0:
        return None
    if not tensor_has_effective_signal(sequence, eps=eps):
        return None
    return sequence


def maybe_effective_singleton(
    sequence: Any,
    *,
    eps: float = 1e-8,
):
    if not isinstance(sequence, torch.Tensor):
        return None
    if sequence.dim() == 2:
        sequence = sequence.unsqueeze(1)
    if sequence.dim() != 3 or sequence.size(1) <= 0:
        return None
    if sequence.size(1) != 1:
        sequence = sequence[:, :1, :]
    if not tensor_has_effective_signal(sequence, eps=eps):
        return None
    return sequence


__all__ = [
    "maybe_effective_sequence",
    "maybe_effective_singleton",
    "tensor_has_effective_signal",
    "tensor_signal_max_abs",
]
