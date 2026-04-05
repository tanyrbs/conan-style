from __future__ import annotations

import math

import torch.nn as nn


DEFAULT_NEARLY_CLOSED_INIT_STD = 1.0e-3


def resolve_nearly_closed_init_std(value, *, default: float = DEFAULT_NEARLY_CLOSED_INIT_STD) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    if not math.isfinite(resolved) or resolved < 0.0:
        resolved = float(default)
    return float(resolved)


def init_nearly_closed_linear(
    linear: nn.Linear,
    *,
    bias: float = 0.0,
    weight_std: float = DEFAULT_NEARLY_CLOSED_INIT_STD,
):
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"init_nearly_closed_linear expects nn.Linear, got {type(linear).__name__}."
        )
    resolved_std = resolve_nearly_closed_init_std(weight_std)
    if resolved_std == 0.0:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.normal_(linear.weight, mean=0.0, std=resolved_std)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, float(bias))
    return linear


__all__ = [
    "DEFAULT_NEARLY_CLOSED_INIT_STD",
    "init_nearly_closed_linear",
    "resolve_nearly_closed_init_std",
]
