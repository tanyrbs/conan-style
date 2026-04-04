from __future__ import annotations

from typing import Any

from torch.nn.utils import remove_weight_norm as _legacy_remove_weight_norm
from torch.nn.utils import weight_norm as _legacy_weight_norm

try:
    from torch.nn.utils import parametrize as _parametrize
    from torch.nn.utils.parametrizations import weight_norm as _param_weight_norm
except Exception:  # pragma: no cover - older torch variants
    _parametrize = None
    _param_weight_norm = None


def apply_weight_norm(module: Any, name: str = "weight", dim: int = 0):
    """Apply weight norm without depending on the deprecated legacy API.

    Prefer ``torch.nn.utils.parametrizations.weight_norm`` when available, and
    fall back to the legacy helper only on older PyTorch releases.
    """
    if _param_weight_norm is not None:
        return _param_weight_norm(module, name=name, dim=dim)
    return _legacy_weight_norm(module, name=name, dim=dim)


# Keep a familiar alias for modules that expect a ``weight_norm(...)`` helper.
weight_norm = apply_weight_norm


def get_weight_param(module: Any, name: str = "weight"):
    """Return the underlying weight parameter (handles parametrized weight_norm)."""
    if _parametrize is not None:
        try:
            if _parametrize.is_parametrized(module, name):
                return getattr(module.parametrizations, name).original
        except (TypeError, ValueError, AttributeError):
            pass
    return getattr(module, name, None)


def remove_weight_norm_compat(module: Any, name: str = "weight"):
    """Remove weight norm regardless of whether legacy or parametrized API was used."""
    if _parametrize is not None:
        try:
            if _parametrize.is_parametrized(module, name):
                _parametrize.remove_parametrizations(module, name, leave_parametrized=True)
                return module
        except (TypeError, ValueError, AttributeError):
            pass
    try:
        _legacy_remove_weight_norm(module, name=name)
    except (TypeError, ValueError, AttributeError):
        pass
    return module


remove_weight_norm = remove_weight_norm_compat


__all__ = [
    "apply_weight_norm",
    "weight_norm",
    "get_weight_param",
    "remove_weight_norm",
    "remove_weight_norm_compat",
]
