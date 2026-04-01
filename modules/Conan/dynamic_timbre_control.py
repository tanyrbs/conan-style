from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


def _first_present(source: Optional[Mapping[str, Any]], *keys: str, default=None):
    if not isinstance(source, Mapping):
        return default
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _masked_mean(sequence: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
    if padding_mask is None:
        return sequence.mean(dim=1)
    valid = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    pooled = (sequence * valid).sum(dim=1) / denom
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def _expand_anchor(anchor: Optional[torch.Tensor], target_len: int):
    if not isinstance(anchor, torch.Tensor):
        return None
    if anchor.dim() == 2:
        anchor = anchor.unsqueeze(1)
    if anchor.dim() != 3:
        return None
    if anchor.size(1) == target_len:
        return anchor
    if anchor.size(1) == 1:
        return anchor.expand(-1, target_len, -1)
    return None


@dataclass(frozen=True)
class DynamicTimbreControlConfig:
    boundary_suppress_strength: float = 0.0
    boundary_radius: int = 2
    anchor_preserve_strength: float = 0.0

    def as_dict(self):
        return asdict(self)


def resolve_dynamic_timbre_control(
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    hparams: Optional[Mapping[str, Any]] = None,
) -> DynamicTimbreControlConfig:
    def _value(*keys: str, default=None):
        return _first_present(
            overrides,
            *keys,
            default=_first_present(hparams, *keys, default=default),
        )

    return DynamicTimbreControlConfig(
        boundary_suppress_strength=float(
            _value("dynamic_timbre_boundary_suppress_strength", default=0.0)
        ),
        boundary_radius=max(0, int(_value("dynamic_timbre_boundary_radius", default=2))),
        anchor_preserve_strength=float(
            _value("dynamic_timbre_anchor_preserve_strength", default=0.0)
        ),
    )


def build_dynamic_timbre_boundary_mask(
    content_tokens: Optional[torch.Tensor],
    *,
    padding_mask: Optional[torch.Tensor] = None,
    padding_idx: Optional[int] = None,
    silent_token: Optional[int] = None,
    radius: int = 2,
):
    if not isinstance(content_tokens, torch.Tensor) or content_tokens.dim() != 2:
        return None

    if padding_mask is None:
        if padding_idx is None:
            padding_mask = content_tokens.eq(0)
        else:
            padding_mask = content_tokens.eq(int(padding_idx))
    else:
        padding_mask = padding_mask.bool().to(content_tokens.device)

    nonpadding = ~padding_mask
    boundary = torch.zeros_like(content_tokens, dtype=torch.float32)
    if content_tokens.size(1) <= 0:
        return boundary.unsqueeze(-1)

    transition = (
        nonpadding[:, 1:]
        & nonpadding[:, :-1]
        & content_tokens[:, 1:].ne(content_tokens[:, :-1])
    )
    boundary[:, 1:] = torch.maximum(boundary[:, 1:], transition.float())
    boundary[:, :-1] = torch.maximum(boundary[:, :-1], transition.float())

    if silent_token is not None:
        silence = content_tokens.eq(int(silent_token)) & nonpadding
        silence_change = silence[:, 1:] ^ silence[:, :-1]
        boundary[:, 1:] = torch.maximum(boundary[:, 1:], silence_change.float())
        boundary[:, :-1] = torch.maximum(boundary[:, :-1], silence_change.float())

    valid_lengths = nonpadding.long().sum(dim=1)
    batch_size, seq_len = content_tokens.shape
    batch_idx = torch.arange(batch_size, device=content_tokens.device)
    has_tokens = valid_lengths > 0
    if has_tokens.any():
        first_index = torch.zeros_like(valid_lengths)
        first_index[has_tokens] = (~nonpadding[has_tokens]).long().argmin(dim=1)
        last_index = (valid_lengths - 1).clamp_min(0)
        boundary[batch_idx[has_tokens], first_index[has_tokens]] = 1.0
        boundary[batch_idx[has_tokens], last_index[has_tokens]] = 1.0

    boundary = boundary.masked_fill(~nonpadding, 0.0)
    radius = max(0, int(radius))
    if radius > 0:
        pooled = F.max_pool1d(
            boundary.unsqueeze(1),
            kernel_size=radius * 2 + 1,
            stride=1,
            padding=radius,
        )
        boundary = pooled.squeeze(1)
    return boundary.unsqueeze(-1)


def recenter_dynamic_timbre_to_anchor(
    aligned: Optional[torch.Tensor],
    *,
    global_anchor: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    preserve_strength: float = 0.0,
):
    if not isinstance(aligned, torch.Tensor):
        return aligned, None
    preserve_strength = float(preserve_strength)
    if preserve_strength <= 0.0:
        return aligned, None
    anchor = _expand_anchor(global_anchor, aligned.size(1))
    if anchor is None:
        return aligned, None
    pooled = _masked_mean(aligned, padding_mask)
    anchor_summary = anchor[:, 0, :]
    shift = (pooled - anchor_summary).unsqueeze(1)
    controlled = aligned - preserve_strength * shift
    if padding_mask is not None:
        controlled = controlled.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    return controlled, shift


def apply_boundary_suppression_to_gate(
    gate: Optional[torch.Tensor],
    *,
    boundary_mask: Optional[torch.Tensor] = None,
    suppress_strength: float = 0.0,
):
    if not isinstance(gate, torch.Tensor):
        return gate, None
    suppress_strength = float(suppress_strength)
    if boundary_mask is None or suppress_strength <= 0.0:
        return gate, None
    boundary_scale = (1.0 - suppress_strength * boundary_mask.to(gate.dtype)).clamp(0.0, 1.0)
    return gate * boundary_scale, boundary_scale


def apply_runtime_budget_to_dynamic_timbre(
    aligned: Optional[torch.Tensor],
    *,
    style_residual: Optional[torch.Tensor] = None,
    slow_style_residual: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    budget_ratio: float = 0.75,
    budget_margin: float = 0.02,
    slow_style_weight: float = 1.0,
):
    if not isinstance(aligned, torch.Tensor):
        return aligned, {"applied": False, "skip_reason": "missing_aligned"}

    style_energy = None
    if isinstance(style_residual, torch.Tensor):
        style_energy = style_residual.detach().abs().mean(dim=-1)
    if isinstance(slow_style_residual, torch.Tensor):
        slow_energy = slow_style_residual.detach().abs().mean(dim=-1)
        style_energy = slow_style_weight * slow_energy if style_energy is None else (style_energy + slow_style_weight * slow_energy)
    if style_energy is None:
        return aligned, {"applied": False, "skip_reason": "style_owner_missing"}
    if bool(style_energy.detach().abs().max().item() <= 1e-8):
        return aligned, {"applied": False, "skip_reason": "style_owner_inactive"}

    timbre_energy = aligned.abs().mean(dim=-1)
    allowed_energy = float(budget_ratio) * style_energy + float(budget_margin)
    denom = timbre_energy.clamp_min(1e-6)
    budget_scale = torch.minimum(torch.ones_like(timbre_energy), allowed_energy / denom)
    over_budget = timbre_energy > allowed_energy

    if isinstance(padding_mask, torch.Tensor):
        if padding_mask.dim() == 3 and padding_mask.size(-1) == 1:
            padding_mask = padding_mask.squeeze(-1)
        if padding_mask.dim() == 2 and tuple(padding_mask.shape) == tuple(timbre_energy.shape):
            valid = (~padding_mask.bool()).to(timbre_energy.dtype)
            budget_scale = budget_scale * valid + (1.0 - valid)
            over_budget = over_budget & valid.bool()

    controlled = aligned * budget_scale.unsqueeze(-1)
    if isinstance(padding_mask, torch.Tensor):
        if padding_mask.dim() == 3 and padding_mask.size(-1) == 1:
            padding_mask = padding_mask.squeeze(-1)
        if padding_mask.dim() == 2 and tuple(padding_mask.shape) == tuple(timbre_energy.shape):
            controlled = controlled.masked_fill(padding_mask.unsqueeze(-1).bool(), 0.0)

    metadata = {
        "applied": bool(over_budget.any().item()),
        "skip_reason": None,
        "budget_scale": budget_scale,
        "timbre_energy": timbre_energy,
        "style_energy": style_energy,
        "allowed_energy": allowed_energy,
        "over_budget_mask": over_budget,
        "active_fraction": over_budget.float().mean(),
    }
    return controlled, metadata


__all__ = [
    "DynamicTimbreControlConfig",
    "apply_boundary_suppression_to_gate",
    "apply_runtime_budget_to_dynamic_timbre",
    "build_dynamic_timbre_boundary_mask",
    "recenter_dynamic_timbre_to_anchor",
    "resolve_dynamic_timbre_control",
]
