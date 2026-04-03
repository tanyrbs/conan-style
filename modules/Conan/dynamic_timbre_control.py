from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from modules.Conan.common import first_present
from modules.Conan.common_utils import expand_sequence_like


def _masked_mean(sequence: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
    if padding_mask is None:
        return sequence.mean(dim=1)
    valid = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    pooled = (sequence * valid).sum(dim=1) / denom
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


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
        return first_present(
            overrides,
            *keys,
            default=first_present(hparams, *keys, default=default),
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
    return_metadata: bool = False,
):
    metadata = {
        "transition_rate": None,
        "dense_units_detected": None,
        "dense_transition_threshold": 0.75,
    }
    if not isinstance(content_tokens, torch.Tensor) or content_tokens.dim() != 2:
        return (None, metadata) if return_metadata else None

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
        empty = boundary.unsqueeze(-1)
        return (empty, metadata) if return_metadata else empty

    valid_adjacent = nonpadding[:, 1:] & nonpadding[:, :-1]
    transition = (
        valid_adjacent
        & content_tokens[:, 1:].ne(content_tokens[:, :-1])
    )
    if valid_adjacent.numel() > 0:
        transition_rate = transition.float().sum(dim=1) / valid_adjacent.float().sum(dim=1).clamp_min(1.0)
    else:
        transition_rate = boundary.new_zeros(content_tokens.size(0))
    dense_transition_threshold = float(metadata["dense_transition_threshold"])
    dense_units_detected = transition_rate >= dense_transition_threshold
    metadata["transition_rate"] = transition_rate
    metadata["dense_units_detected"] = dense_units_detected

    # HuBERT/content-unit sequences often change almost every frame. Treating every token
    # transition as a dynamic-timbre boundary would collapse the mask to all-ones and turn
    # boundary suppression into a global attenuation. In that dense-unit regime, keep only
    # silence/edge boundaries instead of every token change.
    transition_boundary = transition.float() * (~dense_units_detected).unsqueeze(1).to(boundary.dtype)
    boundary[:, 1:] = torch.maximum(boundary[:, 1:], transition_boundary)
    boundary[:, :-1] = torch.maximum(boundary[:, :-1], transition_boundary)

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
    boundary = boundary.unsqueeze(-1)
    return (boundary, metadata) if return_metadata else boundary


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
    anchor = expand_sequence_like(global_anchor, aligned.size(1), mean_fallback=False)
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
    budget_ratio: float = 0.50,
    budget_margin: float = 0.0,
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
        "applied": over_budget.any(),
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
