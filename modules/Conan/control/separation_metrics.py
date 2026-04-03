from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_sequence_mask(mask, sequence):
    if not isinstance(mask, torch.Tensor) or not isinstance(sequence, torch.Tensor):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 2 or tuple(mask.shape) != tuple(sequence.shape[:2]):
        return None
    return mask.bool().to(sequence.device)


def build_sequence_weight(mask=None, *, reference=None, boundary_mask=None, voiced_weight=None, frame_weight=None):
    device = None
    dtype = torch.float32
    shape = None
    if isinstance(reference, torch.Tensor):
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        shape = reference.shape[:2]

    weight = None
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        if mask.dim() == 2:
            weight = (~mask.bool()).to(device=device or mask.device, dtype=dtype)
            shape = tuple(mask.shape)
    if weight is None and shape is not None:
        weight = torch.ones(shape, device=device, dtype=dtype)
    if weight is None:
        return None

    applied_frame_weight = frame_weight if isinstance(frame_weight, torch.Tensor) else voiced_weight
    if isinstance(applied_frame_weight, torch.Tensor):
        if applied_frame_weight.dim() == 3 and applied_frame_weight.size(-1) == 1:
            applied_frame_weight = applied_frame_weight.squeeze(-1)
        if applied_frame_weight.dim() == 2 and tuple(applied_frame_weight.shape) == tuple(weight.shape):
            weight = weight * applied_frame_weight.to(
                device=weight.device,
                dtype=weight.dtype,
            ).clamp(0.0, 1.0)

    if isinstance(boundary_mask, torch.Tensor):
        if boundary_mask.dim() == 3 and boundary_mask.size(-1) == 1:
            boundary_mask = boundary_mask.squeeze(-1)
        if boundary_mask.dim() == 2 and tuple(boundary_mask.shape) == tuple(weight.shape):
            weight = weight * (1.0 - boundary_mask.to(device=weight.device, dtype=weight.dtype).clamp(0.0, 1.0))
    return weight


def weighted_mean(value, weight=None):
    if not isinstance(value, torch.Tensor):
        return None
    if not isinstance(weight, torch.Tensor):
        return value.mean()
    if tuple(value.shape) != tuple(weight.shape):
        return None
    denom = weight.sum().clamp_min(1.0)
    return (value * weight).sum() / denom


def resolve_sample_voiced_weight(sample_uv, reference):
    if not isinstance(sample_uv, torch.Tensor) or not isinstance(reference, torch.Tensor):
        return None
    if sample_uv.dim() == 3 and sample_uv.size(-1) == 1:
        sample_uv = sample_uv.squeeze(-1)
    if sample_uv.dim() != 2 or tuple(sample_uv.shape) != tuple(reference.shape[:2]):
        return None
    return (1.0 - sample_uv.float()).clamp(0.0, 1.0).to(reference.device)


def resolve_dynamic_timbre_frame_weight(
    sample_uv,
    sample_energy,
    reference,
    *,
    mask=None,
    uv_floor=0.25,
    energy_floor=0.10,
    energy_power=0.5,
    energy_quantile=0.90,
):
    reference_energy = sequence_energy_map(reference)
    if not isinstance(reference_energy, torch.Tensor):
        return None
    device = reference_energy.device
    dtype = reference_energy.dtype
    shape = tuple(reference_energy.shape)

    support = None
    uv_floor = max(0.0, min(1.0, float(uv_floor)))
    if isinstance(sample_uv, torch.Tensor):
        if sample_uv.dim() == 3 and sample_uv.size(-1) == 1:
            sample_uv = sample_uv.squeeze(-1)
        if sample_uv.dim() == 2 and tuple(sample_uv.shape) == shape:
            voiced = (1.0 - sample_uv.float()).clamp(0.0, 1.0).to(device=device, dtype=dtype)
            support = uv_floor + (1.0 - uv_floor) * voiced

    energy_source = sequence_energy_map(sample_energy)
    if not isinstance(energy_source, torch.Tensor) or tuple(energy_source.shape) != shape:
        energy_source = reference_energy.detach()
    else:
        energy_source = energy_source.to(device=device, dtype=dtype).abs()

    sequence_mask = normalize_sequence_mask(mask, energy_source)
    if sequence_mask is not None:
        energy_source = energy_source.masked_fill(sequence_mask, 0.0)

    energy_floor = max(0.0, min(1.0, float(energy_floor)))
    energy_power = max(float(energy_power), 1.0e-4)
    energy_quantile = min(max(float(energy_quantile), 1.0e-3), 1.0)
    energy_scales = []
    for row_idx in range(energy_source.size(0)):
        row_energy = energy_source[row_idx]
        if isinstance(sequence_mask, torch.Tensor):
            row_valid = ~sequence_mask[row_idx]
            row_energy = row_energy[row_valid]
        if row_energy.numel() <= 0:
            scale = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            scale = torch.quantile(row_energy, energy_quantile).clamp_min(1.0e-4)
        energy_scales.append(scale)
    energy_scale = torch.stack(energy_scales, dim=0).unsqueeze(1).clamp_min(1.0e-4)
    normalized_energy = (energy_source / energy_scale).clamp(0.0, 1.0)
    energy_support = energy_floor + (1.0 - energy_floor) * normalized_energy.pow(energy_power)
    support = energy_support if support is None else torch.maximum(support, energy_support)
    return support.to(device=device, dtype=dtype)


def sequence_energy_map(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3:
        return value.float().abs().mean(dim=-1)
    if value.dim() == 2:
        return value.float().abs()
    return None


def sequence_energy_mean(value, *, mask=None, boundary_mask=None, voiced_weight=None, frame_weight=None):
    energy = sequence_energy_map(value)
    if not isinstance(energy, torch.Tensor):
        return None
    weight = build_sequence_weight(
        mask,
        reference=energy,
        boundary_mask=boundary_mask,
        voiced_weight=voiced_weight,
        frame_weight=frame_weight,
    )
    return weighted_mean(energy, weight)


def masked_sequence_cosine(
    a,
    b,
    *,
    mask=None,
    boundary_mask=None,
    voiced_weight=None,
    frame_weight=None,
    absolute=False,
    margin=0.0,
    eps=1e-6,
):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return {
            "cosine_map": None,
            "penalty_map": None,
            "weight": None,
            "reduced": None,
        }
    if a.dim() != 3 or b.dim() != 3 or tuple(a.shape) != tuple(b.shape):
        return {
            "cosine_map": None,
            "penalty_map": None,
            "weight": None,
            "reduced": None,
        }
    cosine_map = F.cosine_similarity(a, b, dim=-1, eps=float(eps))
    penalty_map = cosine_map.abs() if absolute else cosine_map
    margin = float(margin)
    if margin > 0.0:
        penalty_map = F.relu(penalty_map - margin)
    weight = build_sequence_weight(
        mask,
        reference=cosine_map,
        boundary_mask=boundary_mask,
        voiced_weight=voiced_weight,
        frame_weight=frame_weight,
    )
    reduced = weighted_mean(penalty_map, weight)
    return {
        "cosine_map": cosine_map,
        "penalty_map": penalty_map,
        "weight": weight,
        "reduced": reduced,
    }


__all__ = [
    "build_sequence_weight",
    "masked_sequence_cosine",
    "normalize_sequence_mask",
    "resolve_dynamic_timbre_frame_weight",
    "resolve_sample_voiced_weight",
    "sequence_energy_map",
    "sequence_energy_mean",
    "weighted_mean",
]
