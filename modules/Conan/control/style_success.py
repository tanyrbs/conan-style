from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from modules.Conan.control.common import summary_vector


STYLE_SUCCESS_PAIR_WEIGHT = 0.50
STYLE_SUCCESS_RANK_WEIGHT = 1.00
STYLE_SUCCESS_RANK_TEMPERATURE = 0.20
STYLE_SUCCESS_SELF_REF_SCALE = 0.35
STYLE_SUCCESS_SELF_DERIVED_RUNTIME_SOURCES = {
    "style_trace_pooled",
    "style_trace_blended_with_reference",
}


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


def resolve_style_success_anchor(
    owner_style_summary: Any,
    slow_style_summary: Any = None,
    *,
    fast_style_summary: Any = None,
    combined_style_summary: Any = None,
):
    """
    Build the style-success anchor around the public owner-style contract.

    Prefer the realized owner-style summary (`style_decoder_residual`) when it is
    available, because that is the mainline style variable that actually reaches
    the decoder. Fall back conservatively instead of averaging multiple raw
    branch summaries together, so the lower-bound anchor does not quietly drift
    away from the public owner-style contract.
    """
    owner_style_summary = summary_vector(owner_style_summary)
    if isinstance(owner_style_summary, torch.Tensor) and owner_style_summary.dim() == 2:
        return owner_style_summary
    slow_style_summary = summary_vector(slow_style_summary)
    if isinstance(slow_style_summary, torch.Tensor) and slow_style_summary.dim() == 2:
        return slow_style_summary
    combined_style_summary = summary_vector(combined_style_summary)
    if isinstance(combined_style_summary, torch.Tensor) and combined_style_summary.dim() == 2:
        return combined_style_summary
    fast_style_summary = summary_vector(fast_style_summary)
    if isinstance(fast_style_summary, torch.Tensor) and fast_style_summary.dim() == 2:
        return fast_style_summary
    return None


def _resolve_length_vector(sample: Mapping[str, Any], key: str, *, batch_size: int, device):
    if not isinstance(sample, Mapping):
        return None
    value = sample.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    value = value.view(-1)
    if value.numel() != batch_size:
        return None
    return value.to(device=device).long().clamp_min(0)


def _resolve_rate_proxy_lengths(sample: Mapping[str, Any], *, batch_size: int, device):
    for key in ("txt_lengths", "content_lengths", "word_lengths"):
        value = _resolve_length_vector(sample, key, batch_size=batch_size, device=device)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _resolve_time_sequence(sample: Mapping[str, Any], key: str, *, batch_size: int, device):
    if not isinstance(sample, Mapping):
        return None
    value = sample.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3 and value.size(-1) == 1:
        value = value.squeeze(-1)
    if value.dim() != 2 or value.size(0) != batch_size:
        return None
    return value.to(device=device, dtype=torch.float32)


def _length_to_valid_mask(lengths: torch.Tensor, max_steps: int):
    if not isinstance(lengths, torch.Tensor) or lengths.dim() != 1:
        return None
    positions = torch.arange(max_steps, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _masked_row_mean(value: torch.Tensor, valid_mask: torch.Tensor | None):
    if not isinstance(value, torch.Tensor) or value.dim() != 2:
        return None
    value = value.float()
    if not isinstance(valid_mask, torch.Tensor) or tuple(valid_mask.shape) != tuple(value.shape):
        return value.mean(dim=-1)
    weight = valid_mask.to(device=value.device, dtype=value.dtype)
    denom = weight.sum(dim=-1).clamp_min(1.0)
    return (value * weight).sum(dim=-1) / denom


def _masked_row_std(value: torch.Tensor, valid_mask: torch.Tensor | None):
    if not isinstance(value, torch.Tensor) or value.dim() != 2:
        return None
    value = value.float()
    mean = _masked_row_mean(value, valid_mask)
    if not isinstance(mean, torch.Tensor):
        return None
    if not isinstance(valid_mask, torch.Tensor) or tuple(valid_mask.shape) != tuple(value.shape):
        return value.std(dim=-1, unbiased=False)
    weight = valid_mask.to(device=value.device, dtype=value.dtype)
    denom = weight.sum(dim=-1).clamp_min(1.0)
    centered = (value - mean.unsqueeze(-1)).pow(2) * weight
    return (centered.sum(dim=-1) / denom).clamp_min(0.0).sqrt()


def style_success_proxy_negative_mask(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    device,
    threshold: float = 1.25,
):
    if not isinstance(sample, Mapping) or batch_size <= 1:
        return None
    feature_bank = []
    mel_lengths = _resolve_length_vector(sample, "mel_lengths", batch_size=batch_size, device=device)

    energy = _resolve_time_sequence(sample, "energy", batch_size=batch_size, device=device)
    if isinstance(energy, torch.Tensor):
        energy_valid = _length_to_valid_mask(mel_lengths, energy.size(1)) if isinstance(mel_lengths, torch.Tensor) else None
        log_energy = energy.abs().clamp_min(1.0e-4).log()
        log_energy_mean = _masked_row_mean(log_energy, energy_valid)
        log_energy_std = _masked_row_std(log_energy, energy_valid)
        if isinstance(log_energy_mean, torch.Tensor):
            feature_bank.append(log_energy_mean)
        if isinstance(log_energy_std, torch.Tensor):
            feature_bank.append(log_energy_std)

    rate_proxy_lengths = _resolve_rate_proxy_lengths(sample, batch_size=batch_size, device=device)
    if isinstance(rate_proxy_lengths, torch.Tensor) and isinstance(mel_lengths, torch.Tensor):
        rate_proxy = rate_proxy_lengths.float() / mel_lengths.float().clamp_min(1.0)
        feature_bank.append(rate_proxy)

    uv = _resolve_time_sequence(sample, "uv", batch_size=batch_size, device=device)
    uv_valid = None
    if isinstance(uv, torch.Tensor):
        uv_valid = _length_to_valid_mask(mel_lengths, uv.size(1)) if isinstance(mel_lengths, torch.Tensor) else None
        voiced_ratio = _masked_row_mean((1.0 - uv).clamp(0.0, 1.0), uv_valid)
        if isinstance(voiced_ratio, torch.Tensor):
            feature_bank.append(voiced_ratio)

    f0 = _resolve_time_sequence(sample, "f0", batch_size=batch_size, device=device)
    if isinstance(f0, torch.Tensor):
        pitch_valid = _length_to_valid_mask(mel_lengths, f0.size(1)) if isinstance(mel_lengths, torch.Tensor) else None
        if isinstance(uv, torch.Tensor) and tuple(uv.shape) == tuple(f0.shape):
            voiced_valid = (uv <= 0.5)
            pitch_valid = voiced_valid if pitch_valid is None else (pitch_valid & voiced_valid)
        log_f0_std = _masked_row_std(f0, pitch_valid)
        if isinstance(log_f0_std, torch.Tensor):
            if isinstance(pitch_valid, torch.Tensor):
                voiced_count = pitch_valid.sum(dim=-1)
                log_f0_std = torch.where(
                    voiced_count > 1,
                    torch.log1p(log_f0_std.clamp_min(0.0)),
                    torch.zeros_like(log_f0_std),
                )
            else:
                log_f0_std = torch.log1p(log_f0_std.clamp_min(0.0))
            feature_bank.append(log_f0_std)

    if not feature_bank:
        return None
    features = torch.stack(feature_bank, dim=-1)
    if features.dim() != 2 or features.size(0) != batch_size or features.size(1) <= 0:
        return None
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    feature_std = features.std(dim=0, unbiased=False)
    informative = feature_std > 1.0e-6
    if informative.any():
        features = features[:, informative]
    if features.size(1) <= 0:
        return None
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-4)
    normalized = (features - feature_mean) / feature_std
    distance = torch.cdist(normalized, normalized, p=2)
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    negative_mask = (distance >= max(float(threshold), 0.0)) & (~eye)
    return negative_mask if negative_mask.any() else None


def _valid_negative_rows(mask: Any, *, min_count: int = 1):
    if not isinstance(mask, torch.Tensor) or mask.dim() != 2:
        return None
    required = max(int(min_count), 1)
    required = min(required, max(mask.size(1) - 1, 1))
    return mask.sum(dim=-1) >= required


def resolve_style_success_negative_masks(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    device,
    proxy_threshold: float = 1.25,
    proxy_min_count: int = 2,
):
    label_negative_mask = style_success_negative_mask(
        sample,
        batch_size=batch_size,
        device=device,
    )
    label_valid_rows = _valid_negative_rows(label_negative_mask, min_count=1)

    proxy_negative_mask = style_success_proxy_negative_mask(
        sample,
        batch_size=batch_size,
        device=device,
        threshold=proxy_threshold,
    )
    proxy_valid_rows = _valid_negative_rows(proxy_negative_mask, min_count=proxy_min_count)
    if isinstance(proxy_negative_mask, torch.Tensor) and isinstance(proxy_valid_rows, torch.Tensor):
        proxy_negative_mask = proxy_negative_mask & proxy_valid_rows.unsqueeze(1)
        if not proxy_negative_mask.any():
            proxy_negative_mask = None
            proxy_valid_rows = None

    resolved_negative_mask = None
    source = "none"
    if isinstance(label_negative_mask, torch.Tensor):
        resolved_negative_mask = label_negative_mask.clone()
        source = "label"
        if (
            isinstance(proxy_negative_mask, torch.Tensor)
            and isinstance(label_valid_rows, torch.Tensor)
            and isinstance(proxy_valid_rows, torch.Tensor)
        ):
            backfill_rows = (~label_valid_rows) & proxy_valid_rows
            if backfill_rows.any():
                resolved_negative_mask[backfill_rows] = proxy_negative_mask[backfill_rows]
                source = "label_plus_proxy_backfill"
    elif isinstance(proxy_negative_mask, torch.Tensor):
        resolved_negative_mask = proxy_negative_mask
        source = "proxy"

    valid_rows = _valid_negative_rows(resolved_negative_mask, min_count=1)
    if isinstance(resolved_negative_mask, torch.Tensor) and isinstance(valid_rows, torch.Tensor) and not valid_rows.any():
        resolved_negative_mask = None
        source = "none"

    return {
        "negative_mask": resolved_negative_mask,
        "label_negative_mask": label_negative_mask,
        "proxy_negative_mask": proxy_negative_mask,
        "valid_rows": valid_rows,
        "label_valid_rows": label_valid_rows,
        "proxy_valid_rows": proxy_valid_rows,
        "source": source,
    }


def resolve_style_success_target_summary(output: Mapping[str, Any]):
    if not isinstance(output, Mapping):
        return {
            "summary": None,
            "source": "none",
            "runtime_source": "",
            "global_source": "",
        }
    runtime_source = str(output.get("global_style_summary_runtime_source", "") or "").strip().lower()
    runtime_summary = summary_vector(output.get("global_style_summary_runtime"))
    if (
        isinstance(runtime_summary, torch.Tensor)
        and runtime_source
        and runtime_source != "fallback_timbre_anchor"
        and runtime_source not in STYLE_SUCCESS_SELF_DERIVED_RUNTIME_SOURCES
    ):
        return {
            "summary": runtime_summary,
            "source": "runtime_reference_derived",
            "runtime_source": runtime_source,
            "global_source": str(output.get("global_style_summary_source", "") or "").strip().lower(),
        }

    global_source = str(output.get("global_style_summary_source", "") or "").strip().lower()
    if global_source == "fallback_timbre_anchor":
        return {
            "summary": None,
            "source": "none",
            "runtime_source": runtime_source,
            "global_source": global_source,
        }
    global_summary = summary_vector(output.get("global_style_summary"))
    if isinstance(global_summary, torch.Tensor):
        return {
            "summary": global_summary,
            "source": "global_reference_derived",
            "runtime_source": runtime_source,
            "global_source": global_source,
        }
    return {
        "summary": None,
        "source": "none",
        "runtime_source": runtime_source,
        "global_source": global_source,
    }


def style_success_target_global_summary(output: Mapping[str, Any]):
    return resolve_style_success_target_summary(output).get("summary")


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
    "STYLE_SUCCESS_SELF_DERIVED_RUNTIME_SOURCES",
    "mean_optional_vectors",
    "normalized_summary_batch",
    "resolve_style_success_anchor",
    "resolve_style_success_negative_masks",
    "resolve_style_success_target_summary",
    "style_success_negative_mask",
    "style_success_proxy_negative_mask",
    "style_success_supervision_scale",
    "style_success_target_global_summary",
]
