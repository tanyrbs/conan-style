from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from modules.Conan.control.common import summary_vector


STYLE_SUCCESS_PAIR_WEIGHT = 0.50
STYLE_SUCCESS_RANK_WEIGHT = 1.00
STYLE_SUCCESS_RANK_TEMPERATURE = 0.20
STYLE_SUCCESS_SELF_REF_SCALE = 0.35
STYLE_SUCCESS_MEMORY_FALLBACK_SCALE = 0.60
STYLE_SUCCESS_PROXY_BACKFILL_MIN_DISTANCE = 1.0e-4
STYLE_SUCCESS_PROXY_MIN_BATCH = 4
STYLE_SUCCESS_PROXY_TARGET_BATCH = 8
STYLE_SUCCESS_LABEL_AUTHORITY_ROW_FRAC = 0.50
STYLE_SUCCESS_LABEL_RANK_SCALE = 1.00
STYLE_SUCCESS_LABEL_PLUS_PROXY_BACKFILL_SCALE = 0.75
STYLE_SUCCESS_PROXY_RANK_SCALE = 0.50
STYLE_SUCCESS_SELF_DERIVED_RUNTIME_SOURCES = {
    "style_trace_pooled",
    "style_trace_blended_with_reference",
}
STYLE_SUCCESS_DISALLOWED_TARGET_SOURCES = {
    "",
    "none",
    "missing",
    "fallback_timbre_anchor",
    *STYLE_SUCCESS_SELF_DERIVED_RUNTIME_SOURCES,
}
_BOOLLIKE_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
_BOOLLIKE_FALSE_STRINGS = {"0", "false", "no", "n", "off"}


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


def resolve_style_success_bool_flag(value: Any, *, default: bool = False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _BOOLLIKE_TRUE_STRINGS:
            return True
        if normalized in _BOOLLIKE_FALSE_STRINGS:
            return False
        return bool(default)
    return bool(value)


def _resolve_runtime_scalar_flag(value: Any, *, default: bool = False):
    if value is None:
        return bool(default)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return bool(default)
        return bool(float(value.detach().float().item()) > 0.5)
    return resolve_style_success_bool_flag(value, default=default)


def resolve_style_success_rank_scale_defaults(config: Mapping[str, Any] | None = None):
    config = config if isinstance(config, Mapping) else {}

    def _clamped_config_float(key: str, default: float):
        return min(max(float(config.get(key, default)), 0.0), 1.0)

    return {
        "label": _clamped_config_float(
            "style_success_label_rank_scale",
            STYLE_SUCCESS_LABEL_RANK_SCALE,
        ),
        "label_plus_proxy_backfill": _clamped_config_float(
            "style_success_label_plus_proxy_backfill_scale",
            STYLE_SUCCESS_LABEL_PLUS_PROXY_BACKFILL_SCALE,
        ),
        "proxy": _clamped_config_float(
            "style_success_proxy_rank_scale",
            float(config.get("style_success_proxy_only_rank_row_scale", STYLE_SUCCESS_PROXY_RANK_SCALE)),
        ),
    }


def resolve_style_success_rank_source_scale(
    source: Any,
    config: Mapping[str, Any] | None = None,
):
    defaults = resolve_style_success_rank_scale_defaults(config)
    normalized_source = str(source or "none").strip().lower()
    if normalized_source in {"", "none", "missing"}:
        return 0.0
    return float(defaults.get(normalized_source, 1.0))


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


def _style_success_proxy_feature_bank(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    device,
    use_rate_proxy: Any = False,
):
    if not isinstance(sample, Mapping) or batch_size <= 1:
        return {
            "features": None,
            "feature_names": (),
            "informative_feature_names": (),
            "feature_count": 0,
            "informative_feature_count": 0,
        }
    feature_bank = []
    feature_names = []
    mel_lengths = _resolve_length_vector(sample, "mel_lengths", batch_size=batch_size, device=device)

    energy = _resolve_time_sequence(sample, "energy", batch_size=batch_size, device=device)
    if isinstance(energy, torch.Tensor):
        energy_valid = _length_to_valid_mask(mel_lengths, energy.size(1)) if isinstance(mel_lengths, torch.Tensor) else None
        log_energy = energy.abs().clamp_min(1.0e-4).log()
        log_energy_mean = _masked_row_mean(log_energy, energy_valid)
        log_energy_std = _masked_row_std(log_energy, energy_valid)
        if isinstance(log_energy_mean, torch.Tensor):
            feature_bank.append(log_energy_mean)
            feature_names.append("log_energy_mean")
        if isinstance(log_energy_std, torch.Tensor):
            feature_bank.append(log_energy_std)
            feature_names.append("log_energy_std")

    if resolve_style_success_bool_flag(use_rate_proxy, default=False):
        rate_proxy_lengths = _resolve_rate_proxy_lengths(sample, batch_size=batch_size, device=device)
        if isinstance(rate_proxy_lengths, torch.Tensor) and isinstance(mel_lengths, torch.Tensor):
            rate_proxy = rate_proxy_lengths.float() / mel_lengths.float().clamp_min(1.0)
            feature_bank.append(rate_proxy)
            feature_names.append("rate_proxy")

    uv = _resolve_time_sequence(sample, "uv", batch_size=batch_size, device=device)
    uv_valid = None
    if isinstance(uv, torch.Tensor):
        uv_valid = _length_to_valid_mask(mel_lengths, uv.size(1)) if isinstance(mel_lengths, torch.Tensor) else None
        voiced_ratio = _masked_row_mean((1.0 - uv).clamp(0.0, 1.0), uv_valid)
        if isinstance(voiced_ratio, torch.Tensor):
            feature_bank.append(voiced_ratio)
            feature_names.append("voiced_ratio")

    f0 = _resolve_time_sequence(sample, "f0", batch_size=batch_size, device=device)
    if isinstance(f0, torch.Tensor):
        pitch_valid = _length_to_valid_mask(mel_lengths, f0.size(1)) if isinstance(mel_lengths, torch.Tensor) else None
        if isinstance(uv, torch.Tensor) and tuple(uv.shape) == tuple(f0.shape):
            voiced_valid = (uv <= 0.5)
            pitch_valid = voiced_valid if pitch_valid is None else (pitch_valid & voiced_valid)
        log_f0 = f0.clamp_min(1.0e-4).log()
        log_f0_std = _masked_row_std(log_f0, pitch_valid)
        if isinstance(log_f0_std, torch.Tensor):
            if isinstance(pitch_valid, torch.Tensor):
                voiced_count = pitch_valid.sum(dim=-1)
                log_f0_std = torch.where(
                    voiced_count > 1,
                    log_f0_std.clamp_min(0.0),
                    torch.zeros_like(log_f0_std),
                )
            feature_bank.append(log_f0_std)
            feature_names.append("log_f0_std")

    if not feature_bank:
        return {
            "features": None,
            "feature_names": tuple(feature_names),
            "informative_feature_names": (),
            "feature_count": int(len(feature_names)),
            "informative_feature_count": 0,
        }
    features = torch.stack(feature_bank, dim=-1)
    if features.dim() != 2 or features.size(0) != batch_size or features.size(1) <= 0:
        return {
            "features": None,
            "feature_names": tuple(feature_names),
            "informative_feature_names": (),
            "feature_count": int(len(feature_names)),
            "informative_feature_count": 0,
        }
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    feature_std = features.std(dim=0, unbiased=False)
    informative = feature_std > 1.0e-6
    informative_feature_names = tuple(
        name for idx, name in enumerate(feature_names) if idx < informative.numel() and bool(informative[idx].item())
    )
    if informative.any():
        features = features[:, informative]
    if features.size(1) <= 0:
        return {
            "features": None,
            "feature_names": tuple(feature_names),
            "informative_feature_names": informative_feature_names,
            "feature_count": int(len(feature_names)),
            "informative_feature_count": int(len(informative_feature_names)),
        }
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-4)
    normalized = (features - feature_mean) / feature_std
    return {
        "features": normalized,
        "feature_names": tuple(feature_names),
        "informative_feature_names": informative_feature_names,
        "feature_count": int(len(feature_names)),
        "informative_feature_count": int(len(informative_feature_names)),
    }


def style_success_proxy_negative_state(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    device,
    threshold: float = 1.25,
    min_count: int = 2,
    min_distance: float = STYLE_SUCCESS_PROXY_BACKFILL_MIN_DISTANCE,
    use_rate_proxy: Any = False,
):
    feature_state = _style_success_proxy_feature_bank(
        sample,
        batch_size=batch_size,
        device=device,
        use_rate_proxy=use_rate_proxy,
    )
    features = feature_state.get("features")
    if not isinstance(features, torch.Tensor):
        return {
            "negative_mask": None,
            "threshold_negative_mask": None,
            "backfill_negative_mask": None,
            "backfill_rows": None,
            "backfill_row_frac": 0.0,
            **feature_state,
        }
    distance = torch.cdist(features, features, p=2)
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    threshold_negative_mask = (distance >= max(float(threshold), 0.0)) & (~eye)
    backfill_negative_mask = _topk_farthest_negative_mask(
        distance,
        existing_mask=threshold_negative_mask,
        min_count=min_count,
        min_distance=min_distance,
    )
    negative_mask = threshold_negative_mask | backfill_negative_mask
    backfill_rows = backfill_negative_mask.any(dim=-1)
    return {
        "negative_mask": negative_mask if negative_mask.any() else None,
        "threshold_negative_mask": (
            threshold_negative_mask if threshold_negative_mask.any() else None
        ),
        "backfill_negative_mask": (
            backfill_negative_mask if backfill_negative_mask.any() else None
        ),
        "backfill_rows": backfill_rows if backfill_rows.any() else None,
        "backfill_row_frac": backfill_rows.float().mean(),
        "distance": distance,
        "threshold": float(max(float(threshold), 0.0)),
        **feature_state,
    }


def style_success_proxy_negative_mask(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    device,
    threshold: float = 1.25,
    min_count: int = 2,
    min_distance: float = STYLE_SUCCESS_PROXY_BACKFILL_MIN_DISTANCE,
    use_rate_proxy: Any = False,
):
    return style_success_proxy_negative_state(
        sample,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
        min_count=min_count,
        min_distance=min_distance,
        use_rate_proxy=use_rate_proxy,
    ).get("negative_mask")


def _topk_farthest_negative_mask(
    distance: Any,
    *,
    existing_mask: Any = None,
    min_count: int = 1,
    min_distance: float = STYLE_SUCCESS_PROXY_BACKFILL_MIN_DISTANCE,
):
    if not isinstance(distance, torch.Tensor) or distance.dim() != 2:
        return None
    batch_size = int(distance.size(0))
    if batch_size <= 1 or int(distance.size(1)) != batch_size:
        return None
    target_count = max(int(min_count), 0)
    target_count = min(target_count, max(batch_size - 1, 0))
    if target_count <= 0:
        return torch.zeros_like(distance, dtype=torch.bool)
    eye = torch.eye(batch_size, dtype=torch.bool, device=distance.device)
    resolved_existing_mask = (
        existing_mask.to(device=distance.device, dtype=torch.bool)
        if isinstance(existing_mask, torch.Tensor)
        and tuple(existing_mask.shape) == tuple(distance.shape)
        else torch.zeros_like(eye)
    )
    min_distance = max(float(min_distance), 0.0)
    missing_count = (
        target_count - resolved_existing_mask.sum(dim=-1, dtype=torch.int64)
    ).clamp_min(0)
    candidate_mask = (~eye) & (~resolved_existing_mask) & torch.isfinite(distance)
    if min_distance > 0.0:
        candidate_mask = candidate_mask & (distance >= min_distance)
    candidate_distance = distance.masked_fill(~candidate_mask, float("-inf"))
    topk_distance, topk_indices = torch.topk(
        candidate_distance,
        k=target_count,
        dim=-1,
        largest=True,
        sorted=False,
    )
    selection_rank = torch.arange(target_count, device=distance.device).unsqueeze(0)
    selected_mask = selection_rank < missing_count.unsqueeze(1)
    selected_mask = selected_mask & torch.isfinite(topk_distance)
    backfill_mask = torch.zeros_like(eye)
    backfill_mask.scatter_(1, topk_indices, selected_mask)
    return backfill_mask


def _valid_negative_rows(mask: Any, *, min_count: int = 1):
    if not isinstance(mask, torch.Tensor) or mask.dim() != 2:
        return None
    required = max(int(min_count), 1)
    required = min(required, max(mask.size(1) - 1, 1))
    return mask.sum(dim=-1) >= required


def _mask_pair_density(mask: Any):
    if not isinstance(mask, torch.Tensor) or mask.dim() != 2:
        return None
    return mask.float().mean()


def _mask_mean_negatives_per_row(mask: Any, valid_rows: Any = None):
    if not isinstance(mask, torch.Tensor) or mask.dim() != 2:
        return None, None
    counts = mask.sum(dim=-1).float()
    mean_all = counts.mean()
    if isinstance(valid_rows, torch.Tensor) and valid_rows.any():
        return mean_all, counts[valid_rows].mean()
    return mean_all, None


def resolve_style_success_negative_masks(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    device,
    proxy_threshold: float = 1.25,
    proxy_min_count: int = 2,
    proxy_min_batch: int = STYLE_SUCCESS_PROXY_MIN_BATCH,
    use_rate_proxy: Any = False,
):
    label_negative_mask = style_success_negative_mask(
        sample,
        batch_size=batch_size,
        device=device,
    )
    label_valid_rows = _valid_negative_rows(label_negative_mask, min_count=1)
    effective_proxy_min_count = min(
        max(int(proxy_min_count), 1),
        max(int(batch_size) - 1, 1),
    )
    label_proxy_sufficient_rows = _valid_negative_rows(
        label_negative_mask,
        min_count=effective_proxy_min_count,
    )
    label_negative_counts = (
        label_negative_mask.sum(dim=-1)
        if isinstance(label_negative_mask, torch.Tensor)
        else None
    )
    label_rows_below_proxy_min_count = (
        (~label_proxy_sufficient_rows)
        if isinstance(label_proxy_sufficient_rows, torch.Tensor)
        else None
    )

    proxy_min_batch = max(int(proxy_min_batch), 0)
    proxy_batch_gate_passed = batch_size >= max(proxy_min_batch, 1)
    proxy_disabled_reason = "active"
    proxy_state = {
        "negative_mask": None,
        "backfill_rows": None,
        "feature_count": 0,
        "informative_feature_count": 0,
        "feature_names": (),
        "informative_feature_names": (),
        "backfill_row_frac": 0.0,
    }
    proxy_negative_mask = None
    proxy_valid_rows = None
    proxy_backfill_rows = None
    if proxy_batch_gate_passed:
        proxy_state = style_success_proxy_negative_state(
            sample,
            batch_size=batch_size,
            device=device,
            threshold=proxy_threshold,
            min_count=proxy_min_count,
            use_rate_proxy=use_rate_proxy,
        )
        proxy_negative_mask = proxy_state.get("negative_mask")
        proxy_valid_rows = _valid_negative_rows(proxy_negative_mask, min_count=proxy_min_count)
        proxy_backfill_rows = proxy_state.get("backfill_rows")
        if isinstance(proxy_negative_mask, torch.Tensor) and isinstance(proxy_valid_rows, torch.Tensor):
            proxy_negative_mask = proxy_negative_mask & proxy_valid_rows.unsqueeze(1)
            if isinstance(proxy_backfill_rows, torch.Tensor):
                proxy_backfill_rows = proxy_backfill_rows & proxy_valid_rows
            if not proxy_negative_mask.any():
                proxy_negative_mask = None
                proxy_valid_rows = None
                proxy_backfill_rows = None
    else:
        proxy_disabled_reason = "batch_size_below_proxy_min_batch"

    resolved_negative_mask = None
    source = "none"
    proxy_supplemented_rows = None
    proxy_augmented_rows = None
    resolved_proxy_backfill_rows = None
    if isinstance(label_negative_mask, torch.Tensor):
        resolved_negative_mask = label_negative_mask.clone()
        source = "label"
        if (
            isinstance(proxy_negative_mask, torch.Tensor)
            and isinstance(proxy_valid_rows, torch.Tensor)
            and isinstance(label_proxy_sufficient_rows, torch.Tensor)
        ):
            backfill_rows = (~label_proxy_sufficient_rows) & proxy_valid_rows
            if backfill_rows.any():
                resolved_negative_mask[backfill_rows] = (
                    resolved_negative_mask[backfill_rows]
                    | proxy_negative_mask[backfill_rows]
                )
                proxy_supplemented_rows = backfill_rows
                proxy_augmented_rows = backfill_rows
                resolved_proxy_backfill_rows = backfill_rows
                source = "label_plus_proxy_backfill"
    elif isinstance(proxy_negative_mask, torch.Tensor):
        resolved_negative_mask = proxy_negative_mask
        source = "proxy"

    valid_rows = _valid_negative_rows(resolved_negative_mask, min_count=1)
    if isinstance(resolved_negative_mask, torch.Tensor) and isinstance(valid_rows, torch.Tensor) and not valid_rows.any():
        resolved_negative_mask = None
        source = "none"
        valid_rows = None

    negative_pair_density = _mask_pair_density(resolved_negative_mask)
    negative_row_density = valid_rows.float().mean() if isinstance(valid_rows, torch.Tensor) else None
    mean_negatives_per_row, mean_negatives_per_valid_row = _mask_mean_negatives_per_row(
        resolved_negative_mask,
        valid_rows,
    )
    label_negative_pair_density = _mask_pair_density(label_negative_mask)
    proxy_negative_pair_density = _mask_pair_density(proxy_negative_mask)
    proxy_backfill_row_frac = (
        proxy_backfill_rows.float().mean()
        if isinstance(proxy_backfill_rows, torch.Tensor)
        else 0.0
    )
    proxy_augmented_row_frac = (
        proxy_augmented_rows.float().mean()
        if isinstance(proxy_augmented_rows, torch.Tensor)
        else 0.0
    )

    return {
        "negative_mask": resolved_negative_mask,
        "label_negative_mask": label_negative_mask,
        "proxy_negative_mask": proxy_negative_mask,
        "valid_rows": valid_rows,
        "label_valid_rows": label_valid_rows,
        "label_proxy_sufficient_rows": label_proxy_sufficient_rows,
        "label_negative_counts": label_negative_counts,
        "label_rows_below_proxy_min_count": label_rows_below_proxy_min_count,
        "proxy_valid_rows": proxy_valid_rows,
        "proxy_supplemented_rows": proxy_supplemented_rows,
        "proxy_augmented_rows": proxy_augmented_rows,
        "proxy_augmented_row_frac": proxy_augmented_row_frac,
        "resolved_proxy_backfill_rows": resolved_proxy_backfill_rows,
        "source": source,
        "negative_pair_density": negative_pair_density,
        "negative_row_density": negative_row_density,
        "mean_negatives_per_row": mean_negatives_per_row,
        "mean_negatives_per_valid_row": mean_negatives_per_valid_row,
        "label_negative_pair_density": label_negative_pair_density,
        "proxy_negative_pair_density": proxy_negative_pair_density,
        "proxy_backfill_rows": proxy_backfill_rows,
        "proxy_backfill_row_frac": proxy_backfill_row_frac,
        "batch_size": int(batch_size),
        "proxy_min_batch": int(proxy_min_batch),
        "proxy_batch_gate_passed": bool(proxy_batch_gate_passed),
        "proxy_disabled_reason": proxy_disabled_reason,
        "proxy_feature_count": int(proxy_state.get("feature_count", 0) or 0),
        "proxy_informative_feature_count": int(proxy_state.get("informative_feature_count", 0) or 0),
        "proxy_feature_names": tuple(proxy_state.get("feature_names", ()) or ()),
        "proxy_informative_feature_names": tuple(
            proxy_state.get("informative_feature_names", ()) or ()
        ),
    }


def resolve_style_success_rank_support_state(
    negative_state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device=None,
):
    negative_mask = negative_state.get("negative_mask") if isinstance(negative_state, Mapping) else None
    valid_rows = negative_state.get("valid_rows") if isinstance(negative_state, Mapping) else None
    source = str(negative_state.get("source", "none")) if isinstance(negative_state, Mapping) else "none"
    label_proxy_sufficient_rows = (
        negative_state.get("label_proxy_sufficient_rows")
        if isinstance(negative_state, Mapping)
        else None
    )
    if device is None:
        if isinstance(negative_mask, torch.Tensor):
            device = negative_mask.device
        elif isinstance(valid_rows, torch.Tensor):
            device = valid_rows.device
        elif isinstance(label_proxy_sufficient_rows, torch.Tensor):
            device = label_proxy_sufficient_rows.device
    scalar_kwargs = {"dtype": torch.float32}
    if device is not None:
        scalar_kwargs["device"] = device

    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(**scalar_kwargs)
        return torch.tensor(float(value), **scalar_kwargs)

    zero = _scalar(0.0)
    one = _scalar(1.0)
    row_density = negative_state.get("negative_row_density") if isinstance(negative_state, Mapping) else None
    if not isinstance(row_density, torch.Tensor):
        row_density = zero
    else:
        row_density = row_density.detach().to(**scalar_kwargs)
    mean_negatives_per_valid_row = (
        negative_state.get("mean_negatives_per_valid_row")
        if isinstance(negative_state, Mapping)
        else None
    )
    if not isinstance(mean_negatives_per_valid_row, torch.Tensor):
        mean_negatives_per_valid_row = zero
    else:
        mean_negatives_per_valid_row = mean_negatives_per_valid_row.detach().to(**scalar_kwargs)

    min_row_frac = max(float(config.get("style_success_rank_min_negative_row_frac", 0.25)), 1.0e-6)
    min_mean_negatives = max(
        float(config.get("style_success_rank_min_mean_negatives_per_row", 2.0)),
        1.0e-6,
    )
    min_effective_support = max(
        float(config.get("style_success_rank_min_effective_support", 0.20)),
        0.0,
    )
    min_proxy_informative_features = max(
        int(config.get("style_success_proxy_min_informative_features", 2)),
        0,
    )
    proxy_informative_feature_count = int(
        negative_state.get("proxy_informative_feature_count", 0)
        if isinstance(negative_state, Mapping)
        else 0
    )
    row_scale = torch.clamp(row_density / min_row_frac, 0.0, 1.0)
    negatives_scale = torch.clamp(
        mean_negatives_per_valid_row / min_mean_negatives,
        0.0,
        1.0,
    )
    proxy_scale = one
    if source == "proxy" and min_proxy_informative_features > 0:
        proxy_scale = torch.clamp(
            _scalar(proxy_informative_feature_count) / float(min_proxy_informative_features),
            0.0,
            1.0,
        )

    batch_size = int(negative_state.get("batch_size", 0) if isinstance(negative_state, Mapping) else 0)
    if batch_size <= 0:
        if isinstance(negative_mask, torch.Tensor):
            batch_size = int(negative_mask.size(0))
        elif isinstance(valid_rows, torch.Tensor):
            batch_size = int(valid_rows.numel())
        elif isinstance(label_proxy_sufficient_rows, torch.Tensor):
            batch_size = int(label_proxy_sufficient_rows.numel())
    proxy_min_batch = max(
        int(
            negative_state.get("proxy_min_batch", STYLE_SUCCESS_PROXY_MIN_BATCH)
            if isinstance(negative_state, Mapping)
            else STYLE_SUCCESS_PROXY_MIN_BATCH
        ),
        0,
    )
    proxy_target_batch = max(
        int(config.get("style_success_proxy_target_batch", max(proxy_min_batch, STYLE_SUCCESS_PROXY_TARGET_BATCH))),
        max(proxy_min_batch, 1),
    )
    proxy_batch_scale = torch.clamp(
        _scalar(float(batch_size)) / float(proxy_target_batch),
        0.0,
        1.0,
    )

    label_proxy_sufficient_row_frac = zero
    if isinstance(label_proxy_sufficient_rows, torch.Tensor):
        label_proxy_sufficient_row_frac = label_proxy_sufficient_rows.float().mean().detach().to(**scalar_kwargs)
    elif isinstance(negative_state, Mapping):
        candidate = negative_state.get("label_proxy_sufficient_row_frac")
        if isinstance(candidate, torch.Tensor):
            label_proxy_sufficient_row_frac = candidate.detach().to(**scalar_kwargs)
        elif isinstance(candidate, (int, float)):
            label_proxy_sufficient_row_frac = _scalar(candidate)
    label_authority_frac = torch.clamp(
        label_proxy_sufficient_row_frac / row_density.clamp_min(1.0e-6),
        0.0,
        1.0,
    )
    min_label_authority_row_frac = max(
        float(config.get("style_success_label_authority_row_frac", STYLE_SUCCESS_LABEL_AUTHORITY_ROW_FRAC)),
        1.0e-6,
    )
    batch_composition_scale = one
    if source == "proxy":
        batch_composition_scale = proxy_batch_scale
    elif source == "label_plus_proxy_backfill":
        batch_composition_scale = torch.clamp(
            label_authority_frac / float(min_label_authority_row_frac),
            0.0,
            1.0,
        )
    support_scale = row_scale * negatives_scale * proxy_scale * batch_composition_scale
    has_valid_rows = bool(isinstance(valid_rows, torch.Tensor) and valid_rows.any())
    gate_passed = bool(
        has_valid_rows
        and float(row_density.item()) >= min_row_frac
        and float(mean_negatives_per_valid_row.item()) >= min_mean_negatives
        and (source != "proxy" or proxy_informative_feature_count >= min_proxy_informative_features)
        and float(support_scale.item()) >= min_effective_support
    )
    if not has_valid_rows:
        disabled_reason = "no_valid_negative_rows"
    elif float(row_density.item()) < min_row_frac:
        disabled_reason = "negative_row_density_below_floor"
    elif float(mean_negatives_per_valid_row.item()) < min_mean_negatives:
        disabled_reason = "mean_negatives_per_row_below_floor"
    elif source == "proxy" and proxy_informative_feature_count < min_proxy_informative_features:
        disabled_reason = "proxy_informative_features_below_floor"
    elif source == "proxy" and float(proxy_batch_scale.item()) < 1.0 and float(support_scale.item()) < min_effective_support:
        disabled_reason = "proxy_batch_support_below_floor"
    elif (
        source == "label_plus_proxy_backfill"
        and float(batch_composition_scale.item()) < 1.0
        and float(support_scale.item()) < min_effective_support
    ):
        disabled_reason = "proxy_backfill_support_below_floor"
    elif float(support_scale.item()) < min_effective_support:
        disabled_reason = "effective_support_below_floor"
    else:
        disabled_reason = "active"
    return {
        "support_scale": support_scale,
        "effective_support": support_scale if gate_passed else zero,
        "gate_passed": _scalar(1.0 if gate_passed else 0.0),
        "rank_term_active": _scalar(1.0 if gate_passed else 0.0),
        "disabled_reason": disabled_reason,
        "min_negative_row_frac": _scalar(min_row_frac),
        "min_mean_negatives_per_row": _scalar(min_mean_negatives),
        "min_effective_support": _scalar(min_effective_support),
        "min_proxy_informative_features": _scalar(float(min_proxy_informative_features)),
        "proxy_batch_scale": proxy_batch_scale,
        "proxy_target_batch": _scalar(float(proxy_target_batch)),
        "label_proxy_sufficient_row_frac": label_proxy_sufficient_row_frac,
        "label_authority_frac": label_authority_frac,
        "min_label_authority_row_frac": _scalar(min_label_authority_row_frac),
        "batch_composition_scale": batch_composition_scale,
    }


def _normalize_style_success_source(value: Any):
    return str(value or "").strip().lower()


def _resolve_style_success_runtime_provenance(output: Mapping[str, Any]):
    runtime_source = _normalize_style_success_source(
        output.get("global_style_summary_runtime_source", "")
    )
    global_source = _normalize_style_success_source(
        output.get("global_style_summary_source", "")
    )
    runtime_inherits_global = bool(runtime_source == "reference_summary")
    effective_runtime_source = global_source if runtime_inherits_global else runtime_source
    return {
        "runtime_source": runtime_source,
        "global_source": global_source,
        "runtime_inherits_global": runtime_inherits_global,
        "effective_runtime_source": _normalize_style_success_source(
            effective_runtime_source
        ),
    }


def _style_success_source_is_reference_derived(source: Any):
    normalized = _normalize_style_success_source(source)
    return bool(normalized) and normalized not in STYLE_SUCCESS_DISALLOWED_TARGET_SOURCES


def resolve_style_success_target_summary(output: Mapping[str, Any]):
    if not isinstance(output, Mapping):
        return {
            "summary": None,
            "source": "none",
            "runtime_source": "",
            "global_source": "",
            "effective_runtime_source": "",
            "runtime_source_inherits_global": False,
        }
    provenance = _resolve_style_success_runtime_provenance(output)
    runtime_source = provenance["runtime_source"]
    global_source = provenance["global_source"]
    effective_runtime_source = provenance["effective_runtime_source"]
    runtime_summary = summary_vector(output.get("global_style_summary_runtime"))
    if (
        isinstance(runtime_summary, torch.Tensor)
        and _style_success_source_is_reference_derived(effective_runtime_source)
    ):
        return {
            "summary": runtime_summary,
            "source": "runtime_reference_derived",
            "runtime_source": runtime_source,
            "global_source": global_source,
            "effective_runtime_source": effective_runtime_source,
            "runtime_source_inherits_global": bool(
                provenance["runtime_inherits_global"]
            ),
        }

    if global_source == "fallback_timbre_anchor":
        return {
            "summary": None,
            "source": "none",
            "runtime_source": runtime_source,
            "global_source": global_source,
            "effective_runtime_source": effective_runtime_source,
            "runtime_source_inherits_global": bool(
                provenance["runtime_inherits_global"]
            ),
        }
    global_summary = summary_vector(output.get("global_style_summary"))
    if (
        isinstance(global_summary, torch.Tensor)
        and _style_success_source_is_reference_derived(global_source)
    ):
        return {
            "summary": global_summary,
            "source": "global_reference_derived",
            "runtime_source": runtime_source,
            "global_source": global_source,
            "effective_runtime_source": effective_runtime_source,
            "runtime_source_inherits_global": bool(
                provenance["runtime_inherits_global"]
            ),
        }
    return {
        "summary": None,
        "source": "none",
        "runtime_source": runtime_source,
        "global_source": global_source,
        "effective_runtime_source": effective_runtime_source,
        "runtime_source_inherits_global": bool(provenance["runtime_inherits_global"]),
    }


def resolve_style_success_target_bank(
    output: Mapping[str, Any],
    *,
    memory_summary: Any = None,
    style_memory_summary: Any = None,
):
    if style_memory_summary is not None and memory_summary is None:
        memory_summary = style_memory_summary
    base_state = resolve_style_success_target_summary(output)
    base_summary = summary_vector(base_state.get("summary"))
    base_source = str(base_state.get("source", "none"))
    memory_summary = summary_vector(memory_summary)
    memory_fallback_used = False
    resolved_summary = base_summary if isinstance(base_summary, torch.Tensor) else None
    resolved_source = base_source if isinstance(base_summary, torch.Tensor) else "none"

    if not isinstance(base_summary, torch.Tensor):
        if isinstance(memory_summary, torch.Tensor) and memory_summary.dim() == 2:
            resolved_summary = memory_summary
            resolved_source = "style_memory_reference_fallback"
            memory_fallback_used = True

    if not isinstance(resolved_summary, torch.Tensor):
        resolved_source = "none"

    return {
        "summary": resolved_summary if isinstance(resolved_summary, torch.Tensor) else None,
        "source": resolved_source,
        "base_source": base_source,
        "runtime_source": str(base_state.get("runtime_source", "")),
        "global_source": str(base_state.get("global_source", "")),
        "effective_runtime_source": str(base_state.get("effective_runtime_source", "")),
        "runtime_source_inherits_global": bool(
            base_state.get("runtime_source_inherits_global", False)
        ),
        "memory_fallback_used": bool(memory_fallback_used),
        "memory_used": bool(memory_fallback_used),
    }


def style_success_target_global_summary(output: Mapping[str, Any]):
    return resolve_style_success_target_summary(output).get("summary")


def style_success_target_bank_summary(
    output: Mapping[str, Any],
    *,
    memory_summary: Any = None,
    style_memory_summary: Any = None,
):
    return resolve_style_success_target_bank(
        output,
        memory_summary=memory_summary,
        style_memory_summary=style_memory_summary,
    ).get("summary")


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
    scale = 1.0
    if _resolve_runtime_scalar_flag(
        output.get("reference_curriculum_use_self_ref", False),
        default=False,
    ):
        scale *= max(
            float(config.get("style_success_self_ref_scale", default_self_ref_scale)),
            0.0,
        )
    memory_fallback_used = _resolve_runtime_scalar_flag(
        output.get("style_success_target_memory_fallback_used", False),
        default=False,
    )
    target_source = _normalize_style_success_source(
        output.get("style_success_target_source", "")
    )
    if memory_fallback_used or target_source == "style_memory_reference_fallback":
        scale *= max(
            float(
                config.get(
                    "style_success_memory_fallback_scale",
                    STYLE_SUCCESS_MEMORY_FALLBACK_SCALE,
                )
            ),
            0.0,
        )
    return min(scale, 1.0)


__all__ = [
    "STYLE_SUCCESS_PAIR_WEIGHT",
    "STYLE_SUCCESS_RANK_WEIGHT",
    "STYLE_SUCCESS_RANK_TEMPERATURE",
    "STYLE_SUCCESS_SELF_REF_SCALE",
    "STYLE_SUCCESS_MEMORY_FALLBACK_SCALE",
    "STYLE_SUCCESS_SELF_DERIVED_RUNTIME_SOURCES",
    "STYLE_SUCCESS_PROXY_MIN_BATCH",
    "STYLE_SUCCESS_PROXY_TARGET_BATCH",
    "STYLE_SUCCESS_LABEL_AUTHORITY_ROW_FRAC",
    "STYLE_SUCCESS_LABEL_RANK_SCALE",
    "STYLE_SUCCESS_LABEL_PLUS_PROXY_BACKFILL_SCALE",
    "STYLE_SUCCESS_PROXY_RANK_SCALE",
    "mean_optional_vectors",
    "normalized_summary_batch",
    "resolve_style_success_anchor",
    "resolve_style_success_bool_flag",
    "resolve_style_success_negative_masks",
    "resolve_style_success_rank_scale_defaults",
    "resolve_style_success_rank_source_scale",
    "resolve_style_success_rank_support_state",
    "resolve_style_success_target_bank",
    "resolve_style_success_target_summary",
    "style_success_negative_mask",
    "style_success_proxy_negative_mask",
    "style_success_proxy_negative_state",
    "style_success_supervision_scale",
    "style_success_target_bank_summary",
    "style_success_target_global_summary",
]
