import torch
import torch.nn.functional as F

from modules.Conan.common import first_present
from modules.Conan.control.budget_support import (
    resolve_dynamic_timbre_budget_support_weight as _resolve_dynamic_timbre_budget_support_weight,
)
from modules.Conan.control.common import summary_vector
from modules.Conan.control.separation_metrics import (
    build_sequence_weight,
    masked_sequence_cosine,
    sequence_energy_mean,
    weighted_mean,
)
from modules.Conan.control.style_success import (
    normalized_summary_batch,
    resolve_style_success_anchor,
    resolve_style_success_bool_flag,
    resolve_style_success_negative_masks,
    resolve_style_success_rank_support_state,
    resolve_style_success_target_bank,
    style_success_supervision_scale,
)
from modules.Conan.dynamic_timbre_control import resolve_stage_dynamic_timbre_budget_terms
from modules.Conan.style_trace_utils import resolve_combined_style_trace
from tasks.Conan.control_schedule import MAINLINE_MINIMAL_CONTROL_LAMBDAS


TRACKED_LAMBDA_KEYS = tuple(MAINLINE_MINIMAL_CONTROL_LAMBDAS)
OPTIONAL_TRACKED_LAMBDA_KEYS = (
    "lambda_style_timbre_runtime_overlap",
)


def _normalize_mask(mask, sequence):
    if not isinstance(mask, torch.Tensor) or not isinstance(sequence, torch.Tensor):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 2 or tuple(mask.shape) != tuple(sequence.shape[:2]):
        return None
    return mask.bool().to(sequence.device)


def _masked_sequence_mean(sequence, mask=None):
    if not isinstance(sequence, torch.Tensor) or sequence.dim() != 3:
        return None
    if mask is None:
        return sequence.mean(dim=1)
    mask = _normalize_mask(mask, sequence)
    if mask is None:
        return None
    valid = (~mask).unsqueeze(-1).to(sequence.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (sequence * valid).sum(dim=1) / denom


def _present(mapping, *keys, default=None):
    return first_present(mapping, *keys, default=default)


def _mask_valid_fraction(mask, sequence):
    mask = _normalize_mask(mask, sequence)
    if mask is None:
        return None
    return (~mask).float().mean()


def _resolve_style_owner_sequence(output):
    style_owner = output.get("style_decoder_residual") if isinstance(output, dict) else None
    style_owner_mask = (
        _present(output, "style_decoder_residual_mask", "style_trace_mask")
        if isinstance(output, dict)
        else None
    )
    if isinstance(style_owner, torch.Tensor):
        return style_owner, style_owner_mask
    return resolve_combined_style_trace(output)


def _label_repeat_fraction(labels):
    if not isinstance(labels, torch.Tensor):
        return None, None, None
    labels = labels.view(-1)
    valid = labels >= 0
    valid_labels = labels[valid]
    if valid_labels.numel() <= 0:
        return None, None, None
    unique, counts = torch.unique(valid_labels, return_counts=True)
    repeated = counts[counts > 1].sum().float()
    valid_count = float(valid_labels.numel())
    repeat_fraction = repeated / max(valid_count, 1.0)
    unique_ratio = unique.numel() / max(valid_count, 1.0)
    valid_fraction = valid.float().mean()
    return repeat_fraction, unique_ratio, valid_fraction


def _safe_mean_std(value):
    if not isinstance(value, torch.Tensor):
        return None, None
    value = value.float()
    return value.mean(), value.std(unbiased=False)


def _weighted_mean_std(value, weight=None):
    if not isinstance(value, torch.Tensor):
        return None, None
    value = value.float()
    mean = weighted_mean(value, weight)
    if mean is None:
        return None, None
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != tuple(value.shape):
        return mean, value.std(unbiased=False)
    centered = (value - mean) ** 2
    var = weighted_mean(centered, weight)
    if var is None:
        return mean, None
    return mean, var.clamp_min(0.0).sqrt()


def _categorical_indicator(value, expected, *, device=None):
    return torch.tensor(
        1.0 if str(value) == str(expected) else 0.0,
        dtype=torch.float32,
        device=device,
    )


def _safe_scalar_ratio(numerator, denominator, *, eps=1e-8):
    if not isinstance(numerator, torch.Tensor) or not isinstance(denominator, torch.Tensor):
        return None
    denominator = denominator.float()
    numerator = numerator.float()
    valid = denominator.abs() > float(eps)
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    ratio = numerator / safe_denominator
    return torch.where(valid, ratio, torch.zeros_like(ratio))


def _cosine_mean(a, b):
    a = summary_vector(a)
    b = summary_vector(b)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    if a.dim() != 2 or b.dim() != 2 or tuple(a.shape) != tuple(b.shape):
        return None
    return F.cosine_similarity(a, b, dim=-1, eps=1e-6).mean()


def _gate_statistics(gate, mask=None, target=0.6):
    if not isinstance(gate, torch.Tensor):
        return {}
    if gate.dim() == 3 and gate.size(-1) == 1:
        gate = gate.squeeze(-1)
    if gate.dim() != 2:
        return {}

    if mask is not None:
        mask = _normalize_mask(mask, gate.unsqueeze(-1))
    if mask is None:
        valid_gate = gate.reshape(-1)
    else:
        valid_gate = gate[(~mask)]
    if valid_gate.numel() <= 0:
        return {}
    return {
        "diag_dynamic_timbre_gate_mean": valid_gate.mean(),
        "diag_dynamic_timbre_gate_std": valid_gate.std(unbiased=False),
        "diag_dynamic_timbre_gate_high_frac": (valid_gate >= float(target)).float().mean(),
    }


def _simple_sequence_statistics(value, mask=None, *, prefix="diag_value"):
    if not isinstance(value, torch.Tensor):
        return {}
    if value.dim() == 3 and value.size(-1) == 1:
        value = value.squeeze(-1)
    if value.dim() != 2:
        return {}
    if mask is not None:
        mask = _normalize_mask(mask, value.unsqueeze(-1))
    if mask is None:
        valid = value.reshape(-1)
    else:
        valid = value[(~mask)]
    if valid.numel() <= 0:
        return {}
    return {
        f"{prefix}_mean": valid.mean(),
        f"{prefix}_std": valid.std(unbiased=False),
    }


def _summary_like(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3:
        return value.mean(dim=1)
    if value.dim() == 2:
        return value
    return None


def _flatten_tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    return value.float().reshape(-1)


def _delta_norm(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3:
        return value.float().norm(dim=-1).mean()
    return value.float().abs().mean()


def _sequence_abs_mean(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3:
        return value.abs().mean(dim=-1)
    if value.dim() == 2:
        return value.abs()
    return None


def _sum_optional_maps(*values):
    present = [value for value in values if isinstance(value, torch.Tensor) and value.dim() == 2]
    if not present:
        return None
    total = present[0]
    for value in present[1:]:
        if tuple(value.shape) != tuple(total.shape):
            continue
        total = total + value
    return total


def _sum_optional_sequences(*values):
    present = [value for value in values if isinstance(value, torch.Tensor)]
    if not present:
        return None
    total = present[0]
    for value in present[1:]:
        if tuple(value.shape) != tuple(total.shape):
            continue
        total = total + value
    return total


def _stage_energy_map(stage_meta, *, branch_keys=(), combine_before_energy=False):
    if not isinstance(stage_meta, dict):
        return None
    if bool(combine_before_energy):
        combined = _sum_optional_sequences(*[stage_meta.get(key) for key in branch_keys])
        combined_energy = _sequence_abs_mean(combined)
        if isinstance(combined_energy, torch.Tensor):
            return combined_energy
    return _sum_optional_maps(
        *[_sequence_abs_mean(stage_meta.get(key)) for key in branch_keys]
    )


def _adapter_stage_energy_map(stage_outputs, *, stage_names=None, branch_keys=(), combine_before_energy=False):
    if not isinstance(stage_outputs, dict):
        return None
    if stage_names is not None:
        stage_names = {str(name).lower() for name in stage_names}
    total = None
    for stage_name, stage_meta in stage_outputs.items():
        if stage_names is not None and str(stage_name).lower() not in stage_names:
            continue
        branch_total = _stage_energy_map(
            stage_meta,
            branch_keys=branch_keys,
            combine_before_energy=combine_before_energy,
        )
        if isinstance(branch_total, torch.Tensor):
            total = branch_total if total is None else _sum_optional_maps(total, branch_total)
    return total


def _sum_optional_delta(*values):
    present = [value for value in values if isinstance(value, torch.Tensor)]
    if not present:
        return None
    total = present[0]
    for value in present[1:]:
        total = total + value
    return total


def _branch_statistics(branch_ctx, branch_gate, prefix, branch_delta=None):
    if not isinstance(branch_ctx, torch.Tensor) or not isinstance(branch_gate, torch.Tensor):
        return {}
    gate_flat = _flatten_tensor(branch_gate)
    if gate_flat is None or gate_flat.numel() <= 0:
        return {}
    branch_value = branch_delta if isinstance(branch_delta, torch.Tensor) else branch_ctx
    if not isinstance(branch_delta, torch.Tensor):
        if branch_gate.dim() == branch_ctx.dim() - 1:
            branch_value = branch_ctx * branch_gate.unsqueeze(-1)
        elif branch_gate.dim() == branch_ctx.dim():
            branch_value = branch_ctx * branch_gate
    branch_norm = branch_value.float().norm(dim=-1).reshape(-1)
    stats = {
        f"{prefix}_gate_mean": gate_flat.mean(),
        f"{prefix}_gate_std": gate_flat.std(unbiased=False),
    }
    if branch_norm.numel() > 0:
        stats[f"{prefix}_residual_norm"] = branch_norm.mean()
    return stats


def _decoder_style_adapter_statistics(stage_outputs, dynamic_timbre_residual=None, style_owner_residual=None):
    if not isinstance(stage_outputs, dict):
        return {}
    stats = {}
    global_timbre_gate_values = []
    global_timbre_gate_stds = []
    global_timbre_residual_values = []
    global_gate_values = []
    global_gate_stds = []
    slow_style_gate_values = []
    slow_style_gate_stds = []
    fast_style_gate_values = []
    fast_style_gate_stds = []
    fast_style_residual_values = []
    dynamic_timbre_gate_values = []
    dynamic_timbre_gate_stds = []
    dynamic_timbre_residual_values = []
    global_residual_values = []
    slow_style_residual_values = []
    global_style_skip_flags = []
    global_style_fallback_skip_flags = []
    global_style_applied_as_fallback_flags = []
    late_style_owner_present_flags = []
    late_style_delta = None
    late_timbre_delta = None
    late_anchor_delta = None
    for stage_meta in stage_outputs.values():
        if not isinstance(stage_meta, dict):
            continue
        stage_name = str(stage_meta.get("stage_name", "")).lower()
        if "global_style_skipped_due_to_local_owner" in stage_meta:
            global_style_skip_flags.append(
                1.0 if bool(stage_meta.get("global_style_skipped_due_to_local_owner", False)) else 0.0
            )
        if "global_style_skipped_due_to_fallback" in stage_meta:
            global_style_fallback_skip_flags.append(
                1.0 if bool(stage_meta.get("global_style_skipped_due_to_fallback", False)) else 0.0
            )
        if "global_style_applied_as_fallback" in stage_meta:
            global_style_applied_as_fallback_flags.append(
                1.0 if bool(stage_meta.get("global_style_applied_as_fallback", False)) else 0.0
            )
        if stage_name == "late" and "late_style_owner_present" in stage_meta:
            late_style_owner_present_flags.append(
                1.0 if bool(stage_meta.get("late_style_owner_present", False)) else 0.0
            )
        branch_stats = _branch_statistics(
            stage_meta.get("global_timbre_ctx"),
            stage_meta.get("global_timbre_gate"),
            "diag_decoder_global_timbre",
            branch_delta=stage_meta.get("global_timbre_delta"),
        )
        if "diag_decoder_global_timbre_gate_mean" in branch_stats:
            global_timbre_gate_values.append(branch_stats["diag_decoder_global_timbre_gate_mean"])
        if "diag_decoder_global_timbre_gate_std" in branch_stats:
            global_timbre_gate_stds.append(branch_stats["diag_decoder_global_timbre_gate_std"])
        if "diag_decoder_global_timbre_residual_norm" in branch_stats:
            global_timbre_residual_values.append(branch_stats["diag_decoder_global_timbre_residual_norm"])

        branch_stats = _branch_statistics(
            stage_meta.get("global_style_ctx"),
            stage_meta.get("global_style_gate"),
            "diag_decoder_global_style",
            branch_delta=stage_meta.get("global_style_delta"),
        )
        if "diag_decoder_global_style_gate_mean" in branch_stats:
            global_gate_values.append(branch_stats["diag_decoder_global_style_gate_mean"])
        if "diag_decoder_global_style_gate_std" in branch_stats:
            global_gate_stds.append(branch_stats["diag_decoder_global_style_gate_std"])
        if "diag_decoder_global_style_residual_norm" in branch_stats:
            global_residual_values.append(branch_stats["diag_decoder_global_style_residual_norm"])

        branch_stats = _branch_statistics(
            stage_meta.get("slow_style_ctx"),
            stage_meta.get("slow_style_gate"),
            "diag_decoder_slow_style",
            branch_delta=stage_meta.get("slow_style_delta"),
        )
        if "diag_decoder_slow_style_gate_mean" in branch_stats:
            slow_style_gate_values.append(branch_stats["diag_decoder_slow_style_gate_mean"])
        if "diag_decoder_slow_style_gate_std" in branch_stats:
            slow_style_gate_stds.append(branch_stats["diag_decoder_slow_style_gate_std"])
        if "diag_decoder_slow_style_residual_norm" in branch_stats:
            slow_style_residual_values.append(branch_stats["diag_decoder_slow_style_residual_norm"])

        branch_stats = _branch_statistics(
            stage_meta.get("style_trace_ctx"),
            stage_meta.get("style_trace_gate"),
            "diag_decoder_fast_style",
            branch_delta=stage_meta.get("style_trace_delta"),
        )
        if "diag_decoder_fast_style_gate_mean" in branch_stats:
            fast_style_gate_values.append(branch_stats["diag_decoder_fast_style_gate_mean"])
        if "diag_decoder_fast_style_gate_std" in branch_stats:
            fast_style_gate_stds.append(branch_stats["diag_decoder_fast_style_gate_std"])
        if "diag_decoder_fast_style_residual_norm" in branch_stats:
            fast_style_residual_values.append(branch_stats["diag_decoder_fast_style_residual_norm"])

        branch_stats = _branch_statistics(
            stage_meta.get("dynamic_timbre_ctx"),
            stage_meta.get("dynamic_timbre_gate"),
            "diag_decoder_dynamic_timbre",
            branch_delta=stage_meta.get("dynamic_timbre_delta"),
        )
        if "diag_decoder_dynamic_timbre_gate_mean" in branch_stats:
            dynamic_timbre_gate_values.append(branch_stats["diag_decoder_dynamic_timbre_gate_mean"])
        if "diag_decoder_dynamic_timbre_gate_std" in branch_stats:
            dynamic_timbre_gate_stds.append(branch_stats["diag_decoder_dynamic_timbre_gate_std"])
        if "diag_decoder_dynamic_timbre_residual_norm" in branch_stats:
            dynamic_timbre_residual_values.append(branch_stats["diag_decoder_dynamic_timbre_residual_norm"])
        if stage_name == "late":
            late_style_delta = _sum_optional_delta(
                late_style_delta,
                stage_meta.get("global_style_delta"),
                stage_meta.get("style_trace_delta"),
            )
            late_timbre_delta = stage_meta.get("dynamic_timbre_delta")
            late_anchor_delta = stage_meta.get("global_timbre_delta")

    if global_timbre_gate_values:
        stats["diag_decoder_global_timbre_gate_mean"] = torch.stack(global_timbre_gate_values).mean()
    if global_timbre_gate_stds:
        stats["diag_decoder_global_timbre_gate_std"] = torch.stack(global_timbre_gate_stds).mean()
    if global_timbre_residual_values:
        stats["diag_decoder_global_timbre_residual_norm"] = torch.stack(global_timbre_residual_values).mean()
    if global_gate_values:
        stats["diag_decoder_global_style_gate_mean"] = torch.stack(global_gate_values).mean()
    if global_gate_stds:
        stats["diag_decoder_global_style_gate_std"] = torch.stack(global_gate_stds).mean()
    if slow_style_gate_values:
        stats["diag_decoder_slow_style_gate_mean"] = torch.stack(slow_style_gate_values).mean()
    if slow_style_gate_stds:
        stats["diag_decoder_slow_style_gate_std"] = torch.stack(slow_style_gate_stds).mean()
    if fast_style_gate_values:
        stats["diag_decoder_fast_style_gate_mean"] = torch.stack(fast_style_gate_values).mean()
    if fast_style_gate_stds:
        stats["diag_decoder_fast_style_gate_std"] = torch.stack(fast_style_gate_stds).mean()
    if dynamic_timbre_gate_values:
        stats["diag_decoder_dynamic_timbre_gate_mean"] = torch.stack(dynamic_timbre_gate_values).mean()
    if dynamic_timbre_gate_stds:
        stats["diag_decoder_dynamic_timbre_gate_std"] = torch.stack(dynamic_timbre_gate_stds).mean()
    if global_residual_values:
        stats["diag_decoder_global_style_residual_norm"] = torch.stack(global_residual_values).mean()
    if slow_style_residual_values:
        stats["diag_decoder_slow_style_residual_norm"] = torch.stack(slow_style_residual_values).mean()
    if fast_style_residual_values:
        stats["diag_decoder_fast_style_residual_norm"] = torch.stack(fast_style_residual_values).mean()
    if dynamic_timbre_residual_values:
        stats["diag_decoder_dynamic_timbre_residual_norm"] = torch.stack(dynamic_timbre_residual_values).mean()
    if global_style_skip_flags:
        stats["diag_decoder_global_style_skipped_due_to_local_owner"] = torch.tensor(
            global_style_skip_flags,
            dtype=torch.float32,
        ).mean()
    if global_style_fallback_skip_flags:
        stats["diag_decoder_global_style_skipped_due_to_fallback"] = torch.tensor(
            global_style_fallback_skip_flags,
            dtype=torch.float32,
        ).mean()
    if global_style_applied_as_fallback_flags:
        stats["diag_decoder_global_style_applied_as_fallback"] = torch.tensor(
            global_style_applied_as_fallback_flags,
            dtype=torch.float32,
        ).mean()
    if late_style_owner_present_flags:
        stats["diag_decoder_late_style_owner_present"] = torch.tensor(
            late_style_owner_present_flags,
            dtype=torch.float32,
        ).mean()
    style_total_residual = None
    if isinstance(style_owner_residual, torch.Tensor):
        style_total_residual, _ = _safe_mean_std(style_owner_residual.norm(dim=-1))
    if style_total_residual is None:
        if "diag_decoder_global_style_residual_norm" in stats and "diag_decoder_slow_style_residual_norm" in stats:
            style_total_residual = (
                stats["diag_decoder_global_style_residual_norm"] + stats["diag_decoder_slow_style_residual_norm"]
            )
        elif "diag_decoder_global_style_residual_norm" in stats:
            style_total_residual = stats["diag_decoder_global_style_residual_norm"]
        elif "diag_decoder_slow_style_residual_norm" in stats:
            style_total_residual = stats["diag_decoder_slow_style_residual_norm"]
    if style_total_residual is not None:
        stats["diag_decoder_style_total_residual_norm"] = style_total_residual
    dynamic_norm, _ = _safe_mean_std(
        dynamic_timbre_residual.norm(dim=-1) if isinstance(dynamic_timbre_residual, torch.Tensor) else None
    )
    if (
        "diag_decoder_style_total_residual_norm" in stats
        and dynamic_norm is not None
    ):
        ratio = _safe_scalar_ratio(
            stats["diag_decoder_style_total_residual_norm"],
            dynamic_norm,
        )
        if ratio is not None:
            stats["diag_style_to_timbre_residual_ratio"] = ratio
    if (
        "diag_decoder_fast_style_residual_norm" in stats
        and dynamic_norm is not None
    ):
        ratio = _safe_scalar_ratio(
            stats["diag_decoder_fast_style_residual_norm"],
            dynamic_norm,
        )
        if ratio is not None:
            stats["diag_fast_style_to_timbre_ratio"] = ratio
    if (
        "diag_decoder_slow_style_residual_norm" in stats
        and dynamic_norm is not None
    ):
        ratio = _safe_scalar_ratio(
            stats["diag_decoder_slow_style_residual_norm"],
            dynamic_norm,
        )
        if ratio is not None:
            stats["diag_slow_style_to_timbre_ratio"] = ratio
    late_style_norm = _delta_norm(late_style_delta)
    late_timbre_norm = _delta_norm(late_timbre_delta)
    late_anchor_norm = _delta_norm(late_anchor_delta)
    if late_style_norm is not None:
        stats["diag_decoder_late_style_delta_norm"] = late_style_norm
    if late_timbre_norm is not None:
        stats["diag_decoder_late_timbre_delta_norm"] = late_timbre_norm
    if late_anchor_norm is not None:
        stats["diag_decoder_late_anchor_delta_norm"] = late_anchor_norm
    ratio = _safe_scalar_ratio(late_style_norm, late_timbre_norm)
    if ratio is not None:
        stats["diag_decoder_late_style_to_timbre_ratio"] = ratio
    ratio = _safe_scalar_ratio(late_anchor_norm, late_style_norm)
    if ratio is not None:
        stats["diag_decoder_late_anchor_to_style_ratio"] = ratio
    return stats


def _record_style_success_diagnostics(
    diagnostics,
    output,
    sample,
    config,
    *,
    style_repr,
    slow_style_repr,
    fast_style_repr,
):
    style_success_anchor = resolve_style_success_anchor(
        style_repr,
        slow_style_repr,
        fast_style_summary=fast_style_repr,
        combined_style_summary=style_repr,
    )
    style_success_target_state = resolve_style_success_target_bank(
        output,
        memory_summary=_masked_sequence_mean(
            output.get("style_trace_memory"),
            output.get("style_trace_memory_mask"),
        ),
    )
    style_success_target = style_success_target_state.get("summary")
    anchor_bank = normalized_summary_batch(style_success_anchor)
    target_bank = normalized_summary_batch(style_success_target)
    diagnostics["diag_style_success_supervision_scale"] = torch.tensor(
        float(style_success_supervision_scale(output, config))
    )
    for output_key, diag_key in (
        ("style_success_rank_support_scale", "diag_style_success_rank_support_scale"),
        ("style_success_rank_effective_support", "diag_style_success_rank_effective_support"),
        ("style_success_rank_gate_passed", "diag_style_success_rank_gate_passed"),
        ("style_success_rank_term_active", "diag_style_success_rank_term_active"),
        ("style_success_rank_row_scale_mean", "diag_style_success_rank_row_scale_mean"),
        ("style_success_proxy_only_negative_row_frac", "diag_style_success_proxy_only_negative_row_frac"),
        ("style_success_proxy_supplemented_row_frac", "diag_style_success_proxy_supplemented_row_frac"),
        ("style_success_proxy_augmented_row_frac", "diag_style_success_proxy_augmented_row_frac"),
        ("style_success_proxy_backfill_row_frac", "diag_style_success_proxy_backfill_row_frac"),
        ("style_success_rank_source_scale", "diag_style_success_rank_source_scale"),
        ("style_success_proxy_min_batch", "diag_style_success_proxy_min_batch"),
        ("style_success_proxy_batch_gate_passed", "diag_style_success_proxy_batch_gate_passed"),
    ):
        value = output.get(output_key)
        if isinstance(value, torch.Tensor):
            diagnostics[diag_key] = value.detach().float().mean()
    for output_key, diag_key in (
        ("style_success_negative_pair_density", "diag_style_success_negative_pair_density"),
        ("style_success_negative_row_density", "diag_style_success_negative_row_density"),
        (
            "style_success_label_mean_negatives_per_row",
            "diag_style_success_label_mean_negatives_per_row",
        ),
        (
            "style_success_label_rows_below_proxy_min_count_frac",
            "diag_style_success_label_rows_below_proxy_min_count_frac",
        ),
        ("style_success_mean_negatives_per_row", "diag_style_success_mean_negatives_per_row"),
        (
            "style_success_mean_negatives_per_valid_row",
            "diag_style_success_mean_negatives_per_valid_row",
        ),
        (
            "style_success_proxy_informative_feature_count",
            "diag_style_success_proxy_informative_feature_count",
        ),
        ("style_success_proxy_feature_count", "diag_style_success_proxy_feature_count"),
        (
            "style_success_label_negative_pair_density",
            "diag_style_success_label_negative_pair_density",
        ),
        (
            "style_success_proxy_negative_pair_density",
            "diag_style_success_proxy_negative_pair_density",
        ),
        (
            "style_success_rank_min_negative_row_frac",
            "diag_style_success_rank_min_negative_row_frac",
        ),
        (
            "style_success_rank_min_mean_negatives_per_row",
            "diag_style_success_rank_min_mean_negatives_per_row",
        ),
        (
            "style_success_rank_min_effective_support",
            "diag_style_success_rank_min_effective_support",
        ),
        (
            "style_success_proxy_min_informative_features",
            "diag_style_success_proxy_min_informative_features",
        ),
    ):
        value = output.get(output_key)
        if isinstance(value, torch.Tensor):
            diagnostics[diag_key] = value.detach().float().mean()
    style_success_target_source = str(style_success_target_state.get("source", "none"))
    style_success_memory_fallback_used = bool(
        style_success_target_state.get(
            "memory_fallback_used",
            style_success_target_state.get("memory_used", False),
        )
    )
    diagnostics["diag_style_success_target_memory_fallback_used"] = torch.tensor(
        1.0 if style_success_memory_fallback_used else 0.0
    )
    diagnostics["diag_style_success_target_memory_used"] = torch.tensor(
        1.0 if style_success_memory_fallback_used else 0.0
    )
    diagnostics["diag_style_success_runtime_target_is_self_derived"] = torch.tensor(
        1.0 if style_success_target_source == "runtime_self_derived" else 0.0
    )
    diagnostics["diag_style_success_runtime_target_is_reference_driven"] = torch.tensor(
        1.0
        if style_success_target_source
        in {
            "runtime_reference_derived",
            "global_reference_derived",
        }
        else 0.0
    )
    diagnostics["diag_style_success_target_source_runtime_reference_derived"] = torch.tensor(
        1.0 if style_success_target_source == "runtime_reference_derived" else 0.0
    )
    diagnostics["diag_style_success_target_source_global_reference_derived"] = torch.tensor(
        1.0 if style_success_target_source == "global_reference_derived" else 0.0
    )
    diagnostics["diag_style_success_target_source_style_memory_reference_fallback"] = torch.tensor(
        1.0 if style_success_target_source == "style_memory_reference_fallback" else 0.0
    )
    diagnostics["diag_style_success_target_source_memory_reference_derived"] = torch.tensor(
        1.0
        if style_success_target_source in {"style_memory_reference_fallback", "memory_reference_derived"}
        else 0.0
    )
    diagnostics["diag_style_success_target_source_none"] = torch.tensor(
        1.0 if style_success_target_source == "none" else 0.0
    )
    diagnostics["diag_style_success_uses_weak_negative_ranking"] = torch.tensor(0.0)
    diagnostics["diag_style_success_uses_proxy_negative_ranking"] = torch.tensor(0.0)
    if not (
        isinstance(anchor_bank, torch.Tensor)
        and isinstance(target_bank, torch.Tensor)
        and tuple(anchor_bank.shape) == tuple(target_bank.shape)
    ):
        return

    pair_cos = (anchor_bank * target_bank).sum(dim=-1)
    diagnostics["diag_style_success_pair_cos"] = pair_cos.mean()
    negative_state = resolve_style_success_negative_masks(
        sample,
        batch_size=anchor_bank.size(0),
        device=anchor_bank.device,
        proxy_threshold=float(config.get("style_success_proxy_negative_threshold", 1.25)),
        proxy_min_count=int(config.get("style_success_proxy_negative_min_count", 2)),
        proxy_min_batch=int(config.get("style_success_proxy_min_batch", 4)),
        use_rate_proxy=resolve_style_success_bool_flag(
            config.get("style_success_proxy_use_rate_proxy", False),
            default=False,
        ),
    )
    negative_mask = negative_state.get("negative_mask")
    label_valid_rows = negative_state.get("label_valid_rows")
    proxy_valid_rows = negative_state.get("proxy_valid_rows")
    valid_rows = negative_state.get("valid_rows")
    negative_source = str(negative_state.get("source", "none"))
    support_state = resolve_style_success_rank_support_state(
        negative_state,
        config,
        device=anchor_bank.device,
    )
    rank_supervision_active = (
        float(config.get("lambda_style_success_rank", 0.0)) > 0.0
        and float(config.get("style_success_rank_weight", 1.0)) > 0.0
    )
    diagnostics["diag_style_success_uses_weak_negative_ranking"] = torch.tensor(
        1.0
        if rank_supervision_active
        and negative_source in {"label", "label_plus_proxy_backfill"}
        and bool(float(support_state["gate_passed"].item()) > 0.0)
        else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_uses_proxy_negative_ranking"] = torch.tensor(
        1.0
        if rank_supervision_active
        and negative_source in {"proxy", "label_plus_proxy_backfill"}
        and bool(float(support_state["gate_passed"].item()) > 0.0)
        else 0.0,
        device=anchor_bank.device,
    )
    if isinstance(label_valid_rows, torch.Tensor):
        diagnostics["diag_style_success_label_negative_row_frac"] = label_valid_rows.float().mean()
    if isinstance(proxy_valid_rows, torch.Tensor):
        diagnostics["diag_style_success_proxy_negative_row_frac"] = proxy_valid_rows.float().mean()
    if isinstance(valid_rows, torch.Tensor):
        diagnostics["diag_style_success_negative_row_frac"] = valid_rows.float().mean()
        diagnostics["diag_style_success_rank_valid_row_frac"] = valid_rows.float().mean()
    for state_key, diag_key in (
        ("negative_pair_density", "diag_style_success_negative_pair_density"),
        ("negative_row_density", "diag_style_success_negative_row_density"),
        ("mean_negatives_per_row", "diag_style_success_mean_negatives_per_row"),
        (
            "mean_negatives_per_valid_row",
            "diag_style_success_mean_negatives_per_valid_row",
        ),
        (
            "label_negative_pair_density",
            "diag_style_success_label_negative_pair_density",
        ),
        (
            "proxy_negative_pair_density",
            "diag_style_success_proxy_negative_pair_density",
        ),
        (
            "proxy_backfill_row_frac",
            "diag_style_success_proxy_backfill_row_frac",
        ),
    ):
        value = negative_state.get(state_key)
        if isinstance(value, torch.Tensor):
            diagnostics[diag_key] = value.detach().float().mean()
        elif isinstance(value, (int, float)):
            diagnostics[diag_key] = torch.tensor(float(value), device=anchor_bank.device)
    diagnostics["diag_style_success_proxy_informative_feature_count"] = torch.tensor(
        float(negative_state.get("proxy_informative_feature_count", 0)),
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_proxy_feature_count"] = torch.tensor(
        float(negative_state.get("proxy_feature_count", 0)),
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_rank_effective_support"] = support_state[
        "effective_support"
    ].detach().float().mean()
    diagnostics["diag_style_success_rank_gate_passed"] = support_state[
        "gate_passed"
    ].detach().float().mean()
    diagnostics["diag_style_success_rank_term_active"] = support_state[
        "rank_term_active"
    ].detach().float().mean()
    diagnostics["diag_style_success_negative_source_label"] = torch.tensor(
        1.0 if negative_source == "label" else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_negative_source_proxy"] = torch.tensor(
        1.0 if negative_source == "proxy" else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_negative_source_label_plus_proxy_backfill"] = torch.tensor(
        1.0 if negative_source == "label_plus_proxy_backfill" else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_rank_disabled_reason_active"] = torch.tensor(
        1.0 if str(support_state.get("disabled_reason", "unknown")) == "active" else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_rank_disabled_reason_no_valid_rows"] = torch.tensor(
        1.0
        if str(support_state.get("disabled_reason", "unknown")) == "no_valid_negative_rows"
        else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_rank_disabled_reason_low_row_density"] = torch.tensor(
        1.0
        if str(support_state.get("disabled_reason", "unknown")) == "negative_row_density_below_floor"
        else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_rank_disabled_reason_low_negatives"] = torch.tensor(
        1.0
        if str(support_state.get("disabled_reason", "unknown")) == "mean_negatives_per_row_below_floor"
        else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_rank_disabled_reason_low_proxy_features"] = torch.tensor(
        1.0
        if str(support_state.get("disabled_reason", "unknown")) == "proxy_informative_features_below_floor"
        else 0.0,
        device=anchor_bank.device,
    )
    diagnostics["diag_style_success_rank_disabled_reason_low_effective_support"] = torch.tensor(
        1.0
        if str(support_state.get("disabled_reason", "unknown")) == "effective_support_below_floor"
        else 0.0,
        device=anchor_bank.device,
    )
    if isinstance(negative_mask, torch.Tensor):
        diagnostics["diag_style_success_negative_pair_frac"] = negative_mask.float().mean()
        if isinstance(valid_rows, torch.Tensor) and valid_rows.any():
            similarity = anchor_bank @ target_bank.transpose(0, 1)
            hard_negative = similarity.masked_fill(
                ~negative_mask,
                float("-inf"),
            ).max(dim=-1).values
            hard_negative = hard_negative[valid_rows]
            if hard_negative.numel() > 0 and torch.isfinite(hard_negative).any():
                hard_negative = hard_negative[torch.isfinite(hard_negative)]
                diagnostics["diag_style_success_hard_negative_cos"] = hard_negative.mean()
                diagnostics["diag_style_success_pair_margin"] = (
                    pair_cos[valid_rows].mean() - hard_negative.mean()
                )


def collect_control_diagnostics(output, sample, config):
    if not config.get("log_control_diagnostics", True):
        return {}

    diagnostics = {}
    with torch.no_grad():
        for sample_key, diag_mean_key, diag_std_key in (
            ("style_strengths", "diag_style_strength_mean", "diag_style_strength_std"),
        ):
            mean, std = _safe_mean_std(sample.get(sample_key))
            if mean is not None:
                diagnostics[diag_mean_key] = mean
            if std is not None:
                diagnostics[diag_std_key] = std

        style_trace, style_trace_mask = _resolve_style_owner_sequence(output)
        slow_style_trace = output.get("slow_style_trace")
        dynamic_timbre = output.get("dynamic_timbre")
        style_repr = _masked_sequence_mean(style_trace, style_trace_mask)
        fast_style_repr = summary_vector(output.get("style_trace_summary"))
        if not isinstance(fast_style_repr, torch.Tensor) or fast_style_repr.dim() != 2:
            fast_style_repr = _masked_sequence_mean(
                output.get("style_trace"),
                output.get("style_trace_mask"),
            )
        slow_style_repr = _masked_sequence_mean(
            output.get("slow_style_trace"),
            output.get("slow_style_trace_mask"),
        )
        timbre_repr = _masked_sequence_mean(
            dynamic_timbre,
            _present(output, "dynamic_timbre_mask", "style_trace_mask"),
        )
        global_timbre_anchor = summary_vector(output.get("global_timbre_anchor"))
        global_style_summary = summary_vector(
            _present(output, "global_style_summary_runtime", "global_style_summary")
        )
        output_identity_embed = summary_vector(output.get("output_identity_embed"))
        reference_identity_embed = summary_vector(output.get("reference_identity_embed"))
        style_query_repr = _summary_like(output.get("style_query_inp"))
        timbre_query_repr = _summary_like(output.get("timbre_query_inp"))

        style_valid = _mask_valid_fraction(style_trace_mask, style_trace)
        if style_valid is not None:
            diagnostics["diag_style_trace_valid_frac"] = style_valid
        slow_style_valid = _mask_valid_fraction(output.get("slow_style_trace_mask"), slow_style_trace)
        if slow_style_valid is not None:
            diagnostics["diag_slow_style_trace_valid_frac"] = slow_style_valid
        timbre_valid = _mask_valid_fraction(
            _present(output, "dynamic_timbre_mask", "style_trace_mask"),
            dynamic_timbre,
        )
        if timbre_valid is not None:
            diagnostics["diag_dynamic_timbre_valid_frac"] = timbre_valid

        style_trace_norm, _ = _safe_mean_std(
            style_repr.norm(dim=-1) if isinstance(style_repr, torch.Tensor) else None
        )
        if style_trace_norm is not None:
            diagnostics["diag_style_trace_norm"] = style_trace_norm
        timbre_norm, _ = _safe_mean_std(
            timbre_repr.norm(dim=-1) if isinstance(timbre_repr, torch.Tensor) else None
        )
        if timbre_norm is not None:
            diagnostics["diag_dynamic_timbre_norm"] = timbre_norm
        style_query_norm, style_query_norm_std = _safe_mean_std(
            style_query_repr.norm(dim=-1) if isinstance(style_query_repr, torch.Tensor) else None
        )
        if style_query_norm is not None:
            diagnostics["diag_style_query_norm"] = style_query_norm
        if style_query_norm_std is not None:
            diagnostics["diag_style_query_norm_std"] = style_query_norm_std
        timbre_query_norm, timbre_query_norm_std = _safe_mean_std(
            timbre_query_repr.norm(dim=-1) if isinstance(timbre_query_repr, torch.Tensor) else None
        )
        if timbre_query_norm is not None:
            diagnostics["diag_timbre_query_norm"] = timbre_query_norm
        if timbre_query_norm_std is not None:
            diagnostics["diag_timbre_query_norm_std"] = timbre_query_norm_std

        style_timbre_cos = _cosine_mean(style_repr, timbre_repr)
        if style_timbre_cos is not None:
            diagnostics["diag_style_dynamic_timbre_cos"] = style_timbre_cos
        fast_slow_style_cos = _cosine_mean(
            output.get("fast_style_decoder_residual"),
            output.get("slow_style_decoder_residual"),
        )
        if fast_slow_style_cos is not None:
            diagnostics["diag_fast_slow_style_cos"] = fast_slow_style_cos
        style_owner_base_norm = _delta_norm(
            _present(output, "style_owner_base_residual", "slow_style_decoder_residual")
        )
        if style_owner_base_norm is not None:
            diagnostics["diag_style_owner_base_norm"] = style_owner_base_norm
        style_owner_innovation_norm = _delta_norm(output.get("style_owner_innovation_residual"))
        if style_owner_innovation_norm is not None:
            diagnostics["diag_style_owner_innovation_norm"] = style_owner_innovation_norm
        style_owner_innovation_ratio = _safe_scalar_ratio(
            style_owner_innovation_norm,
            style_owner_base_norm,
        )
        if style_owner_innovation_ratio is not None:
            diagnostics["diag_style_owner_innovation_to_base_ratio"] = style_owner_innovation_ratio
        style_global_cos = _cosine_mean(style_repr, global_style_summary)
        if style_global_cos is not None:
            diagnostics["diag_style_global_cos"] = style_global_cos
        slow_style_global_cos = _cosine_mean(slow_style_repr, global_style_summary)
        if slow_style_global_cos is not None:
            diagnostics["diag_slow_style_global_cos"] = slow_style_global_cos
        _record_style_success_diagnostics(
            diagnostics,
            output,
            sample,
            config,
            style_repr=style_repr,
            slow_style_repr=slow_style_repr,
            fast_style_repr=fast_style_repr,
        )
        timbre_anchor_cos = _cosine_mean(timbre_repr, global_timbre_anchor)
        if timbre_anchor_cos is not None:
            diagnostics["diag_timbre_anchor_cos"] = timbre_anchor_cos
        output_identity_target_embed = summary_vector(output.get("output_identity_target_used_for_loss"))
        if output_identity_target_embed is None:
            output_identity_target_embed = summary_vector(output.get("output_identity_target_embed"))
        if output_identity_target_embed is None:
            output_identity_target_embed = summary_vector(output.get("output_identity_reference_target"))
        if output_identity_target_embed is None:
            output_identity_target_embed = summary_vector(
                _present(output, "output_identity_anchor_target", "global_timbre_anchor")
            )
        target_source = _present(
            output,
            "output_identity_target_source_resolved_for_loss",
            "output_identity_target_source",
        )
        if target_source is not None:
            diagnostics["diag_output_identity_target_source_is_reference"] = _categorical_indicator(
                target_source,
                "reference_identity_embed",
            )
            diagnostics["diag_output_identity_target_source_is_anchor"] = _categorical_indicator(
                target_source,
                "output_identity_anchor_target",
            )
        output_identity_cos = _cosine_mean(output_identity_embed, global_timbre_anchor)
        if output_identity_cos is not None:
            diagnostics["diag_output_identity_anchor_cos"] = output_identity_cos
            diagnostics["diag_output_identity_anchor_distance"] = 1.0 - output_identity_cos
        output_identity_ref_cos = _cosine_mean(output_identity_embed, reference_identity_embed)
        if output_identity_ref_cos is not None:
            diagnostics["diag_output_identity_ref_cos"] = output_identity_ref_cos
        output_identity_target_cos = _cosine_mean(
            output_identity_embed,
            output_identity_target_embed,
        )
        if output_identity_target_cos is not None:
            diagnostics["diag_output_identity_target_cos"] = output_identity_target_cos
            diagnostics["diag_output_identity_target_distance"] = 1.0 - output_identity_target_cos
        query_cos = _cosine_mean(
            style_query_repr,
            timbre_query_repr,
        )
        if query_cos is not None:
            diagnostics["diag_style_timbre_query_cos"] = query_cos
        output_identity_norm, output_identity_norm_std = _safe_mean_std(
            output_identity_embed.norm(dim=-1) if isinstance(output_identity_embed, torch.Tensor) else None
        )
        if output_identity_norm is not None:
            diagnostics["diag_output_identity_norm"] = output_identity_norm
        if output_identity_norm_std is not None:
            diagnostics["diag_output_identity_norm_std"] = output_identity_norm_std
        runtime_budget_clip_frac = output.get("runtime_dynamic_timbre_style_budget_clip_frac")
        if isinstance(runtime_budget_clip_frac, torch.Tensor):
            diagnostics["diag_runtime_dynamic_timbre_budget_clip_frac"] = runtime_budget_clip_frac.float().mean()
        elif runtime_budget_clip_frac is not None:
            diagnostics["diag_runtime_dynamic_timbre_budget_clip_frac"] = torch.tensor(float(runtime_budget_clip_frac))
        runtime_budget_applied = output.get("runtime_dynamic_timbre_style_budget_applied")
        if runtime_budget_applied is not None:
            diagnostics["diag_runtime_dynamic_timbre_budget_applied"] = torch.tensor(float(bool(runtime_budget_applied)))
        prebudget_dynamic_timbre = _present(
            output,
            "dynamic_timbre_decoder_residual_prebudget",
            "dynamic_timbre_decoder_residual",
        )
        postbudget_dynamic_timbre = output.get("dynamic_timbre_decoder_residual")
        budget_support_weight = output.get("dynamic_timbre_budget_support_weight")
        if not isinstance(budget_support_weight, torch.Tensor):
            budget_support_weight = _resolve_dynamic_timbre_budget_support_weight(
                sample,
                output,
                config,
                prebudget_dynamic_timbre,
            )
        if isinstance(budget_support_weight, torch.Tensor):
            support_mask = _normalize_mask(
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                prebudget_dynamic_timbre,
            )
            valid_frame_mask = (
                (~support_mask)
                if isinstance(support_mask, torch.Tensor)
                else torch.ones_like(budget_support_weight, dtype=torch.bool)
            )
            diagnostics["diag_dynamic_timbre_budget_support_mean"] = (
                budget_support_weight[valid_frame_mask].float().mean()
                if valid_frame_mask.any()
                else budget_support_weight.float().mean()
            )
            sample_uv = sample.get("uv")
            if (
                isinstance(sample_uv, torch.Tensor)
                and sample_uv.dim() == 2
                and tuple(sample_uv.shape) == tuple(budget_support_weight.shape)
            ):
                uv_mask = (sample_uv > 0) & valid_frame_mask
                voiced_mask = (~(sample_uv > 0)) & valid_frame_mask
                if uv_mask.any():
                    diagnostics["diag_dynamic_timbre_budget_support_uv_mean"] = budget_support_weight[uv_mask].float().mean()
                if voiced_mask.any():
                    diagnostics["diag_dynamic_timbre_budget_support_voiced_mean"] = budget_support_weight[voiced_mask].float().mean()
        prebudget_norm = _delta_norm(prebudget_dynamic_timbre)
        postbudget_norm = _delta_norm(postbudget_dynamic_timbre)
        if prebudget_norm is not None:
            diagnostics["diag_dynamic_timbre_prebudget_norm"] = prebudget_norm
        if postbudget_norm is not None:
            diagnostics["diag_dynamic_timbre_postbudget_norm"] = postbudget_norm
        budget_ratio = _safe_scalar_ratio(postbudget_norm, prebudget_norm)
        if budget_ratio is not None:
            diagnostics["diag_dynamic_timbre_post_to_pre_budget_ratio"] = budget_ratio
        style_decoder_residual = output.get("style_decoder_residual")
        overlap_margin = float(config.get("style_timbre_runtime_overlap_margin", 0.10))
        overlap_use_abs = resolve_style_success_bool_flag(
            config.get("style_timbre_runtime_overlap_use_abs", True),
            default=True,
        )
        overlap_terms = masked_sequence_cosine(
            style_decoder_residual,
            prebudget_dynamic_timbre,
            mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
            boundary_mask=output.get("dynamic_timbre_boundary_mask"),
            frame_weight=budget_support_weight,
            absolute=False,
            margin=0.0,
        )
        overlap_abs_terms = masked_sequence_cosine(
            style_decoder_residual,
            prebudget_dynamic_timbre,
            mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
            boundary_mask=output.get("dynamic_timbre_boundary_mask"),
            frame_weight=budget_support_weight,
            absolute=True,
            margin=0.0,
        )
        overlap_margin_terms = masked_sequence_cosine(
            style_decoder_residual,
            prebudget_dynamic_timbre,
            mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
            boundary_mask=output.get("dynamic_timbre_boundary_mask"),
            frame_weight=budget_support_weight,
            absolute=overlap_use_abs,
            margin=overlap_margin,
        )
        overlap_postbudget_terms = masked_sequence_cosine(
            style_decoder_residual,
            postbudget_dynamic_timbre,
            mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
            boundary_mask=output.get("dynamic_timbre_boundary_mask"),
            frame_weight=budget_support_weight,
            absolute=True,
            margin=0.0,
        )
        if overlap_terms.get("reduced") is not None:
            diagnostics["diag_style_timbre_runtime_cos"] = overlap_terms["reduced"]
        if overlap_abs_terms.get("reduced") is not None:
            diagnostics["diag_style_timbre_runtime_abs_cos"] = overlap_abs_terms["reduced"]
        if overlap_margin_terms.get("reduced") is not None:
            diagnostics["diag_style_timbre_runtime_overlap_margin_violation"] = (
                overlap_margin_terms["reduced"]
            )
        if overlap_postbudget_terms.get("reduced") is not None:
            diagnostics["diag_style_timbre_runtime_postbudget_abs_cos"] = (
                overlap_postbudget_terms["reduced"]
            )
        if (
            overlap_abs_terms.get("reduced") is not None
            and overlap_postbudget_terms.get("reduced") is not None
        ):
            pre_abs_cos = overlap_abs_terms["reduced"]
            post_abs_cos = overlap_postbudget_terms["reduced"]
            diagnostics["diag_style_timbre_runtime_abs_cos_reduction"] = (
                pre_abs_cos - post_abs_cos
            )
            diagnostics["diag_style_timbre_runtime_post_to_pre_abs_cos_ratio"] = (
                post_abs_cos / pre_abs_cos.clamp_min(1e-6)
            )
        style_energy = sequence_energy_mean(
            style_decoder_residual,
            mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
            boundary_mask=output.get("dynamic_timbre_boundary_mask"),
            frame_weight=budget_support_weight,
        )
        timbre_pre_energy = sequence_energy_mean(
            prebudget_dynamic_timbre,
            mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
            boundary_mask=output.get("dynamic_timbre_boundary_mask"),
            frame_weight=budget_support_weight,
        )
        timbre_post_energy = sequence_energy_mean(
            postbudget_dynamic_timbre,
            mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
            boundary_mask=output.get("dynamic_timbre_boundary_mask"),
            frame_weight=budget_support_weight,
        )
        if style_energy is not None:
            diagnostics["diag_style_decoder_residual_energy"] = style_energy
        if timbre_pre_energy is not None:
            diagnostics["diag_dynamic_timbre_prebudget_energy"] = timbre_pre_energy
        if timbre_post_energy is not None:
            diagnostics["diag_dynamic_timbre_postbudget_energy"] = timbre_post_energy
        if style_energy is not None and timbre_pre_energy is not None:
            diagnostics["diag_dynamic_timbre_prebudget_to_style_energy_ratio"] = (
                timbre_pre_energy / style_energy.clamp_min(1e-6)
            )
        if style_energy is not None and timbre_post_energy is not None:
            diagnostics["diag_dynamic_timbre_postbudget_to_style_energy_ratio"] = (
                timbre_post_energy / style_energy.clamp_min(1e-6)
            )
        identity_backend = str(output.get("identity_encoder_backend", "none"))
        diagnostics["diag_identity_backend_is_external"] = torch.tensor(
            float(identity_backend == "external_speaker_verifier")
        )
        diagnostics["diag_identity_encoder_frozen_for_loss"] = torch.tensor(
            float(
                bool(
                    _present(
                        output,
                        "identity_encoder_frozen_for_loss",
                        "identity_encoder_params_frozen_for_loss",
                        default=False,
                    )
                )
            )
        )
        tensor_kwargs = {"dtype": torch.float32}
        if isinstance(style_repr, torch.Tensor):
            tensor_kwargs["device"] = style_repr.device
        diagnostics["diag_global_timbre_to_pitch_applied"] = torch.tensor(
            1.0 if bool(output.get("global_timbre_to_pitch_applied", False)) else 0.0,
            **tensor_kwargs,
        )
        diagnostics["diag_query_anchor_split_applied"] = torch.tensor(
            1.0 if bool(output.get("query_anchor_split_applied", False)) else 0.0,
            **tensor_kwargs,
        )
        reference_contract = output.get("reference_contract", {})
        diagnostics["diag_factorization_guaranteed"] = torch.tensor(
            1.0 if bool(reference_contract.get("factorization_guaranteed", False)) else 0.0,
            **tensor_kwargs,
        )
        diagnostics["diag_runtime_dynamic_timbre_style_budget_applied"] = torch.tensor(
            1.0 if bool(output.get("runtime_dynamic_timbre_style_budget_applied", False)) else 0.0,
            **tensor_kwargs,
        )
        runtime_budget_clip_frac = output.get("runtime_dynamic_timbre_style_budget_clip_frac")
        if isinstance(runtime_budget_clip_frac, torch.Tensor):
            diagnostics["diag_runtime_dynamic_timbre_style_budget_clip_frac"] = runtime_budget_clip_frac.to(
                **tensor_kwargs
            )
        for output_key, diag_key in (
            (
                "runtime_dynamic_timbre_style_budget_overflow",
                "diag_runtime_dynamic_timbre_style_budget_overflow",
            ),
            (
                "runtime_dynamic_timbre_style_budget_relative_overflow",
                "diag_runtime_dynamic_timbre_style_budget_relative_overflow",
            ),
            (
                "runtime_dynamic_timbre_style_energy",
                "diag_runtime_dynamic_timbre_style_energy",
            ),
            (
                "runtime_dynamic_timbre_style_owner_energy",
                "diag_runtime_dynamic_timbre_style_owner_energy",
            ),
            (
                "runtime_dynamic_timbre_slow_style_energy",
                "diag_runtime_dynamic_timbre_slow_style_energy",
            ),
            (
                "runtime_dynamic_timbre_slow_style_excess_energy",
                "diag_runtime_dynamic_timbre_slow_style_excess_energy",
            ),
            (
                "runtime_dynamic_timbre_style_base_energy",
                "diag_runtime_dynamic_timbre_style_base_energy",
            ),
            (
                "runtime_dynamic_timbre_style_innovation_energy",
                "diag_runtime_dynamic_timbre_style_innovation_energy",
            ),
            (
                "runtime_dynamic_timbre_dynamic_energy",
                "diag_runtime_dynamic_timbre_dynamic_energy",
            ),
        ):
            diagnostics.update(
                _simple_sequence_statistics(
                    output.get(output_key),
                    _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                    prefix=diag_key,
                )
            )
        budget_ratio = float(
            output.get(
                "runtime_dynamic_timbre_style_budget_ratio",
                config.get(
                    "runtime_dynamic_timbre_style_budget_ratio",
                    config.get("dynamic_timbre_budget_ratio", 0.40),
                ),
            )
        )
        budget_margin = float(
            output.get(
                "runtime_dynamic_timbre_style_budget_margin",
                config.get(
                    "runtime_dynamic_timbre_style_budget_margin",
                    config.get("dynamic_timbre_budget_margin", 0.0),
                ),
            )
        )
        budget_slow_style_weight = float(
            output.get(
                "runtime_dynamic_timbre_style_budget_slow_style_weight",
                config.get(
                    "runtime_dynamic_timbre_style_budget_slow_style_weight",
                    config.get("dynamic_timbre_budget_slow_style_weight", 1.0),
                ),
            )
        )
        budget_epsilon = float(
            output.get(
                "runtime_dynamic_timbre_style_budget_epsilon",
                config.get(
                    "runtime_dynamic_timbre_style_budget_epsilon",
                    config.get("dynamic_timbre_budget_epsilon", 1e-6),
                ),
            )
        )
        stage_outputs = output.get("decoder_style_adapter_stages")
        stage_mean_buckets = {
            "style": [],
            "timbre": [],
            "overflow": [],
        }
        stage_std_buckets = {
            "style": [],
            "timbre": [],
            "overflow": [],
        }
        if isinstance(stage_outputs, dict):
            for stage_name in ("mid", "late"):
                stage_meta = stage_outputs.get(stage_name)
                stage_terms = resolve_stage_dynamic_timbre_budget_terms(
                    stage_meta,
                    padding_mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
                    budget_ratio=budget_ratio,
                    budget_margin=budget_margin,
                    slow_style_weight=budget_slow_style_weight,
                    budget_epsilon=budget_epsilon,
                )
                stage_style_budget = stage_terms.get("style_energy")
                stage_timbre_budget = stage_terms.get("timbre_energy")
                if not isinstance(stage_style_budget, torch.Tensor) or not isinstance(stage_timbre_budget, torch.Tensor):
                    continue
                stage_overflow = stage_terms.get("overflow")
                if not isinstance(stage_overflow, torch.Tensor):
                    stage_overflow = F.relu(
                        stage_timbre_budget - budget_ratio * stage_style_budget.detach() - budget_margin
                    )
                stage_weight = build_sequence_weight(
                    _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                    reference=stage_timbre_budget,
                    boundary_mask=output.get("dynamic_timbre_boundary_mask"),
                    frame_weight=budget_support_weight,
                )
                for value, bucket_key, prefix in (
                    (stage_style_budget, "style", f"diag_decoder_{stage_name}_style_budget_energy"),
                    (stage_timbre_budget, "timbre", f"diag_decoder_{stage_name}_dynamic_timbre_energy"),
                    (stage_overflow, "overflow", f"diag_decoder_{stage_name}_dynamic_timbre_budget_overflow"),
                ):
                    mean, std = _weighted_mean_std(value, stage_weight)
                    if mean is not None:
                        diagnostics[f"{prefix}_mean"] = mean
                        stage_mean_buckets[bucket_key].append(mean)
                    if std is not None:
                        diagnostics[f"{prefix}_std"] = std
                        stage_std_buckets[bucket_key].append(std)
        aggregate_prefix = {
            "style": "diag_decoder_stage_style_budget_energy",
            "timbre": "diag_decoder_stage_dynamic_timbre_energy",
            "overflow": "diag_decoder_stage_dynamic_timbre_budget_overflow",
        }
        for bucket_key, prefix in aggregate_prefix.items():
            if stage_mean_buckets[bucket_key]:
                diagnostics[f"{prefix}_mean"] = torch.stack(stage_mean_buckets[bucket_key]).mean()
            if stage_std_buckets[bucket_key]:
                diagnostics[f"{prefix}_std"] = torch.stack(stage_std_buckets[bucket_key]).mean()
        dynamic_timbre_prebudget = _present(
            output,
            "dynamic_timbre_decoder_residual_prebudget",
            "dynamic_timbre_decoder_residual",
        )
        if isinstance(dynamic_timbre_prebudget, torch.Tensor) and dynamic_timbre_prebudget.dim() == 3:
            prebudget_mask = _normalize_mask(
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                dynamic_timbre_prebudget,
            )
            prebudget_norm = dynamic_timbre_prebudget.float().norm(dim=-1)
            if prebudget_mask is None:
                valid_prebudget = prebudget_norm.reshape(-1)
            else:
                valid_prebudget = prebudget_norm[(~prebudget_mask)]
            if valid_prebudget.numel() > 0:
                diagnostics["diag_dynamic_timbre_prebudget_mean"] = valid_prebudget.mean()
                diagnostics["diag_dynamic_timbre_prebudget_std"] = valid_prebudget.std(unbiased=False)
        if "pitch_residual_safe_target_available" in output:
            diagnostics["diag_pitch_residual_target_available"] = torch.tensor(
                float(bool(output.get("pitch_residual_safe_target_available")))
            )
        for output_key, diag_key in (
            ("dynamic_timbre_budget_prebudget_term", "diag_dynamic_timbre_budget_prebudget_term"),
            ("dynamic_timbre_budget_mid_term", "diag_dynamic_timbre_budget_mid_term"),
            ("dynamic_timbre_budget_late_term", "diag_dynamic_timbre_budget_late_term"),
            ("pitch_residual_safe_align_term", "diag_pitch_residual_align_term"),
            ("pitch_residual_safe_budget_overflow_term", "diag_pitch_residual_budget_overflow"),
            ("pitch_residual_safe_slope_term", "diag_pitch_residual_slope_term"),
            ("pitch_residual_safe_zero_fallback_term", "diag_pitch_residual_zero_fallback_term"),
            ("decoder_late_owner_overflow_term", "diag_decoder_late_owner_overflow_term"),
            ("decoder_late_owner_style_floor_term", "diag_decoder_late_owner_style_floor_term"),
            ("decoder_late_owner_full_style_energy", "diag_decoder_late_owner_full_style_energy"),
            ("decoder_late_owner_late_style_energy", "diag_decoder_late_owner_late_style_energy"),
        ):
            value = output.get(output_key)
            if isinstance(value, torch.Tensor):
                diagnostics[diag_key] = value.detach().float().mean()
        for output_key, diag_key in (
            ("reference_curriculum_progress", "diag_reference_curriculum_progress"),
            ("reference_curriculum_external_prob", "diag_reference_curriculum_external_prob"),
            ("reference_curriculum_self_prob", "diag_reference_curriculum_self_prob"),
            ("reference_curriculum_gloss_scale", "diag_reference_curriculum_gloss_scale"),
            ("forcing_prob", "diag_prosody_forcing_prob"),
            ("forcing_schedule_progress", "diag_prosody_forcing_progress"),
        ):
            value = output.get(output_key)
            if isinstance(value, torch.Tensor):
                diagnostics[diag_key] = value.to(**tensor_kwargs)
            elif isinstance(value, (int, float)):
                diagnostics[diag_key] = torch.tensor(float(value), **tensor_kwargs)
        identity_frozen_output_key = None
        identity_frozen_value = _present(
            output,
            "identity_encoder_frozen_for_loss",
            "identity_encoder_params_frozen_for_loss",
        )
        if identity_frozen_value is not None:
            identity_frozen_output_key = (
                "identity_encoder_frozen_for_loss"
                if output.get("identity_encoder_frozen_for_loss") is not None
                else "identity_encoder_params_frozen_for_loss"
            )
        for output_key, diag_key in (
            ("reference_curriculum_use_external_ref", "diag_reference_curriculum_use_external_ref"),
            ("forcing_enabled", "diag_prosody_forcing_enabled"),
        ):
            value = output.get(output_key, None)
            if value is not None:
                diagnostics[diag_key] = torch.tensor(
                    1.0 if bool(value) else 0.0,
                    **tensor_kwargs,
                )
        if identity_frozen_output_key is not None:
            diagnostics["diag_output_identity_internal_encoder_frozen"] = torch.tensor(
                1.0 if bool(identity_frozen_value) else 0.0,
                **tensor_kwargs,
            )
        identity_encoder_backend = output.get("identity_encoder_backend", None)
        if identity_encoder_backend is not None:
            diagnostics["diag_output_identity_uses_external_verifier"] = torch.tensor(
                1.0 if str(identity_encoder_backend) == "external_speaker_verifier" else 0.0,
                **tensor_kwargs,
            )
            diagnostics["diag_output_identity_internal_encoder_trainable"] = torch.tensor(
                1.0 if str(identity_encoder_backend) == "model_encode_spk_embed_trainable_for_loss" else 0.0,
                **tensor_kwargs,
            )

        diagnostics.update(
            _decoder_style_adapter_statistics(
                output.get("decoder_style_adapter_stages"),
                dynamic_timbre_residual=output.get("dynamic_timbre_decoder_residual"),
                style_owner_residual=output.get("style_decoder_residual"),
            )
        )
        style_summary_source = _present(
            output,
            "global_style_summary_runtime_source",
            "global_style_summary_source",
        )
        if style_summary_source is not None:
            fallback_flag = 1.0 if str(style_summary_source) == "fallback_timbre_anchor" else 0.0
            if isinstance(style_repr, torch.Tensor):
                diagnostics["diag_global_style_summary_fallback"] = torch.tensor(
                    fallback_flag,
                    dtype=torch.float32,
                    device=style_repr.device,
                )
            else:
                diagnostics["diag_global_style_summary_fallback"] = torch.tensor(
                    fallback_flag,
                    dtype=torch.float32,
                )

        diagnostics.update(
            _gate_statistics(
                output.get("dynamic_timbre_gate"),
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                target=float(config.get("dynamic_timbre_gate_target", 0.6)),
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("dynamic_timbre_gate_raw"),
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                prefix="diag_dynamic_timbre_gate_raw",
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("dynamic_timbre_gate_logit_raw"),
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                prefix="diag_dynamic_timbre_gate_logit_raw",
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("dynamic_timbre_boundary_mask"),
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                prefix="diag_dynamic_timbre_boundary_mask",
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("dynamic_timbre_boundary_scale"),
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                prefix="diag_dynamic_timbre_boundary_scale",
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("style_router_gate"),
                output.get("style_trace_mask"),
                prefix="diag_style_router_gate",
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("style_burst_score"),
                output.get("style_trace_mask"),
                prefix="diag_style_burst_score",
            )
        )
        style_owner_source = output.get("style_owner_source", None)
        if style_owner_source is not None:
            diagnostics["diag_style_owner_source_dual_router"] = _categorical_indicator(
                style_owner_source,
                "dual_router",
                device=tensor_kwargs.get("device", None),
            )
            diagnostics["diag_style_owner_source_dual_sum"] = _categorical_indicator(
                style_owner_source,
                "dual_sum",
                device=tensor_kwargs.get("device", None),
            )
            diagnostics["diag_style_owner_source_fast_only"] = _categorical_indicator(
                style_owner_source,
                "fast_only",
                device=tensor_kwargs.get("device", None),
            )
            diagnostics["diag_style_owner_source_slow_only"] = _categorical_indicator(
                style_owner_source,
                "slow_only",
                device=tensor_kwargs.get("device", None),
            )
        dynamic_timbre_style_context_source = output.get("dynamic_timbre_style_context_source", None)
        if dynamic_timbre_style_context_source is not None:
            diagnostics["diag_dynamic_timbre_style_context_source_owner_innovation"] = _categorical_indicator(
                dynamic_timbre_style_context_source,
                "style_owner_innovation",
                device=tensor_kwargs.get("device", None),
            )
            diagnostics["diag_dynamic_timbre_style_context_source_owner"] = _categorical_indicator(
                dynamic_timbre_style_context_source,
                "style_owner",
                device=tensor_kwargs.get("device", None),
            )
        anchor_shift = output.get("dynamic_timbre_anchor_shift")
        if isinstance(anchor_shift, torch.Tensor):
            anchor_shift_norm, _ = _safe_mean_std(anchor_shift.norm(dim=-1))
            if anchor_shift_norm is not None:
                diagnostics["diag_dynamic_timbre_anchor_shift_norm"] = anchor_shift_norm

        if config.get("control_diagnostics_track_lambdas", True):
            device = None
            for value in diagnostics.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
            active_mainline_count = 0
            for key in TRACKED_LAMBDA_KEYS:
                if key in config:
                    if float(config.get(key, 0.0)) > 0.0:
                        active_mainline_count += 1
                    if device is not None:
                        diagnostics[f"diag_{key}"] = torch.tensor(
                            float(config.get(key, 0.0)),
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        diagnostics[f"diag_{key}"] = torch.tensor(
                            float(config.get(key, 0.0)),
                            dtype=torch.float32,
                        )
            for key in OPTIONAL_TRACKED_LAMBDA_KEYS:
                if key not in config:
                    continue
                if device is not None:
                    diagnostics[f"diag_{key}"] = torch.tensor(
                        float(config.get(key, 0.0)),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    diagnostics[f"diag_{key}"] = torch.tensor(
                        float(config.get(key, 0.0)),
                        dtype=torch.float32,
                    )
            diagnostics["diag_active_mainline_control_loss_count"] = torch.tensor(
                float(active_mainline_count),
                dtype=torch.float32,
                device=device,
            ) if device is not None else torch.tensor(float(active_mainline_count), dtype=torch.float32)
    return diagnostics
