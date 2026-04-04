import torch
import torch.nn.functional as F

from modules.Conan.common import first_present
from modules.Conan.control.budget_support import (
    resolve_dynamic_timbre_budget_support_weight as _resolve_dynamic_timbre_budget_support_weight,
)
from modules.Conan.control.separation_metrics import (
    build_sequence_weight as _shared_sequence_weight,
    masked_sequence_cosine,
    weighted_mean as _shared_weighted_mean,
)
from modules.Conan.control.style_success import (
    STYLE_SUCCESS_PAIR_WEIGHT,
    STYLE_SUCCESS_RANK_TEMPERATURE,
    STYLE_SUCCESS_RANK_WEIGHT,
    normalized_summary_batch,
    resolve_style_success_anchor,
    resolve_style_success_bool_flag,
    resolve_style_success_negative_masks,
    resolve_style_success_rank_scale_defaults,
    resolve_style_success_rank_source_scale,
    resolve_style_success_rank_support_state,
    resolve_style_success_target_bank,
    style_success_supervision_scale,
)
from modules.Conan.dynamic_timbre_control import (
    resolve_dynamic_timbre_budget_terms,
    resolve_stage_dynamic_timbre_budget_terms,
)
from modules.Conan.style_trace_utils import resolve_combined_style_trace
from tasks.Conan.control_schedule import MAINLINE_MINIMAL_CONTROL_LAMBDAS


def _shape_match_time_mask(mask, target):
    if not isinstance(mask, torch.Tensor) or not isinstance(target, torch.Tensor):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if target.dim() == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)
    if mask.dim() == 2 and target.dim() == 2 and tuple(mask.shape) == tuple(target.shape):
        return mask.float().to(target.device)
    return None


def _summary_vector(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3:
        return value.squeeze(1)
    return value


def _mean_abs_cosine(a, b):
    a = _summary_vector(a)
    b = _summary_vector(b)
    if a is None or b is None:
        return None
    if tuple(a.shape) != tuple(b.shape):
        return None
    return F.cosine_similarity(a, b, dim=-1, eps=1e-6).abs().mean()


def _cosine_distance(a, b):
    a = _summary_vector(a)
    b = _summary_vector(b)
    if a is None or b is None:
        return None
    if tuple(a.shape) != tuple(b.shape):
        return None
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-6).mean()


def _variance_floor_penalty(value, *, target=0.05):
    value = _summary_vector(value)
    if value is None or value.dim() != 2:
        return None
    std = value.float().std(dim=0, unbiased=False)
    if std.numel() <= 0:
        return None
    target = float(target)
    return F.relu(std.new_full(std.shape, target) - std).mean()


def _normalize_sequence_mask(mask, sequence):
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
    mask = _normalize_sequence_mask(mask, sequence)
    if mask is None:
        return None
    valid = (~mask).unsqueeze(-1).to(sequence.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    pooled = (sequence * valid).sum(dim=1) / denom
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def _present(mapping, *keys, default=None):
    return first_present(mapping, *keys, default=default)


def _style_representation(output):
    style_owner = output.get("style_decoder_residual")
    style_owner_mask = _present(output, "style_decoder_residual_mask", "style_trace_mask")
    if isinstance(style_owner, torch.Tensor):
        return _masked_sequence_mean(style_owner, style_owner_mask)
    combined_style_trace, combined_style_trace_mask = resolve_combined_style_trace(output)
    return _masked_sequence_mean(
        combined_style_trace,
        combined_style_trace_mask,
    )


def _fast_style_representation(output):
    fast_summary = _summary_vector(output.get("style_trace_summary"))
    if isinstance(fast_summary, torch.Tensor) and fast_summary.dim() == 2:
        return fast_summary
    return _masked_sequence_mean(
        output.get("style_trace"),
        output.get("style_trace_mask"),
    )


def _timbre_representation(output):
    return _masked_sequence_mean(
        output.get("dynamic_timbre"),
        _present(output, "dynamic_timbre_mask", "style_trace_mask"),
    )


def _global_timbre_anchor(output):
    return _summary_vector(output.get("global_timbre_anchor"))


def _global_style_summary(output):
    return _summary_vector(
        _present(output, "global_style_summary_runtime", "global_style_summary")
    )


def _slow_style_representation(output):
    return _masked_sequence_mean(
        output.get("slow_style_trace"),
        output.get("slow_style_trace_mask"),
    )


def _gate_mean(gate, mask=None):
    if not isinstance(gate, torch.Tensor):
        return None
    if gate.dim() == 3 and gate.size(-1) == 1:
        gate = gate.squeeze(-1)
    if gate.dim() == 2:
        if mask is None:
            return gate.mean()
        mask = _normalize_sequence_mask(mask, gate.unsqueeze(-1))
        if mask is None:
            return None
        valid = (~mask).to(gate.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (gate * valid).sum() / denom
    if gate.dim() == 3:
        return gate.mean()
    return None


def _sequence_abs_mean(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3:
        return value.abs().mean(dim=-1)
    if value.dim() == 2:
        return value.abs()
    return None


def _sequence_weight(mask=None, *, reference=None, boundary_mask=None, voiced_weight=None, frame_weight=None):
    return _shared_sequence_weight(
        mask,
        reference=reference,
        boundary_mask=boundary_mask,
        voiced_weight=voiced_weight,
        frame_weight=frame_weight,
    )


def _weighted_mean(value, weight=None):
    return _shared_weighted_mean(value, weight)


def _weighted_smooth_l1(value, target, weight=None, *, beta=1.0):
    if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
        return None
    if tuple(value.shape) != tuple(target.shape):
        return None
    beta = max(float(beta), 1.0e-6)
    target = target.detach()
    loss = F.smooth_l1_loss(value, target, reduction="none", beta=beta)
    return _weighted_mean(loss, weight)


def _tensor_abs_mean(value):
    if not isinstance(value, torch.Tensor):
        return None
    return value.abs().mean()


def _sum_optional_scalars(*values):
    present = [value for value in values if isinstance(value, torch.Tensor)]
    if not present:
        return None
    total = present[0]
    for value in present[1:]:
        total = total + value
    return total


def _mean_optional_scalars(*values):
    present = [value for value in values if isinstance(value, torch.Tensor)]
    if not present:
        return None
    total = present[0]
    for value in present[1:]:
        total = total + value
    return total / float(len(present))


def _weighted_optional_scalars(*weighted_values):
    total = None
    total_weight = 0.0
    for value, weight in weighted_values:
        if not isinstance(value, torch.Tensor):
            continue
        weight = float(weight)
        if weight <= 0.0:
            continue
        weighted_value = value * weight
        total = weighted_value if total is None else total + weighted_value
        total_weight += weight
    if total is None or total_weight <= 0.0:
        return None
    return total / total_weight


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


def _reduce_budget_overflow(
    timbre_energy,
    style_energy,
    *,
    budget_ratio,
    budget_margin,
    weight=None,
    relative_weight=0.0,
    budget_epsilon=1e-6,
):
    if not isinstance(timbre_energy, torch.Tensor) or not isinstance(style_energy, torch.Tensor):
        return None
    allowed = float(budget_ratio) * style_energy.detach() + float(budget_margin)
    overflow = F.relu(timbre_energy - allowed)
    reduced = _weighted_mean(overflow, weight)
    relative_weight = float(relative_weight)
    if relative_weight > 0.0:
        relative_overflow = overflow / allowed.clamp_min(max(float(budget_epsilon), 1e-8))
        relative_reduced = _weighted_mean(relative_overflow, weight)
        if relative_reduced is not None:
            reduced = (
                relative_reduced * relative_weight
                if reduced is None
                else reduced + relative_reduced * relative_weight
            )
    return reduced


def add_classification_losses(losses, output, sample, *, specs):
    for loss_name, logit_key, target_key, lambda_value in specs:
        if lambda_value <= 0:
            continue
        logits = output.get(logit_key)
        targets = sample.get(target_key)
        if logits is None or targets is None:
            continue
        valid = targets >= 0
        if valid.any():
            losses[loss_name] = F.cross_entropy(logits[valid], targets[valid]) * lambda_value


def add_regression_losses(losses, output, sample, *, specs):
    for loss_name, pred_key, target_key, lambda_value in specs:
        if lambda_value <= 0:
            continue
        pred = output.get(pred_key)
        target = sample.get(target_key)
        if pred is None or target is None:
            continue
        losses[loss_name] = F.mse_loss(pred, target) * lambda_value


def add_energy_loss(losses, output, sample, *, lambda_energy=0.0, nonpadding=None):
    if lambda_energy <= 0 or output.get("energy_pred") is None or sample.get("energy") is None:
        return
    energy_pred, energy = output["energy_pred"], sample["energy"]
    nonpadding = _shape_match_time_mask(nonpadding, energy)
    if nonpadding is None:
        nonpadding = (energy != 0).float()
    denom = nonpadding.sum().clamp_min(1.0)
    losses["energy"] = (
        (F.mse_loss(energy_pred, energy, reduction="none") * nonpadding).sum() / denom
    ) * lambda_energy


def add_weighted_output_losses(losses, output, *, specs):
    for loss_name, output_key, lambda_value in specs:
        if lambda_value <= 0:
            continue
        value = output.get(output_key)
        if value is not None:
            losses[loss_name] = value * lambda_value


def add_optional_passthrough_losses(losses, output, *, specs):
    for loss_name, output_key, enabled in specs:
        if not enabled:
            continue
        value = output.get(output_key)
        if value is not None:
            losses[loss_name] = value


def add_prompt_regularization_losses(losses, output, config):
    global_timbre_anchor = output.get("global_timbre_anchor")
    disentangle_specs = (
        (
            "emotion_style_dis",
            config.get("lambda_emotion_style_disentangle", 0.0),
            output.get("emotion_prompt"),
            global_timbre_anchor,
        ),
        (
            "accent_style_dis",
            config.get("lambda_accent_style_disentangle", 0.0),
            output.get("accent_prompt"),
            global_timbre_anchor,
        ),
        (
            "emotion_accent_dis",
            config.get("lambda_emotion_accent_disentangle", 0.0),
            output.get("emotion_prompt"),
            output.get("accent_prompt"),
        ),
    )
    for loss_name, lambda_value, a, b in disentangle_specs:
        if lambda_value <= 0:
            continue
        value = _mean_abs_cosine(a, b)
        if value is not None:
            losses[loss_name] = value * lambda_value

    gate_specs = (
        (
            "emotion_gate_reg",
            config.get("lambda_emotion_gate", 0.0),
            output.get("emotion_gate"),
            float(config.get("emotion_gate_target", 0.85)),
        ),
        (
            "accent_gate_reg",
            config.get("lambda_accent_gate", 0.0),
            output.get("accent_gate"),
            float(config.get("accent_gate_target", 0.75)),
        ),
    )
    for loss_name, lambda_value, gate, target in gate_specs:
        if lambda_value <= 0:
            continue
        gate_mean = _gate_mean(gate)
        if gate_mean is not None:
            losses[loss_name] = (gate_mean - target).abs() * lambda_value


def _store_detached_scalar(output, key, value, *, device=None):
    if isinstance(value, torch.Tensor):
        output[key] = value.detach()
        return
    if isinstance(value, (bool, int, float)):
        kwargs = {"dtype": torch.float32}
        if device is not None:
            kwargs["device"] = device
        output[key] = torch.tensor(float(value), **kwargs)


def _compute_style_success_control_loss(
    output,
    sample,
    config,
    *,
    style_repr,
    slow_style_repr,
    fast_style_repr,
    style_memory_repr,
):
    style_success_anchor = resolve_style_success_anchor(
        style_repr,
        slow_style_repr,
        fast_style_summary=fast_style_repr,
        combined_style_summary=style_repr,
    )
    style_success_target_state = resolve_style_success_target_bank(
        output,
        memory_summary=style_memory_repr,
    )
    style_success_target = style_success_target_state.get("summary")
    output["style_success_target_source"] = str(
        style_success_target_state.get("source", "none")
    )
    output["style_success_target_base_source"] = str(
        style_success_target_state.get("base_source", "none")
    )
    anchor_bank = normalized_summary_batch(style_success_anchor)
    target_input = (
        style_success_target.detach()
        if isinstance(style_success_target, torch.Tensor)
        else style_success_target
    )
    target_flag_device = None
    for candidate in (target_input, style_success_target, style_success_anchor):
        if isinstance(candidate, torch.Tensor):
            target_flag_device = candidate.device
            break
    if target_flag_device is not None:
        memory_fallback_used = bool(
            style_success_target_state.get(
                "memory_fallback_used",
                style_success_target_state.get("memory_used", False),
            )
        )
        fallback_used = torch.tensor(
            1.0 if memory_fallback_used else 0.0,
            dtype=torch.float32,
            device=target_flag_device,
        )
        output["style_success_target_memory_fallback_used"] = fallback_used
        output["style_success_target_memory_used"] = fallback_used
    target_bank = normalized_summary_batch(target_input)
    if (
        not isinstance(anchor_bank, torch.Tensor)
        or not isinstance(target_bank, torch.Tensor)
        or tuple(anchor_bank.shape) != tuple(target_bank.shape)
    ):
        return

    total_style_success = None
    pair_weight = float(config.get("style_success_pair_weight", STYLE_SUCCESS_PAIR_WEIGHT))
    rank_weight = float(config.get("style_success_rank_weight", STYLE_SUCCESS_RANK_WEIGHT))
    pos_sim = (anchor_bank * target_bank).sum(dim=-1)
    if pair_weight > 0.0:
        total_style_success = (1.0 - pos_sim.mean()) * pair_weight

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
    output["style_success_negative_source"] = str(negative_state.get("source", "none"))
    _store_detached_scalar(
        output,
        "style_success_rank_source_scale",
        resolve_style_success_rank_source_scale(negative_state.get("source", "none"), config),
        device=anchor_bank.device,
    )
    for state_key, output_key in (
        ("label_valid_rows", "style_success_label_negative_row_frac"),
        ("label_negative_counts", "style_success_label_mean_negatives_per_row"),
        (
            "label_rows_below_proxy_min_count",
            "style_success_label_rows_below_proxy_min_count_frac",
        ),
        (
            "label_proxy_sufficient_rows",
            "style_success_label_proxy_sufficient_row_frac",
        ),
        ("proxy_valid_rows", "style_success_proxy_negative_row_frac"),
        ("proxy_supplemented_rows", "style_success_proxy_supplemented_row_frac"),
        ("proxy_augmented_rows", "style_success_proxy_augmented_row_frac"),
        ("valid_rows", "style_success_negative_row_frac"),
        ("negative_pair_density", "style_success_negative_pair_density"),
        ("negative_row_density", "style_success_negative_row_density"),
        ("mean_negatives_per_row", "style_success_mean_negatives_per_row"),
        ("mean_negatives_per_valid_row", "style_success_mean_negatives_per_valid_row"),
        ("label_negative_pair_density", "style_success_label_negative_pair_density"),
        ("proxy_negative_pair_density", "style_success_proxy_negative_pair_density"),
        ("proxy_backfill_row_frac", "style_success_proxy_backfill_row_frac"),
    ):
        value = negative_state.get(state_key)
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            value = value.float().mean()
        _store_detached_scalar(
            output,
            output_key,
            value,
            device=anchor_bank.device,
        )
    _store_detached_scalar(
        output,
        "style_success_proxy_min_batch",
        negative_state.get("proxy_min_batch", int(config.get("style_success_proxy_min_batch", 4))),
        device=anchor_bank.device,
    )
    _store_detached_scalar(
        output,
        "style_success_proxy_batch_gate_passed",
        1.0 if bool(negative_state.get("proxy_batch_gate_passed", True)) else 0.0,
        device=anchor_bank.device,
    )
    output["style_success_proxy_disabled_reason"] = str(
        negative_state.get("proxy_disabled_reason", "active")
    )
    _store_detached_scalar(
        output,
        "style_success_proxy_feature_count",
        negative_state.get("proxy_feature_count", 0),
        device=anchor_bank.device,
    )
    _store_detached_scalar(
        output,
        "style_success_proxy_informative_feature_count",
        negative_state.get("proxy_informative_feature_count", 0),
        device=anchor_bank.device,
    )
    output["style_success_proxy_informative_feature_names"] = list(
        negative_state.get("proxy_informative_feature_names", ()) or ()
    )

    if not isinstance(negative_state.get("negative_mask"), torch.Tensor) or rank_weight <= 0.0:
        return (
            total_style_success * style_success_supervision_scale(output, config)
            if total_style_success is not None
            else None
        )

    rank_mask = negative_state["negative_mask"] | torch.eye(
        anchor_bank.size(0),
        dtype=torch.bool,
        device=anchor_bank.device,
    )
    valid_rows = rank_mask.sum(dim=-1) > 1
    support_state = resolve_style_success_rank_support_state(
        negative_state,
        config,
        device=anchor_bank.device,
    )
    for state_key, output_key in (
        ("support_scale", "style_success_rank_support_scale"),
        ("effective_support", "style_success_rank_effective_support"),
        ("gate_passed", "style_success_rank_gate_passed"),
        ("rank_term_active", "style_success_rank_term_active"),
        ("min_negative_row_frac", "style_success_rank_min_negative_row_frac"),
        ("min_mean_negatives_per_row", "style_success_rank_min_mean_negatives_per_row"),
        ("min_effective_support", "style_success_rank_min_effective_support"),
        ("min_proxy_informative_features", "style_success_proxy_min_informative_features"),
        ("proxy_batch_scale", "style_success_proxy_batch_scale"),
        ("proxy_target_batch", "style_success_proxy_target_batch"),
        ("label_proxy_sufficient_row_frac", "style_success_label_proxy_sufficient_row_frac"),
        ("label_authority_frac", "style_success_label_authority_frac"),
        ("min_label_authority_row_frac", "style_success_label_authority_row_frac"),
        ("batch_composition_scale", "style_success_batch_composition_scale"),
    ):
        _store_detached_scalar(
            output,
            output_key,
            support_state.get(state_key),
            device=anchor_bank.device,
        )
    output["style_success_rank_disabled_reason"] = str(
        support_state.get("disabled_reason", "unknown")
    )
    if not bool(float(support_state.get("gate_passed", torch.tensor(0.0)).item()) > 0.0) or not valid_rows.any():
        return (
            total_style_success * style_success_supervision_scale(output, config)
            if total_style_success is not None
            else None
        )

    temperature = max(
        float(config.get("style_success_rank_temperature", STYLE_SUCCESS_RANK_TEMPERATURE)),
        1e-4,
    )
    logits = (anchor_bank @ target_bank.transpose(0, 1)) / temperature
    logits = logits.masked_fill(
        ~rank_mask,
        torch.finfo(logits.dtype).min,
    )
    targets = torch.arange(anchor_bank.size(0), device=anchor_bank.device)
    row_scale = torch.ones(
        anchor_bank.size(0),
        dtype=anchor_bank.dtype,
        device=anchor_bank.device,
    )
    scale_defaults = resolve_style_success_rank_scale_defaults(config)
    label_rank_scale = float(scale_defaults.get("label", 1.0))
    label_plus_proxy_backfill_scale = float(
        scale_defaults.get("label_plus_proxy_backfill", 0.75)
    )
    proxy_rank_scale = float(scale_defaults.get("proxy", 0.5))
    label_valid_rows = negative_state.get("label_valid_rows")
    proxy_valid_rows = negative_state.get("proxy_valid_rows")
    proxy_backfill_rows = negative_state.get("resolved_proxy_backfill_rows")
    if not isinstance(proxy_backfill_rows, torch.Tensor):
        proxy_backfill_rows = negative_state.get("proxy_augmented_rows")
    if not isinstance(proxy_backfill_rows, torch.Tensor):
        proxy_backfill_rows = negative_state.get("proxy_supplemented_rows")
    negative_source = str(negative_state.get("source", "none"))
    row_scale = row_scale * label_rank_scale
    proxy_only_rows = None
    if negative_source == "proxy":
        row_scale = row_scale.new_full(row_scale.shape, proxy_rank_scale)
        proxy_only_rows = valid_rows.clone()
    elif negative_source == "label_plus_proxy_backfill":
        proxy_backfilled_rows = None
        if isinstance(proxy_backfill_rows, torch.Tensor):
            proxy_backfilled_rows = proxy_backfill_rows.clone()
        proxy_only_rows = proxy_valid_rows.clone() if isinstance(proxy_valid_rows, torch.Tensor) else None
        if isinstance(label_valid_rows, torch.Tensor):
            if isinstance(proxy_only_rows, torch.Tensor):
                proxy_only_rows = proxy_only_rows & (~label_valid_rows)
        if isinstance(proxy_backfilled_rows, torch.Tensor):
            proxy_backfilled_rows = proxy_backfilled_rows & valid_rows
        if isinstance(proxy_backfilled_rows, torch.Tensor) and proxy_backfilled_rows.any():
            row_scale = torch.where(
                proxy_backfilled_rows,
                row_scale.new_full(row_scale.shape, label_plus_proxy_backfill_scale),
                row_scale,
            )
        if isinstance(proxy_only_rows, torch.Tensor) and proxy_only_rows.any():
            row_scale = torch.where(
                proxy_only_rows,
                row_scale.new_full(row_scale.shape, proxy_rank_scale),
                row_scale,
            )

    per_row_rank_loss = F.cross_entropy(
        logits[valid_rows],
        targets[valid_rows],
        reduction="none",
    )
    row_weight = row_scale[valid_rows].clamp_min(1.0e-6)
    rank_loss = (per_row_rank_loss * row_weight).sum() / row_weight.sum().clamp_min(1.0e-6)
    rank_loss = rank_loss * support_state["effective_support"]
    output["style_success_rank_row_scale_mean"] = row_scale[valid_rows].mean().detach()
    if isinstance(proxy_only_rows, torch.Tensor):
        output["style_success_proxy_only_negative_row_frac"] = (
            (proxy_only_rows & valid_rows).float().mean().detach()
        )
    total_style_success = (
        rank_loss * rank_weight
        if total_style_success is None
        else total_style_success + rank_loss * rank_weight
    )
    if total_style_success is not None:
        total_style_success = total_style_success * style_success_supervision_scale(output, config)
    return total_style_success


def _add_style_success_rank_loss(
    losses,
    output,
    sample,
    config,
    *,
    lambda_value,
    style_repr,
    slow_style_repr,
    fast_style_repr,
    style_memory_repr,
):
    if lambda_value <= 0.0:
        return
    total_style_success = _compute_style_success_control_loss(
        output,
        sample,
        config,
        style_repr=style_repr,
        slow_style_repr=slow_style_repr,
        fast_style_repr=fast_style_repr,
        style_memory_repr=style_memory_repr,
    )
    if total_style_success is not None:
        losses["style_success_rank"] = total_style_success * lambda_value


def _add_dynamic_timbre_budget_loss(losses, output, sample, config, *, lambda_value):
    if lambda_value <= 0.0:
        return
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
    relative_budget_weight = float(config.get("dynamic_timbre_budget_relative_weight", 0.0))
    style_decoder_residual = output.get("style_decoder_residual")
    slow_style_decoder_residual = output.get("slow_style_decoder_residual")
    style_owner_base_residual = _present(
        output,
        "style_owner_base_residual",
        "slow_style_decoder_residual",
        default=slow_style_decoder_residual,
    )
    style_owner_innovation_residual = output.get("style_owner_innovation_residual")
    dynamic_timbre_decoder_residual = _present(
        output,
        "dynamic_timbre_decoder_residual_prebudget",
        "dynamic_timbre_decoder_residual",
    )
    budget_terms = resolve_dynamic_timbre_budget_terms(
        dynamic_timbre_decoder_residual,
        style_residual=style_decoder_residual,
        slow_style_residual=slow_style_decoder_residual,
        style_base_residual=style_owner_base_residual,
        style_innovation_residual=style_owner_innovation_residual,
        padding_mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
        budget_ratio=budget_ratio,
        budget_margin=budget_margin,
        slow_style_weight=budget_slow_style_weight,
        budget_epsilon=budget_epsilon,
    )
    timbre_budget = budget_terms.get("timbre_energy")
    style_budget = budget_terms.get("style_energy")
    budget_support_weight = _resolve_dynamic_timbre_budget_support_weight(
        sample,
        output,
        config,
        timbre_budget,
    )
    if isinstance(budget_support_weight, torch.Tensor):
        output["dynamic_timbre_budget_support_weight"] = budget_support_weight.detach()
    valid_weight = _sequence_weight(
        _present(output, "dynamic_timbre_mask", "style_trace_mask"),
        reference=timbre_budget,
        boundary_mask=output.get("dynamic_timbre_boundary_mask"),
        frame_weight=budget_support_weight,
    )
    prebudget_term_weight = float(config.get("dynamic_timbre_budget_prebudget_term_weight", 1.0))
    mid_term_weight = float(config.get("dynamic_timbre_budget_mid_term_weight", 0.5))
    late_term_weight = float(config.get("dynamic_timbre_budget_late_term_weight", 0.75))
    budget_terms_reduced = []
    prebudget_term = _reduce_budget_overflow(
        timbre_budget,
        style_budget,
        budget_ratio=budget_ratio,
        budget_margin=budget_margin,
        weight=valid_weight,
        relative_weight=relative_budget_weight,
        budget_epsilon=budget_epsilon,
    )
    if prebudget_term is not None:
        output["dynamic_timbre_budget_prebudget_term"] = prebudget_term.detach()
        budget_terms_reduced.append((prebudget_term, prebudget_term_weight))
    stage_outputs = output.get("decoder_style_adapter_stages")
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
            stage_valid_weight = _sequence_weight(
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                reference=stage_timbre_budget,
                boundary_mask=output.get("dynamic_timbre_boundary_mask"),
                frame_weight=budget_support_weight,
            )
            stage_term = _reduce_budget_overflow(
                stage_timbre_budget,
                stage_style_budget,
                budget_ratio=budget_ratio,
                budget_margin=budget_margin,
                weight=stage_valid_weight,
                relative_weight=relative_budget_weight,
                budget_epsilon=budget_epsilon,
            )
            if stage_term is not None:
                output[f"dynamic_timbre_budget_{stage_name}_term"] = stage_term.detach()
                budget_terms_reduced.append((stage_term, mid_term_weight if stage_name == "mid" else late_term_weight))
    reduced_budget = _weighted_optional_scalars(*budget_terms_reduced)
    if reduced_budget is not None:
        losses["dynamic_timbre_budget"] = reduced_budget * lambda_value


def _add_style_timbre_runtime_overlap_loss(losses, output, sample, config, *, lambda_value):
    if lambda_value <= 0.0:
        return
    style_decoder_residual = output.get("style_decoder_residual")
    dynamic_timbre_prebudget = _present(
        output,
        "dynamic_timbre_decoder_residual_prebudget",
        "dynamic_timbre_decoder_residual",
    )
    if not isinstance(style_decoder_residual, torch.Tensor) or not isinstance(dynamic_timbre_prebudget, torch.Tensor):
        return
    support_weight = _resolve_dynamic_timbre_budget_support_weight(
        sample,
        output,
        config,
        dynamic_timbre_prebudget,
    )
    overlap_terms = masked_sequence_cosine(
        style_decoder_residual,
        dynamic_timbre_prebudget,
        mask=_present(output, "dynamic_timbre_mask", "style_trace_mask"),
        boundary_mask=output.get("dynamic_timbre_boundary_mask"),
        frame_weight=support_weight,
        absolute=resolve_style_success_bool_flag(
            config.get("style_timbre_runtime_overlap_use_abs", True),
            default=True,
        ),
        margin=float(config.get("style_timbre_runtime_overlap_margin", 0.10)),
    )
    overlap_penalty = overlap_terms.get("reduced")
    if overlap_penalty is not None:
        losses["style_timbre_runtime_overlap"] = overlap_penalty * lambda_value


def _add_pitch_residual_safe_loss(losses, output, sample, config, *, lambda_value):
    if lambda_value <= 0.0:
        return
    pitch_residual = output.get("style_to_pitch_residual")
    pitch_residual_target = output.get("style_to_pitch_residual_target")
    if not isinstance(pitch_residual, torch.Tensor):
        return
    voiced_weight = None
    voiced_mask = output.get("style_to_pitch_residual_voiced_mask")
    if isinstance(voiced_mask, torch.Tensor):
        if voiced_mask.dim() == 3 and voiced_mask.size(-1) == 1:
            voiced_mask = voiced_mask.squeeze(-1)
        if voiced_mask.dim() == 2 and tuple(voiced_mask.shape) == tuple(pitch_residual.shape):
            voiced_weight = voiced_mask.to(device=pitch_residual.device, dtype=pitch_residual.dtype).clamp(0.0, 1.0)
    if voiced_weight is None:
        sample_uv = sample.get("uv")
        if (
            isinstance(sample_uv, torch.Tensor)
            and sample_uv.dim() == 2
            and tuple(sample_uv.shape) == tuple(pitch_residual.shape)
        ):
            voiced_weight = (1.0 - sample_uv.float()).clamp(0.0, 1.0).to(pitch_residual.device)
    residual_weight = _sequence_weight(
        _present(
            output,
            "style_to_pitch_residual_mask",
            "dynamic_timbre_mask",
            "style_trace_mask",
        ),
        reference=pitch_residual,
        voiced_weight=voiced_weight,
    )
    residual_total = None
    huber_delta = float(config.get("pitch_residual_huber_delta", 0.02))
    budget_weight = float(config.get("pitch_residual_budget_weight", 0.15))
    budget_margin = float(config.get("pitch_residual_budget_margin", 0.015))
    has_aligned_target = (
        isinstance(pitch_residual_target, torch.Tensor)
        and tuple(pitch_residual_target.shape) == tuple(pitch_residual.shape)
    )
    contour_weight = None
    smoothing_valid_mask = output.get("style_to_pitch_residual_smoothing_valid_mask")
    if isinstance(smoothing_valid_mask, torch.Tensor):
        if smoothing_valid_mask.dim() == 3 and smoothing_valid_mask.size(-1) == 1:
            smoothing_valid_mask = smoothing_valid_mask.squeeze(-1)
        if smoothing_valid_mask.dim() == 2 and tuple(smoothing_valid_mask.shape) == tuple(pitch_residual.shape):
            contour_weight = smoothing_valid_mask.to(
                device=pitch_residual.device,
                dtype=pitch_residual.dtype,
            ).clamp(0.0, 1.0)
    if contour_weight is None and isinstance(residual_weight, torch.Tensor):
        if residual_weight.dim() == 2 and tuple(residual_weight.shape) == tuple(pitch_residual.shape):
            contour_weight = residual_weight
    if has_aligned_target:
        target_ref = pitch_residual_target.detach()
        align_term = _weighted_smooth_l1(
            pitch_residual,
            target_ref,
            residual_weight,
            beta=huber_delta,
        )
        if align_term is not None:
            residual_total = align_term
            output["pitch_residual_safe_align_term"] = align_term.detach()
        if budget_weight > 0.0:
            overflow = F.relu(pitch_residual.abs() - target_ref.abs() - max(0.0, budget_margin))
            overflow_term = _weighted_mean(overflow, residual_weight)
            if overflow_term is not None:
                output["pitch_residual_safe_budget_overflow_term"] = (overflow_term * budget_weight).detach()
                residual_total = (
                    overflow_term * budget_weight
                    if residual_total is None
                    else residual_total + overflow_term * budget_weight
                )
    else:
        zero_fallback_weight = float(config.get("pitch_residual_zero_fallback_weight", 0.05))
        if zero_fallback_weight > 0.0:
            zero_anchor = _weighted_mean(pitch_residual.pow(2), residual_weight)
            if zero_anchor is not None:
                output["pitch_residual_safe_zero_fallback_term"] = (zero_anchor * zero_fallback_weight).detach()
                residual_total = zero_anchor * zero_fallback_weight
    if pitch_residual.size(1) > 1:
        pitch_delta = pitch_residual[:, 1:] - pitch_residual[:, :-1]
        smooth_weight = None
        if isinstance(contour_weight, torch.Tensor) and contour_weight.size(1) == pitch_residual.size(1):
            smooth_weight = contour_weight[:, 1:] * contour_weight[:, :-1]
        elif (
            isinstance(residual_weight, torch.Tensor)
            and residual_weight.dim() == 2
            and residual_weight.size(1) == pitch_residual.size(1)
        ):
            smooth_weight = residual_weight[:, 1:] * residual_weight[:, :-1]
        if has_aligned_target:
            target_delta = pitch_residual_target[:, 1:] - pitch_residual_target[:, :-1]
            smooth_term = _weighted_smooth_l1(
                pitch_delta,
                target_delta,
                smooth_weight,
                beta=huber_delta,
            )
        else:
            smooth_term = _weighted_mean(pitch_delta.abs(), smooth_weight)
        smooth_scale = float(config.get("pitch_residual_smooth_weight", 0.10))
        if smooth_term is not None and smooth_scale > 0.0:
            output["pitch_residual_safe_slope_term"] = (smooth_term * smooth_scale).detach()
            residual_total = (
                smooth_term * smooth_scale
                if residual_total is None
                else residual_total + smooth_term * smooth_scale
            )
    output["pitch_residual_safe_target_available"] = bool(has_aligned_target)
    if residual_total is not None:
        losses["pitch_residual_safe"] = residual_total * lambda_value


def _add_decoder_late_owner_losses(losses, output, config, *, late_owner_lambda, late_anchor_lambda):
    if late_owner_lambda <= 0.0 and late_anchor_lambda <= 0.0:
        return
    stage_outputs = output.get("decoder_style_adapter_stages")
    late_stage = stage_outputs.get("late") if isinstance(stage_outputs, dict) else None
    if not isinstance(late_stage, dict):
        return
    late_style_map = _stage_energy_map(
        late_stage,
        branch_keys=("global_style_delta", "style_trace_delta"),
        combine_before_energy=True,
    )
    late_timbre_map = _stage_energy_map(late_stage, branch_keys=("dynamic_timbre_delta",))
    late_anchor_map = _stage_energy_map(late_stage, branch_keys=("global_timbre_delta",))
    late_reference_map = (
        late_timbre_map
        if isinstance(late_timbre_map, torch.Tensor)
        else late_anchor_map
        if isinstance(late_anchor_map, torch.Tensor)
        else late_style_map
    )
    late_valid_weight = _sequence_weight(
        _present(
            output,
            "dynamic_timbre_mask",
            "style_decoder_residual_mask",
            "style_trace_mask",
        ),
        reference=late_reference_map,
    )
    late_style_energy = _weighted_mean(late_style_map, late_valid_weight)
    late_timbre_energy = _weighted_mean(late_timbre_map, late_valid_weight)
    late_anchor_energy = _weighted_mean(late_anchor_map, late_valid_weight)
    if late_owner_lambda > 0.0:
        owner_ratio = float(config.get("decoder_late_timbre_owner_ratio", 0.50))
        owner_margin = float(config.get("decoder_late_owner_margin", 0.0))
        reduced_late_owner = _reduce_budget_overflow(
            late_timbre_map,
            late_style_map,
            budget_ratio=owner_ratio,
            budget_margin=owner_margin,
            weight=late_valid_weight,
        )
        if reduced_late_owner is None and isinstance(late_style_energy, torch.Tensor) and isinstance(late_timbre_energy, torch.Tensor):
            reduced_late_owner = _reduce_budget_overflow(
                late_timbre_energy,
                late_style_energy,
                budget_ratio=owner_ratio,
                budget_margin=owner_margin,
            )
        if reduced_late_owner is not None:
            output["decoder_late_owner_overflow_term"] = reduced_late_owner.detach()
        full_owner_style_map = _sequence_abs_mean(output.get("style_decoder_residual"))
        if not isinstance(full_owner_style_map, torch.Tensor):
            full_owner_style_map = late_style_map
        full_owner_energy = None
        if isinstance(full_owner_style_map, torch.Tensor):
            full_owner_valid_weight = _sequence_weight(
                _present(
                    output,
                    "style_decoder_residual_mask",
                    "dynamic_timbre_mask",
                    "style_trace_mask",
                ),
                reference=full_owner_style_map,
            )
            full_owner_energy = _weighted_mean(full_owner_style_map, full_owner_valid_weight)
        if full_owner_energy is None:
            full_owner_energy = _tensor_abs_mean(output.get("style_decoder_residual"))
        if isinstance(full_owner_energy, torch.Tensor):
            output["decoder_late_owner_full_style_energy"] = full_owner_energy.detach()
        if isinstance(late_style_energy, torch.Tensor):
            output["decoder_late_owner_late_style_energy"] = late_style_energy.detach()
        style_floor_term = None
        style_floor_ratio = float(config.get("decoder_late_style_floor_ratio", 0.15))
        style_floor_margin = float(config.get("decoder_late_style_floor_margin", 0.0))
        style_floor_weight = float(config.get("decoder_late_style_floor_weight", 0.35))
        if (
            style_floor_weight > 0.0
            and style_floor_ratio > 0.0
            and isinstance(full_owner_energy, torch.Tensor)
            and isinstance(late_style_energy, torch.Tensor)
        ):
            style_floor_target = style_floor_ratio * full_owner_energy.detach() + style_floor_margin
            style_floor_term = F.relu(style_floor_target - late_style_energy)
            output["decoder_late_owner_style_floor_term"] = style_floor_term.detach()
        reduced_late_owner_total = _sum_optional_scalars(
            reduced_late_owner,
            None if style_floor_term is None else style_floor_term * max(style_floor_weight, 0.0),
        )
        if reduced_late_owner_total is not None:
            losses["decoder_late_owner"] = reduced_late_owner_total * late_owner_lambda
    if late_anchor_lambda > 0.0 and isinstance(late_anchor_energy, torch.Tensor):
        anchor_reference_map = late_style_map
        if not isinstance(anchor_reference_map, torch.Tensor):
            anchor_reference_map = _sequence_abs_mean(output.get("style_decoder_residual"))
        anchor_ratio = float(config.get("decoder_late_anchor_budget_ratio", 0.35))
        anchor_floor = float(config.get("decoder_late_anchor_budget_floor", 0.0))
        reduced_anchor_budget = _reduce_budget_overflow(
            late_anchor_map,
            anchor_reference_map,
            budget_ratio=anchor_ratio,
            budget_margin=anchor_floor,
            weight=late_valid_weight,
        )
        if reduced_anchor_budget is None:
            anchor_reference = _weighted_mean(anchor_reference_map, late_valid_weight)
            if not isinstance(anchor_reference, torch.Tensor):
                anchor_reference = _tensor_abs_mean(output.get("style_decoder_residual"))
            if isinstance(anchor_reference, torch.Tensor):
                reduced_anchor_budget = _reduce_budget_overflow(
                    late_anchor_energy,
                    anchor_reference,
                    budget_ratio=anchor_ratio,
                    budget_margin=anchor_floor,
                )
        if reduced_anchor_budget is not None:
            losses["decoder_late_anchor_budget"] = reduced_anchor_budget * late_anchor_lambda


def add_style_timbre_regularization_losses(losses, output, sample, config):
    control_loss_profile = str(config.get("control_loss_profile", "mainline_minimal") or "mainline_minimal").strip().lower()
    minimal_mainline = control_loss_profile in {"mainline_minimal", "minimal", "core", "mainline"}
    minimal_allowed = set(MAINLINE_MINIMAL_CONTROL_LAMBDAS)
    optional_runtime_regularizers = {
        "lambda_style_timbre_runtime_overlap",
    }

    def _lambda(key):
        if minimal_mainline and key not in minimal_allowed and key not in optional_runtime_regularizers:
            return 0.0
        return float(config.get(key, 0.0))

    style_repr = _style_representation(output)
    fast_style_repr = _fast_style_representation(output)
    slow_style_repr = _slow_style_representation(output)
    style_memory_repr = _masked_sequence_mean(
        output.get("style_trace_memory"),
        output.get("style_trace_memory_mask"),
    )
    timbre_repr = _timbre_representation(output)
    global_timbre_anchor = _global_timbre_anchor(output)
    global_style_summary = _global_style_summary(output)
    style_query_repr = _masked_sequence_mean(
        output.get("style_query_inp"),
        output.get("style_trace_mask"),
    )
    timbre_query_repr = _masked_sequence_mean(
        output.get("timbre_query_inp"),
        _present(output, "dynamic_timbre_mask", "style_trace_mask"),
    )

    style_trace_consistency_lambda = _lambda("lambda_style_trace_consistency")
    if style_trace_consistency_lambda > 0:
        consistency = _cosine_distance(style_repr, style_memory_repr)
        if consistency is not None:
            losses["style_trace_consistency"] = consistency * style_trace_consistency_lambda

    timbre_anchor_cos_lambda = _lambda("lambda_timbre_anchor_cosine")
    if timbre_anchor_cos_lambda > 0:
        timbre_anchor_cos = _cosine_distance(timbre_repr, global_timbre_anchor)
        if timbre_anchor_cos is not None:
            losses["timbre_anchor_cosine"] = timbre_anchor_cos * timbre_anchor_cos_lambda

    output_identity_lambda = _lambda("lambda_output_identity_cosine")
    if output_identity_lambda > 0:
        output_identity_embed = _summary_vector(output.get("output_identity_embed"))
        output_identity_target = _summary_vector(output.get("output_identity_target_embed"))
        output_identity_target_source = output.get("output_identity_target_source", None)
        if output_identity_target is None:
            output_identity_target = _summary_vector(output.get("output_identity_reference_target"))
            if output_identity_target is not None:
                output_identity_target_source = "output_identity_reference_target"
        if output_identity_target is None:
            output_identity_target = _summary_vector(
                _present(output, "output_identity_anchor_target", "global_timbre_anchor")
            )
            if output_identity_target is not None:
                output_identity_target_source = "output_identity_anchor_target"
        if isinstance(output_identity_target, torch.Tensor):
            output["output_identity_target_used_for_loss"] = output_identity_target.detach()
        if output_identity_target_source is not None:
            output["output_identity_target_source_resolved_for_loss"] = str(output_identity_target_source)
        output_identity_cos = _cosine_distance(output_identity_embed, output_identity_target)
        if output_identity_cos is not None:
            losses["output_identity_cosine"] = output_identity_cos * output_identity_lambda

    style_timbre_dis_lambda = _lambda("lambda_style_timbre_disentangle")
    if style_timbre_dis_lambda > 0:
        style_timbre_dis = _mean_abs_cosine(style_repr, global_timbre_anchor)
        if style_timbre_dis is not None:
            losses["style_timbre_dis"] = style_timbre_dis * style_timbre_dis_lambda

    style_summary_align_lambda = _lambda("lambda_global_style_summary_align")
    if style_summary_align_lambda > 0:
        style_summary_align = _cosine_distance(style_repr, global_style_summary)
        if style_summary_align is not None:
            losses["global_style_summary_align"] = style_summary_align * style_summary_align_lambda

    style_dynamic_timbre_dis_lambda = _lambda("lambda_style_dynamic_timbre_disentangle")
    if style_dynamic_timbre_dis_lambda > 0:
        style_dynamic_timbre_dis = _mean_abs_cosine(style_repr, timbre_repr)
        if style_dynamic_timbre_dis is not None:
            losses["style_dynamic_timbre_dis"] = (
                style_dynamic_timbre_dis * style_dynamic_timbre_dis_lambda
            )

    query_dis_lambda = _lambda("lambda_style_timbre_query_disentangle")
    if query_dis_lambda > 0:
        query_dis = _mean_abs_cosine(style_query_repr, timbre_query_repr)
        if query_dis is not None:
            losses["style_timbre_query_dis"] = query_dis * query_dis_lambda

    style_query_var_lambda = _lambda("lambda_style_query_var")
    if style_query_var_lambda > 0:
        style_query_var = _variance_floor_penalty(
            style_query_repr,
            target=float(config.get("style_query_var_target", 0.05)),
        )
        if style_query_var is not None:
            losses["style_query_var"] = style_query_var * style_query_var_lambda

    timbre_query_var_lambda = _lambda("lambda_timbre_query_var")
    if timbre_query_var_lambda > 0:
        timbre_query_var = _variance_floor_penalty(
            timbre_query_repr,
            target=float(config.get("timbre_query_var_target", 0.05)),
        )
        if timbre_query_var is not None:
            losses["timbre_query_var"] = timbre_query_var * timbre_query_var_lambda

    slow_style_summary_align_lambda = _lambda("lambda_slow_style_summary_align")
    if slow_style_summary_align_lambda > 0:
        slow_style_summary_align = _cosine_distance(slow_style_repr, global_style_summary)
        if slow_style_summary_align is not None:
            losses["slow_style_summary_align"] = (
                slow_style_summary_align * slow_style_summary_align_lambda
            )

    style_success_rank_lambda = _lambda("lambda_style_success_rank")
    _add_style_success_rank_loss(
        losses,
        output,
        sample,
        config,
        lambda_value=style_success_rank_lambda,
        style_repr=style_repr,
        slow_style_repr=slow_style_repr,
        fast_style_repr=fast_style_repr,
        style_memory_repr=style_memory_repr,
    )

    dynamic_timbre_gate_lambda = _lambda("lambda_dynamic_timbre_gate")
    if dynamic_timbre_gate_lambda > 0:
        gate_mean = _gate_mean(
            output.get("dynamic_timbre_gate"),
            _present(output, "dynamic_timbre_mask", "style_trace_mask"),
        )
        if gate_mean is not None:
            gate_target = float(config.get("dynamic_timbre_gate_target", 0.6))
            losses["dynamic_timbre_gate_reg"] = (gate_mean - gate_target).abs() * dynamic_timbre_gate_lambda

    dynamic_timbre_budget_lambda = _lambda("lambda_dynamic_timbre_budget")
    _add_dynamic_timbre_budget_loss(
        losses,
        output,
        sample,
        config,
        lambda_value=dynamic_timbre_budget_lambda,
    )

    style_timbre_runtime_overlap_lambda = _lambda("lambda_style_timbre_runtime_overlap")
    _add_style_timbre_runtime_overlap_loss(
        losses,
        output,
        sample,
        config,
        lambda_value=style_timbre_runtime_overlap_lambda,
    )

    pitch_residual_safe_lambda = _lambda("lambda_pitch_residual_safe")
    _add_pitch_residual_safe_loss(
        losses,
        output,
        sample,
        config,
        lambda_value=pitch_residual_safe_lambda,
    )

    dynamic_timbre_boundary_lambda = _lambda("lambda_dynamic_timbre_boundary")
    if dynamic_timbre_boundary_lambda > 0:
        dynamic_timbre_decoder_residual = output.get("dynamic_timbre_decoder_residual")
        boundary_mask = output.get("dynamic_timbre_boundary_mask")
        if isinstance(dynamic_timbre_decoder_residual, torch.Tensor) and isinstance(boundary_mask, torch.Tensor):
            boundary_penalty = _sequence_abs_mean(dynamic_timbre_decoder_residual)
            if isinstance(boundary_penalty, torch.Tensor):
                valid_weight = _sequence_weight(
                    _present(output, "dynamic_timbre_mask", "style_trace_mask"),
                    reference=boundary_penalty,
                )
                if isinstance(boundary_mask, torch.Tensor):
                    if boundary_mask.dim() == 3 and boundary_mask.size(-1) == 1:
                        boundary_mask = boundary_mask.squeeze(-1)
                    if boundary_mask.dim() == 2 and tuple(boundary_mask.shape) == tuple(boundary_penalty.shape):
                        boundary_weight = boundary_mask.to(boundary_penalty.device, dtype=boundary_penalty.dtype).clamp(0.0, 1.0)
                        if isinstance(valid_weight, torch.Tensor):
                            boundary_weight = boundary_weight * valid_weight
                        reduced_boundary = _weighted_mean(boundary_penalty, boundary_weight)
                        if reduced_boundary is not None:
                            losses["dynamic_timbre_boundary"] = (
                                reduced_boundary * dynamic_timbre_boundary_lambda
                            )

    dynamic_timbre_anchor_lambda = _lambda("lambda_dynamic_timbre_anchor")
    if dynamic_timbre_anchor_lambda > 0:
        anchor_shift = output.get("dynamic_timbre_anchor_shift")
        if isinstance(anchor_shift, torch.Tensor):
            losses["dynamic_timbre_anchor"] = anchor_shift.abs().mean() * dynamic_timbre_anchor_lambda

    gate_rank_lambda = _lambda("lambda_gate_rank")
    if gate_rank_lambda > 0:
        stage_outputs = output.get("decoder_style_adapter_stages")
        style_gate = None
        timbre_gate = None
        if isinstance(stage_outputs, dict):
            mid_stage = stage_outputs.get("mid") if isinstance(stage_outputs.get("mid"), dict) else None
            late_stage = stage_outputs.get("late") if isinstance(stage_outputs.get("late"), dict) else None
            style_gate = _mean_optional_scalars(
                _gate_mean(mid_stage.get("slow_style_gate")) if isinstance(mid_stage, dict) else None,
                _gate_mean(late_stage.get("global_style_gate")) if isinstance(late_stage, dict) else None,
                _gate_mean(late_stage.get("style_trace_gate")) if isinstance(late_stage, dict) else None,
            )
            timbre_gate = _mean_optional_scalars(
                _gate_mean(mid_stage.get("dynamic_timbre_gate")) if isinstance(mid_stage, dict) else None,
                _gate_mean(late_stage.get("dynamic_timbre_gate")) if isinstance(late_stage, dict) else None,
            )
        if style_gate is None:
            style_gate = _gate_mean(output.get("style_trace_gate"), output.get("style_trace_mask"))
        if timbre_gate is None:
            timbre_gate = _gate_mean(
                output.get("dynamic_timbre_gate"),
                _present(output, "dynamic_timbre_mask", "style_trace_mask"),
            )
        if isinstance(style_gate, torch.Tensor) and isinstance(timbre_gate, torch.Tensor):
            gate_rank_ratio = float(config.get("gate_rank_ratio", 0.60))
            losses["gate_rank"] = F.relu(timbre_gate - gate_rank_ratio * style_gate) * gate_rank_lambda

    late_owner_lambda = _lambda("lambda_decoder_late_owner")
    late_anchor_lambda = _lambda("lambda_decoder_late_anchor_budget")
    _add_decoder_late_owner_losses(
        losses,
        output,
        config,
        late_owner_lambda=late_owner_lambda,
        late_anchor_lambda=late_anchor_lambda,
    )
