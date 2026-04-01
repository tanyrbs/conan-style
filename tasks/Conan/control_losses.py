import torch
import torch.nn.functional as F

from modules.Conan.style_trace_utils import resolve_combined_style_trace


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


def _style_representation(output):
    combined_style_trace, combined_style_trace_mask = resolve_combined_style_trace(output)
    return _masked_sequence_mean(
        combined_style_trace,
        combined_style_trace_mask,
    )


def _timbre_representation(output):
    return _masked_sequence_mean(
        output.get("dynamic_timbre"),
        output.get("dynamic_timbre_mask"),
    )


def _global_timbre_anchor(output):
    return _summary_vector(output.get("global_timbre_anchor"))


def _global_style_summary(output):
    return _summary_vector(
        output.get("global_style_summary_runtime", output.get("global_style_summary"))
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


def _sequence_weight(mask=None, *, reference=None, boundary_mask=None, voiced_weight=None):
    device = None
    dtype = torch.float32
    if isinstance(reference, torch.Tensor):
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        shape = reference.shape[:2]
    else:
        shape = None

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

    if isinstance(voiced_weight, torch.Tensor):
        if voiced_weight.dim() == 3 and voiced_weight.size(-1) == 1:
            voiced_weight = voiced_weight.squeeze(-1)
        if voiced_weight.dim() == 2 and tuple(voiced_weight.shape) == tuple(weight.shape):
            weight = weight * voiced_weight.to(device=weight.device, dtype=weight.dtype).clamp(0.0, 1.0)

    if isinstance(boundary_mask, torch.Tensor):
        if boundary_mask.dim() == 3 and boundary_mask.size(-1) == 1:
            boundary_mask = boundary_mask.squeeze(-1)
        if boundary_mask.dim() == 2 and tuple(boundary_mask.shape) == tuple(weight.shape):
            weight = weight * (1.0 - boundary_mask.to(device=weight.device, dtype=weight.dtype).clamp(0.0, 1.0))
    return weight


def _weighted_mean(value, weight=None):
    if not isinstance(value, torch.Tensor):
        return None
    if not isinstance(weight, torch.Tensor):
        return value.mean()
    if tuple(value.shape) != tuple(weight.shape):
        return None
    denom = weight.sum().clamp_min(1.0)
    return (value * weight).sum() / denom


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


def _adapter_stage_energy_map(stage_outputs, *, stage_names=None, branch_keys=()):
    if not isinstance(stage_outputs, dict):
        return None
    if stage_names is not None:
        stage_names = {str(name).lower() for name in stage_names}
    total = None
    for stage_name, stage_meta in stage_outputs.items():
        if stage_names is not None and str(stage_name).lower() not in stage_names:
            continue
        if not isinstance(stage_meta, dict):
            continue
        branch_total = _sum_optional_maps(
            *[_sequence_abs_mean(stage_meta.get(key)) for key in branch_keys]
        )
        if isinstance(branch_total, torch.Tensor):
            total = branch_total if total is None else _sum_optional_maps(total, branch_total)
    return total


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


def add_style_timbre_regularization_losses(losses, output, sample, config):
    control_loss_profile = str(config.get("control_loss_profile", "mainline_minimal") or "mainline_minimal").strip().lower()
    minimal_mainline = control_loss_profile in {"mainline_minimal", "minimal", "core", "mainline"}
    minimal_allowed = {
        "lambda_style_trace_consistency",
        "lambda_output_identity_cosine",
        "lambda_dynamic_timbre_budget",
        "lambda_decoder_late_owner",
    }

    def _lambda(key):
        if minimal_mainline and key not in minimal_allowed:
            return 0.0
        return float(config.get(key, 0.0))

    style_repr = _style_representation(output)
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
        output.get("dynamic_timbre_mask", output.get("style_trace_mask")),
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
        output_identity_target = _summary_vector(
            output.get("output_identity_anchor_target", output.get("global_timbre_anchor"))
        )
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

    dynamic_timbre_gate_lambda = _lambda("lambda_dynamic_timbre_gate")
    if dynamic_timbre_gate_lambda > 0:
        gate_mean = _gate_mean(
            output.get("dynamic_timbre_gate"),
            output.get("dynamic_timbre_mask"),
        )
        if gate_mean is not None:
            gate_target = float(config.get("dynamic_timbre_gate_target", 0.6))
            losses["dynamic_timbre_gate_reg"] = (gate_mean - gate_target).abs() * dynamic_timbre_gate_lambda

    dynamic_timbre_budget_lambda = _lambda("lambda_dynamic_timbre_budget")
    if dynamic_timbre_budget_lambda > 0:
        stage_outputs = output.get("decoder_style_adapter_stages")
        style_budget = _adapter_stage_energy_map(
            stage_outputs,
            stage_names=("mid", "late"),
            branch_keys=("global_style_delta", "slow_style_delta", "style_trace_delta"),
        )
        timbre_budget = _adapter_stage_energy_map(
            stage_outputs,
            stage_names=("mid", "late"),
            branch_keys=("dynamic_timbre_delta",),
        )
        if style_budget is None or timbre_budget is None:
            style_decoder_residual = output.get("style_decoder_residual")
            dynamic_timbre_decoder_residual = output.get("dynamic_timbre_decoder_residual")
            if isinstance(style_decoder_residual, torch.Tensor):
                style_budget = _sequence_abs_mean(style_decoder_residual.detach())
            if isinstance(dynamic_timbre_decoder_residual, torch.Tensor):
                timbre_budget = _sequence_abs_mean(dynamic_timbre_decoder_residual)
        if isinstance(style_budget, torch.Tensor) and isinstance(timbre_budget, torch.Tensor):
            budget_ratio = float(config.get("dynamic_timbre_budget_ratio", 0.55))
            budget_margin = float(config.get("dynamic_timbre_budget_margin", 0.02))
            style_budget_reference = style_budget.detach()
            voiced_weight = None
            sample_uv = sample.get("uv")
            if isinstance(sample_uv, torch.Tensor) and sample_uv.dim() == 2 and tuple(sample_uv.shape) == tuple(timbre_budget.shape):
                voiced_weight = (1.0 - sample_uv.float()).clamp(0.0, 1.0).to(timbre_budget.device)
            valid_weight = _sequence_weight(
                output.get("dynamic_timbre_mask", output.get("style_trace_mask")),
                reference=timbre_budget,
                boundary_mask=output.get("dynamic_timbre_boundary_mask"),
                voiced_weight=voiced_weight,
            )
            local_budget = F.relu(timbre_budget - budget_ratio * style_budget_reference - budget_margin)
            reduced_budget = _weighted_mean(local_budget, valid_weight)
            if reduced_budget is not None:
                losses["dynamic_timbre_budget"] = reduced_budget * dynamic_timbre_budget_lambda

    dynamic_timbre_boundary_lambda = _lambda("lambda_dynamic_timbre_boundary")
    if dynamic_timbre_boundary_lambda > 0:
        dynamic_timbre_decoder_residual = output.get("dynamic_timbre_decoder_residual")
        boundary_mask = output.get("dynamic_timbre_boundary_mask")
        if isinstance(dynamic_timbre_decoder_residual, torch.Tensor) and isinstance(boundary_mask, torch.Tensor):
            boundary_penalty = _sequence_abs_mean(dynamic_timbre_decoder_residual)
            if isinstance(boundary_penalty, torch.Tensor):
                nonboundary_weight = _sequence_weight(
                    output.get("dynamic_timbre_mask", output.get("style_trace_mask")),
                    reference=boundary_penalty,
                )
                if isinstance(boundary_mask, torch.Tensor):
                    if boundary_mask.dim() == 3 and boundary_mask.size(-1) == 1:
                        boundary_mask = boundary_mask.squeeze(-1)
                    if boundary_mask.dim() == 2 and tuple(boundary_mask.shape) == tuple(boundary_penalty.shape):
                        boundary_weight = boundary_mask.to(boundary_penalty.device, dtype=boundary_penalty.dtype).clamp(0.0, 1.0)
                        if isinstance(nonboundary_weight, torch.Tensor):
                            boundary_weight = boundary_weight * nonboundary_weight
                        reduced_boundary = _weighted_mean(boundary_penalty, boundary_weight)
                        if reduced_boundary is not None:
                            losses["dynamic_timbre_boundary"] = (
                                reduced_boundary * dynamic_timbre_boundary_lambda
                            )

    late_owner_lambda = _lambda("lambda_decoder_late_owner")
    late_anchor_lambda = _lambda("lambda_decoder_late_anchor_budget")
    if late_owner_lambda > 0 or late_anchor_lambda > 0:
        stage_outputs = output.get("decoder_style_adapter_stages")
        late_stage = stage_outputs.get("late") if isinstance(stage_outputs, dict) else None
        if isinstance(late_stage, dict):
            late_style_energy = _sum_optional_scalars(
                _tensor_abs_mean(late_stage.get("global_style_delta")),
                _tensor_abs_mean(late_stage.get("slow_style_delta")),
                _tensor_abs_mean(late_stage.get("style_trace_delta")),
            )
            late_timbre_energy = _tensor_abs_mean(late_stage.get("dynamic_timbre_delta"))
            late_anchor_energy = _tensor_abs_mean(late_stage.get("global_timbre_delta"))
            if late_owner_lambda > 0 and isinstance(late_style_energy, torch.Tensor) and isinstance(late_timbre_energy, torch.Tensor):
                owner_ratio = float(config.get("decoder_late_timbre_owner_ratio", 0.55))
                losses["decoder_late_owner"] = (
                    F.relu(late_timbre_energy - owner_ratio * late_style_energy) * late_owner_lambda
                )
            if late_anchor_lambda > 0 and isinstance(late_anchor_energy, torch.Tensor):
                anchor_reference = late_style_energy
                if not isinstance(anchor_reference, torch.Tensor):
                    anchor_reference = _tensor_abs_mean(output.get("style_decoder_residual"))
                if isinstance(anchor_reference, torch.Tensor):
                    anchor_ratio = float(config.get("decoder_late_anchor_budget_ratio", 0.35))
                    anchor_floor = float(config.get("decoder_late_anchor_budget_floor", 0.0))
                    losses["decoder_late_anchor_budget"] = (
                        F.relu(late_anchor_energy - anchor_ratio * anchor_reference - anchor_floor)
                        * late_anchor_lambda
                    )
