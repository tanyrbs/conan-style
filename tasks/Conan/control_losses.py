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

    style_trace_consistency_lambda = float(config.get("lambda_style_trace_consistency", 0.0))
    if style_trace_consistency_lambda > 0:
        consistency = _cosine_distance(style_repr, style_memory_repr)
        if consistency is not None:
            losses["style_trace_consistency"] = consistency * style_trace_consistency_lambda

    timbre_anchor_cos_lambda = float(config.get("lambda_timbre_anchor_cosine", 0.0))
    if timbre_anchor_cos_lambda > 0:
        timbre_anchor_cos = _cosine_distance(timbre_repr, global_timbre_anchor)
        if timbre_anchor_cos is not None:
            losses["timbre_anchor_cosine"] = timbre_anchor_cos * timbre_anchor_cos_lambda

    style_timbre_dis_lambda = float(config.get("lambda_style_timbre_disentangle", 0.0))
    if style_timbre_dis_lambda > 0:
        style_timbre_dis = _mean_abs_cosine(style_repr, global_timbre_anchor)
        if style_timbre_dis is not None:
            losses["style_timbre_dis"] = style_timbre_dis * style_timbre_dis_lambda

    style_summary_align_lambda = float(config.get("lambda_global_style_summary_align", 0.0))
    if style_summary_align_lambda > 0:
        style_summary_align = _cosine_distance(style_repr, global_style_summary)
        if style_summary_align is not None:
            losses["global_style_summary_align"] = style_summary_align * style_summary_align_lambda

    style_dynamic_timbre_dis_lambda = float(config.get("lambda_style_dynamic_timbre_disentangle", 0.0))
    if style_dynamic_timbre_dis_lambda > 0:
        style_dynamic_timbre_dis = _mean_abs_cosine(style_repr, timbre_repr)
        if style_dynamic_timbre_dis is not None:
            losses["style_dynamic_timbre_dis"] = (
                style_dynamic_timbre_dis * style_dynamic_timbre_dis_lambda
            )

    query_dis_lambda = float(config.get("lambda_style_timbre_query_disentangle", 0.0))
    if query_dis_lambda > 0:
        query_dis = _mean_abs_cosine(style_query_repr, timbre_query_repr)
        if query_dis is not None:
            losses["style_timbre_query_dis"] = query_dis * query_dis_lambda

    style_query_var_lambda = float(config.get("lambda_style_query_var", 0.0))
    if style_query_var_lambda > 0:
        style_query_var = _variance_floor_penalty(
            style_query_repr,
            target=float(config.get("style_query_var_target", 0.05)),
        )
        if style_query_var is not None:
            losses["style_query_var"] = style_query_var * style_query_var_lambda

    timbre_query_var_lambda = float(config.get("lambda_timbre_query_var", 0.0))
    if timbre_query_var_lambda > 0:
        timbre_query_var = _variance_floor_penalty(
            timbre_query_repr,
            target=float(config.get("timbre_query_var_target", 0.05)),
        )
        if timbre_query_var is not None:
            losses["timbre_query_var"] = timbre_query_var * timbre_query_var_lambda

    slow_style_summary_align_lambda = float(config.get("lambda_slow_style_summary_align", 0.0))
    if slow_style_summary_align_lambda > 0:
        slow_style_summary_align = _cosine_distance(slow_style_repr, global_style_summary)
        if slow_style_summary_align is not None:
            losses["slow_style_summary_align"] = (
                slow_style_summary_align * slow_style_summary_align_lambda
            )

    dynamic_timbre_gate_lambda = float(config.get("lambda_dynamic_timbre_gate", 0.0))
    if dynamic_timbre_gate_lambda > 0:
        gate_mean = _gate_mean(
            output.get("dynamic_timbre_gate"),
            output.get("dynamic_timbre_mask"),
        )
        if gate_mean is not None:
            gate_target = float(config.get("dynamic_timbre_gate_target", 0.6))
            losses["dynamic_timbre_gate_reg"] = (gate_mean - gate_target).abs() * dynamic_timbre_gate_lambda

    dynamic_timbre_budget_lambda = float(config.get("lambda_dynamic_timbre_budget", 0.0))
    if dynamic_timbre_budget_lambda > 0:
        style_decoder_residual = output.get("style_decoder_residual")
        dynamic_timbre_decoder_residual = output.get("dynamic_timbre_decoder_residual")
        if isinstance(style_decoder_residual, torch.Tensor) and isinstance(dynamic_timbre_decoder_residual, torch.Tensor):
            style_budget = style_decoder_residual.abs().mean()
            timbre_budget = dynamic_timbre_decoder_residual.abs().mean()
            budget_ratio = float(config.get("dynamic_timbre_budget_ratio", 0.75))
            losses["dynamic_timbre_budget"] = (
                F.relu(timbre_budget - budget_ratio * style_budget) * dynamic_timbre_budget_lambda
            )
