import torch
import torch.nn.functional as F

from modules.Conan.control.common import summary_vector
from modules.Conan.style_trace_utils import resolve_combined_style_trace


TRACKED_LAMBDA_KEYS = (
    "lambda_style_contrastive",
    "lambda_global_style_contrastive",
    "lambda_style_trace_consistency",
    "lambda_timbre_anchor_cosine",
    "lambda_style_timbre_disentangle",
    "lambda_style_dynamic_timbre_disentangle",
    "lambda_dynamic_timbre_gate",
    "lambda_global_style_summary_align",
    "lambda_tv_timbre_smooth",
    "lambda_tv_timbre_anchor",
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


def _mask_valid_fraction(mask, sequence):
    mask = _normalize_mask(mask, sequence)
    if mask is None:
        return None
    return (~mask).float().mean()


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


def _branch_statistics(branch_ctx, branch_gate, prefix):
    if not isinstance(branch_ctx, torch.Tensor) or not isinstance(branch_gate, torch.Tensor):
        return {}
    gate_flat = _flatten_tensor(branch_gate)
    if gate_flat is None or gate_flat.numel() <= 0:
        return {}
    branch_value = branch_ctx
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


def _decoder_style_adapter_statistics(stage_outputs, dynamic_timbre_residual=None):
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
    global_residual_values = []
    slow_style_residual_values = []
    for stage_meta in stage_outputs.values():
        if not isinstance(stage_meta, dict):
            continue
        branch_stats = _branch_statistics(
            stage_meta.get("global_timbre_ctx"),
            stage_meta.get("global_timbre_gate"),
            "diag_decoder_global_timbre",
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
        )
        if "diag_decoder_global_style_gate_mean" in branch_stats:
            global_gate_values.append(branch_stats["diag_decoder_global_style_gate_mean"])
        if "diag_decoder_global_style_gate_std" in branch_stats:
            global_gate_stds.append(branch_stats["diag_decoder_global_style_gate_std"])
        if "diag_decoder_global_style_residual_norm" in branch_stats:
            global_residual_values.append(branch_stats["diag_decoder_global_style_residual_norm"])

        branch_stats = _branch_statistics(
            stage_meta.get("slow_style_ctx", stage_meta.get("style_trace_ctx")),
            stage_meta.get("slow_style_gate", stage_meta.get("style_trace_gate")),
            "diag_decoder_slow_style",
        )
        if "diag_decoder_slow_style_gate_mean" in branch_stats:
            slow_style_gate_values.append(branch_stats["diag_decoder_slow_style_gate_mean"])
        if "diag_decoder_slow_style_gate_std" in branch_stats:
            slow_style_gate_stds.append(branch_stats["diag_decoder_slow_style_gate_std"])
        if "diag_decoder_slow_style_residual_norm" in branch_stats:
            slow_style_residual_values.append(branch_stats["diag_decoder_slow_style_residual_norm"])

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
    if global_residual_values:
        stats["diag_decoder_global_style_residual_norm"] = torch.stack(global_residual_values).mean()
    if slow_style_residual_values:
        stats["diag_decoder_slow_style_residual_norm"] = torch.stack(slow_style_residual_values).mean()
    style_total_residual = None
    if "diag_decoder_global_style_residual_norm" in stats and "diag_decoder_slow_style_residual_norm" in stats:
        style_total_residual = (
            stats["diag_decoder_global_style_residual_norm"] + stats["diag_decoder_slow_style_residual_norm"]
        )
    elif "diag_decoder_global_style_residual_norm" in stats:
        style_total_residual = stats["diag_decoder_global_style_residual_norm"]
    elif "diag_decoder_slow_style_residual_norm" in stats:
        style_total_residual = stats["diag_decoder_slow_style_residual_norm"]
    if style_total_residual is not None:
        if "diag_decoder_global_timbre_residual_norm" in stats:
            style_total_residual = style_total_residual + stats["diag_decoder_global_timbre_residual_norm"]
        stats["diag_decoder_style_total_residual_norm"] = style_total_residual
    dynamic_norm, _ = _safe_mean_std(
        dynamic_timbre_residual.norm(dim=-1) if isinstance(dynamic_timbre_residual, torch.Tensor) else None
    )
    if (
        "diag_decoder_style_total_residual_norm" in stats
        and dynamic_norm is not None
        and float(dynamic_norm.item()) > 1e-8
    ):
        stats["diag_style_to_timbre_residual_ratio"] = (
            stats["diag_decoder_style_total_residual_norm"] / dynamic_norm
        )
    return stats


def collect_control_diagnostics(output, sample, config):
    if not config.get("log_control_diagnostics", True):
        return {}

    diagnostics = {}
    with torch.no_grad():
        style_repeat, style_unique_ratio, style_valid_fraction = _label_repeat_fraction(
            sample.get("style_ids")
        )
        if style_repeat is not None:
            diagnostics["diag_style_batch_repeat_frac"] = style_repeat
        if style_unique_ratio is not None:
            if isinstance(style_repeat, torch.Tensor):
                diagnostics["diag_style_batch_unique_ratio"] = style_repeat.new_tensor(
                    float(style_unique_ratio)
                )
            else:
                diagnostics["diag_style_batch_unique_ratio"] = torch.tensor(
                    float(style_unique_ratio),
                    dtype=torch.float32,
                )
        if style_valid_fraction is not None:
            diagnostics["diag_style_batch_valid_frac"] = style_valid_fraction

        for sample_key, diag_mean_key, diag_std_key in (
            ("style_strengths", "diag_style_strength_mean", "diag_style_strength_std"),
            (
                "dynamic_timbre_strengths",
                "diag_dynamic_timbre_strength_mean",
                "diag_dynamic_timbre_strength_std",
            ),
        ):
            mean, std = _safe_mean_std(sample.get(sample_key))
            if mean is not None:
                diagnostics[diag_mean_key] = mean
            if std is not None:
                diagnostics[diag_std_key] = std

        style_trace, style_trace_mask = resolve_combined_style_trace(output)
        slow_style_trace = output.get("slow_style_trace")
        dynamic_timbre = output.get("dynamic_timbre")
        style_repr = _masked_sequence_mean(style_trace, style_trace_mask)
        timbre_repr = _masked_sequence_mean(dynamic_timbre, output.get("dynamic_timbre_mask"))
        global_timbre_anchor = summary_vector(output.get("global_timbre_anchor", output.get("style_embed")))
        global_style_summary = summary_vector(
            output.get("global_style_summary_runtime", output.get("global_style_summary", output.get("style_embed")))
        )

        style_valid = _mask_valid_fraction(style_trace_mask, style_trace)
        if style_valid is not None:
            diagnostics["diag_style_trace_valid_frac"] = style_valid
        slow_style_valid = _mask_valid_fraction(output.get("slow_style_trace_mask"), slow_style_trace)
        if slow_style_valid is not None:
            diagnostics["diag_slow_style_trace_valid_frac"] = slow_style_valid
        timbre_valid = _mask_valid_fraction(output.get("dynamic_timbre_mask"), dynamic_timbre)
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

        style_timbre_cos = _cosine_mean(style_repr, timbre_repr)
        if style_timbre_cos is not None:
            diagnostics["diag_style_dynamic_timbre_cos"] = style_timbre_cos
        style_global_cos = _cosine_mean(style_repr, global_style_summary)
        if style_global_cos is not None:
            diagnostics["diag_style_global_cos"] = style_global_cos
        timbre_anchor_cos = _cosine_mean(timbre_repr, global_timbre_anchor)
        if timbre_anchor_cos is not None:
            diagnostics["diag_timbre_anchor_cos"] = timbre_anchor_cos
        query_cos = _cosine_mean(
            _summary_like(output.get("style_query_inp")),
            _summary_like(output.get("timbre_query_inp")),
        )
        if query_cos is not None:
            diagnostics["diag_style_timbre_query_cos"] = query_cos

        diagnostics.update(
            _decoder_style_adapter_statistics(
                output.get("decoder_style_adapter_stages"),
                dynamic_timbre_residual=output.get("dynamic_timbre_decoder_residual"),
            )
        )
        style_summary_source = output.get("global_style_summary_source")
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
                output.get("dynamic_timbre_mask"),
                target=float(config.get("dynamic_timbre_gate_target", 0.6)),
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("dynamic_timbre_gate_raw"),
                output.get("dynamic_timbre_mask"),
                prefix="diag_dynamic_timbre_gate_raw",
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("dynamic_timbre_boundary_mask"),
                output.get("dynamic_timbre_mask"),
                prefix="diag_dynamic_timbre_boundary_mask",
            )
        )
        diagnostics.update(
            _simple_sequence_statistics(
                output.get("dynamic_timbre_boundary_scale"),
                output.get("dynamic_timbre_mask"),
                prefix="diag_dynamic_timbre_boundary_scale",
            )
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
            for key in TRACKED_LAMBDA_KEYS:
                if key in config:
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
    return diagnostics
