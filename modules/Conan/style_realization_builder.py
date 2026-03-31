from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


def _style_trace_mode_from_controls(controls) -> str:
    mode = getattr(controls, "style_trace_mode", "fast")
    normalized = str(mode or "fast").strip().lower() or "fast"
    if normalized not in {"none", "fast", "slow"}:
        return "fast"
    return normalized


def _masked_mean(model, sequence, mask=None):
    if not isinstance(sequence, torch.Tensor):
        return None
    return model._masked_mean(sequence, mask)


def _blend_global_summary(model, global_style_summary, style_trace, style_trace_mask, *, blend=0.0):
    blend = float(blend)
    if blend <= 0.0:
        return global_style_summary
    style_trace_summary = _masked_mean(model, style_trace, style_trace_mask)
    if style_trace_summary is None:
        return global_style_summary
    if isinstance(global_style_summary, torch.Tensor):
        base = global_style_summary[:, 0, :] if global_style_summary.dim() == 3 else global_style_summary
        fused = base * (1.0 - blend) + style_trace_summary * blend
        return model._normalize_style_embed(fused)
    return model._normalize_style_embed(style_trace_summary)


def build_style_realization_payload(
    model,
    *,
    query,
    ret,
    reference_cache: Optional[Mapping[str, Any]],
    ref_style,
    infer: bool,
    global_steps: int,
    controls,
    global_style_summary,
    style_strength,
):
    style_trace_mode = _style_trace_mode_from_controls(controls)
    zero_residual = torch.zeros_like(query)
    payload = {
        "style_trace_mode": style_trace_mode,
        "style_trace_available": False,
        "style_trace_source": "disabled_by_mode" if style_trace_mode == "none" else "missing",
        "style_decoder_residual": zero_residual,
        "slow_style_decoder_residual": zero_residual,
        "global_style_summary_runtime": global_style_summary,
        "style_trace_skip_reason": None,
    }

    ret["style_trace_runtime_mode"] = style_trace_mode
    ret["style_trace_memory_mode_used"] = "none"
    ret["global_style_trace_blend"] = float(getattr(controls, "global_style_trace_blend", 0.0))

    has_cached_prosody = False
    if isinstance(reference_cache, Mapping):
        has_cached_prosody = (
            reference_cache.get("prosody_memory") is not None
            or reference_cache.get("prosody_memory_slow") is not None
        )

    if style_trace_mode == "none":
        payload["style_trace_skip_reason"] = "decoder_style_condition_mode"
        return payload
    if not getattr(model, "prosody_extractor", None) or not bool(model.hparams.get("style", False)):
        payload["style_trace_skip_reason"] = "style_disabled"
        payload["style_trace_source"] = "style_disabled"
        return payload
    if ref_style is None and not has_cached_prosody:
        payload["style_trace_skip_reason"] = "reference_missing"
        return payload

    chosen_memory_mode = "slow" if style_trace_mode == "slow" else getattr(controls, "style_memory_mode", "fast")
    chosen_memory_mode = model._normalize_memory_mode(chosen_memory_mode, default="fast")
    style_trace = model.get_prosody(
        query,
        ref_style,
        ret,
        infer=infer,
        global_steps=global_steps,
        reference_cache=reference_cache,
        memory_mode=chosen_memory_mode,
        style_temperature=float(getattr(controls, "style_temperature", 1.0)),
    )
    style_decoder_residual = style_trace * style_strength
    payload["style_trace_available"] = True
    payload["style_trace_source"] = (
        f"reference_cache_{chosen_memory_mode}" if has_cached_prosody else f"reference_audio_{chosen_memory_mode}"
    )
    ret["style_trace_memory_mode_used"] = chosen_memory_mode
    ret["style_trace_source_runtime"] = payload["style_trace_source"]

    if style_trace_mode == "slow":
        ret["slow_style_trace"] = ret.get("style_trace")
        ret["slow_style_trace_mask"] = ret.get("style_trace_mask")
        ret["slow_style_trace_memory"] = ret.get("style_trace_memory")
        ret["slow_style_trace_memory_mask"] = ret.get("style_trace_memory_mask")
        payload["slow_style_decoder_residual"] = style_decoder_residual
    else:
        payload["style_decoder_residual"] = style_decoder_residual

    payload["global_style_summary_runtime"] = _blend_global_summary(
        model,
        global_style_summary,
        style_decoder_residual,
        ret.get("style_trace_mask"),
        blend=float(getattr(controls, "global_style_trace_blend", 0.0)),
    )
    if isinstance(payload["global_style_summary_runtime"], torch.Tensor):
        ret["global_style_summary_runtime"] = payload["global_style_summary_runtime"]
    return payload


__all__ = [
    "build_style_realization_payload",
]
