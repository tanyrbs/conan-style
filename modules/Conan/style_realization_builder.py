from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from modules.Conan.style_mainline import normalize_style_trace_mode
from modules.Conan.style_trace_utils import combine_style_traces, resolve_combined_style_trace


STYLE_TRACE_RUNTIME_FIELDS = (
    "ref_upsample",
    "vq_loss",
    "ppl",
    "gloss",
    "attn",
)


def _style_trace_mode_from_controls(controls) -> str:
    return normalize_style_trace_mode(getattr(controls, "style_trace_mode", "slow"), default="slow")


def _masked_mean(model, sequence, mask=None):
    if not isinstance(sequence, torch.Tensor):
        return None
    return model._masked_mean(sequence, mask)


def _preferred_fast_memory_mode(model, controls) -> str:
    chosen_memory_mode = model._normalize_memory_mode(
        getattr(controls, "style_memory_mode", "fast"),
        default="fast",
    )
    if chosen_memory_mode == "slow":
        return "fast"
    return chosen_memory_mode


def _trace_source_label(*, has_cached_prosody: bool, memory_mode: str):
    source_prefix = "reference_cache" if has_cached_prosody else "reference_audio"
    return f"{source_prefix}_{memory_mode}"


def _run_style_trace(
    model,
    *,
    query,
    content,
    ref_style,
    infer: bool,
    global_steps: int,
    reference_cache: Optional[Mapping[str, Any]],
    memory_mode: str,
    style_temperature: float,
    forcing_schedule_state=None,
):
    trace_ret = {"content": content}
    style_trace = model.get_prosody(
        query,
        ref_style,
        trace_ret,
        infer=infer,
        global_steps=global_steps,
        reference_cache=reference_cache,
        memory_mode=memory_mode,
        style_temperature=style_temperature,
        forcing_schedule_state=forcing_schedule_state,
    )
    return {
        "trace": style_trace,
        "mask": trace_ret.get("style_trace_mask"),
        "memory": trace_ret.get("style_trace_memory"),
        "memory_mask": trace_ret.get("style_trace_memory_mask"),
        "smooth": trace_ret.get("style_trace_smooth"),
        "runtime": {
            key: trace_ret.get(key)
            for key in STYLE_TRACE_RUNTIME_FIELDS
            if trace_ret.get(key) is not None
        },
    }


def _write_trace_result(ret, trace_result, *, prefix: str):
    trace = trace_result.get("trace")
    if not isinstance(trace, torch.Tensor):
        return
    ret[prefix] = trace
    if trace_result.get("mask") is not None:
        ret[f"{prefix}_mask"] = trace_result["mask"]
    if trace_result.get("memory") is not None:
        ret[f"{prefix}_memory"] = trace_result["memory"]
    if trace_result.get("memory_mask") is not None:
        ret[f"{prefix}_memory_mask"] = trace_result["memory_mask"]
    if trace_result.get("smooth") is not None:
        ret[f"{prefix}_smooth"] = trace_result["smooth"]
    if trace_result.get("source") is not None:
        ret[f"{prefix}_source_runtime"] = trace_result["source"]
    if trace_result.get("memory_mode") is not None:
        ret[f"{prefix}_memory_mode_used"] = trace_result["memory_mode"]


def _resolve_global_summary_runtime(model, global_style_summary, style_trace, style_trace_mask, *, blend=0.0):
    blend = float(blend)
    style_trace_summary = _masked_mean(model, style_trace, style_trace_mask)
    if style_trace_summary is None:
        return global_style_summary, "reference_summary"
    if isinstance(global_style_summary, torch.Tensor):
        base = global_style_summary[:, 0, :] if global_style_summary.dim() == 3 else global_style_summary
        if blend > 0.0:
            fused = base * (1.0 - blend) + style_trace_summary * blend
            return model._normalize_style_embed(fused), "style_trace_blended_with_reference"
    return model._normalize_style_embed(style_trace_summary), "style_trace_pooled"


def _branch_strength(style_strength, branch_scale: float):
    branch_scale = float(branch_scale)
    if isinstance(style_strength, torch.Tensor):
        return style_strength * branch_scale
    return float(style_strength) * branch_scale


def _merge_style_masks(*masks, reference=None):
    normalized = []
    for mask in masks:
        if not isinstance(mask, torch.Tensor):
            continue
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        if mask.dim() != 2:
            continue
        if reference is not None and tuple(mask.shape) != tuple(reference.shape[:2]):
            continue
        normalized.append(mask.bool())
    if not normalized:
        return None
    merged = normalized[0]
    for mask in normalized[1:]:
        merged = merged | mask.to(device=merged.device)
    return merged


def _route_style_owner(
    model,
    *,
    query,
    fast_trace,
    slow_trace,
    fast_mask=None,
    slow_mask=None,
    router_enabled: bool = True,
):
    has_fast = isinstance(fast_trace, torch.Tensor) and fast_trace.dim() == 3
    has_slow = isinstance(slow_trace, torch.Tensor) and slow_trace.dim() == 3

    if has_fast and has_slow and tuple(fast_trace.shape) == tuple(slow_trace.shape):
        merged_mask = _merge_style_masks(fast_mask, slow_mask, reference=fast_trace)
        router = getattr(model, "style_router", None)
        if (
            bool(router_enabled)
            and router is not None
            and isinstance(query, torch.Tensor)
            and tuple(query.shape) == tuple(fast_trace.shape)
        ):
            router_input = torch.cat([query, slow_trace, fast_trace], dim=-1)
            gate = torch.sigmoid(router(router_input))
            routed = slow_trace + gate * (fast_trace - slow_trace)
            burst_score = gate.squeeze(-1) * (fast_trace - slow_trace).abs().mean(dim=-1)
            return {
                "residual": routed,
                "mask": merged_mask,
                "gate": gate.squeeze(-1),
                "burst_score": burst_score,
                "source": "dual_router",
            }
        combined = combine_style_traces(fast_trace, slow_trace)
        burst_score = (fast_trace - slow_trace).abs().mean(dim=-1)
        return {
            "residual": combined,
            "mask": merged_mask,
            "gate": None,
            "burst_score": burst_score,
            "source": "dual_sum",
        }

    if has_fast:
        return {
            "residual": fast_trace,
            "mask": _merge_style_masks(fast_mask, reference=fast_trace),
            "gate": None,
            "burst_score": fast_trace.abs().mean(dim=-1),
            "source": "fast_only",
        }
    if has_slow:
        return {
            "residual": slow_trace,
            "mask": _merge_style_masks(slow_mask, reference=slow_trace),
            "gate": None,
            "burst_score": slow_trace.abs().mean(dim=-1),
            "source": "slow_only",
        }
    return {
        "residual": None,
        "mask": None,
        "gate": None,
        "burst_score": None,
        "source": "missing",
    }


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
    forcing_schedule_state=None,
    upper_bound_progress: float = 1.0,
):
    style_trace_mode = _style_trace_mode_from_controls(controls)
    upper_bound_progress = max(0.0, min(float(upper_bound_progress), 1.0))
    payload = {
        "style_trace_mode": style_trace_mode,
        "style_trace_available": False,
        "style_trace_source": "disabled_by_mode" if style_trace_mode == "none" else "missing",
        "fast_style_decoder_residual": None,
        "slow_style_decoder_residual": None,
        "style_decoder_residual": None,
        "style_decoder_residual_mask": None,
        "style_router_gate": None,
        "style_burst_score": None,
        "style_owner_source": "missing",
        "global_style_summary_runtime": global_style_summary,
        "style_trace_skip_reason": None,
        "global_style_summary_runtime_source": "reference_summary",
    }

    ret["style_trace_runtime_mode"] = style_trace_mode
    ret["style_trace_memory_mode_used"] = "none"
    ret["global_style_trace_blend"] = float(getattr(controls, "global_style_trace_blend", 0.0))
    ret["expressive_upper_bound_progress"] = float(upper_bound_progress)
    ret["style_fast_upper_bound_progress"] = float(upper_bound_progress)

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

    style_temperature = float(getattr(controls, "style_temperature", 1.0))
    fast_style_progress = float(upper_bound_progress)
    fast_style_strength = _branch_strength(
        style_strength,
        getattr(controls, "fast_style_strength_scale", 1.0) * fast_style_progress,
    )
    slow_style_strength = _branch_strength(
        style_strength,
        getattr(controls, "slow_style_strength_scale", 1.0),
    )
    ret["fast_style_strength_scale_runtime"] = float(
        getattr(controls, "fast_style_strength_scale", 1.0) * fast_style_progress
    )
    ret["slow_style_strength_scale_runtime"] = float(
        getattr(controls, "slow_style_strength_scale", 1.0)
    )
    fast_trace_result = None
    slow_trace_result = None

    if style_trace_mode in {"fast", "dual"} and fast_style_progress > 0.0:
        fast_memory_mode = _preferred_fast_memory_mode(model, controls)
        fast_trace_result = _run_style_trace(
            model,
            query=query,
            content=ret["content"],
            ref_style=ref_style,
            infer=infer,
            global_steps=global_steps,
            reference_cache=reference_cache,
            memory_mode=fast_memory_mode,
            style_temperature=style_temperature,
            forcing_schedule_state=forcing_schedule_state,
        )
        fast_trace_result["source"] = _trace_source_label(
            has_cached_prosody=has_cached_prosody,
            memory_mode=fast_memory_mode,
        )
        fast_trace_result["memory_mode"] = fast_memory_mode
        _write_trace_result(ret, fast_trace_result, prefix="style_trace")
        payload["fast_style_decoder_residual"] = fast_trace_result["trace"] * fast_style_strength
        for key, value in fast_trace_result.get("runtime", {}).items():
            ret[key] = value
        style_trace_summary = _masked_mean(
            model,
            fast_trace_result["trace"],
            fast_trace_result.get("mask"),
        )
        if style_trace_summary is not None:
            ret["style_trace_summary"] = style_trace_summary
    elif style_trace_mode in {"fast", "dual"}:
        ret["style_fast_branch_deferred_by_curriculum"] = True
        if style_trace_mode == "fast":
            payload["style_trace_skip_reason"] = "curriculum_deferred"
            payload["style_trace_source"] = "curriculum_deferred"

    if style_trace_mode in {"slow", "dual"}:
        slow_trace_result = _run_style_trace(
            model,
            query=query,
            content=ret["content"],
            ref_style=ref_style,
            infer=infer,
            global_steps=global_steps,
            reference_cache=reference_cache,
            memory_mode="slow",
            style_temperature=style_temperature,
            forcing_schedule_state=forcing_schedule_state,
        )
        slow_trace_result["source"] = _trace_source_label(
            has_cached_prosody=has_cached_prosody,
            memory_mode="slow",
        )
        slow_trace_result["memory_mode"] = "slow"
        _write_trace_result(ret, slow_trace_result, prefix="slow_style_trace")
        payload["slow_style_decoder_residual"] = slow_trace_result["trace"] * slow_style_strength
        if style_trace_mode == "slow":
            for key, value in slow_trace_result.get("runtime", {}).items():
                ret[key] = value
        slow_style_summary = _masked_mean(
            model,
            slow_trace_result["trace"],
            slow_trace_result.get("mask"),
        )
        if slow_style_summary is not None:
            ret["slow_style_trace_summary"] = slow_style_summary

    payload["style_trace_available"] = bool(
        isinstance(fast_trace_result, dict) and isinstance(fast_trace_result.get("trace"), torch.Tensor)
        or isinstance(slow_trace_result, dict) and isinstance(slow_trace_result.get("trace"), torch.Tensor)
    )
    if style_trace_mode == "dual":
        if fast_trace_result is None and isinstance(slow_trace_result, dict):
            payload["style_trace_source"] = (
                "reference_cache_slow_curriculum_floor"
                if has_cached_prosody else "reference_audio_slow_curriculum_floor"
            )
            ret["style_trace_memory_mode_used"] = "slow"
        else:
            payload["style_trace_source"] = "reference_cache_dual" if has_cached_prosody else "reference_audio_dual"
            ret["style_trace_memory_mode_used"] = "dual"
        ret["style_trace_source_runtime"] = payload["style_trace_source"]
    elif style_trace_mode == "slow":
        payload["style_trace_source"] = slow_trace_result.get("source", "missing") if isinstance(slow_trace_result, dict) else "missing"
        ret["style_trace_memory_mode_used"] = "slow"
        ret["style_trace_source_runtime"] = payload["style_trace_source"]
    else:
        payload["style_trace_source"] = fast_trace_result.get("source", "missing") if isinstance(fast_trace_result, dict) else "missing"
        ret["style_trace_memory_mode_used"] = (
            fast_trace_result.get("memory_mode", "none") if isinstance(fast_trace_result, dict) else "none"
        )
        ret["style_trace_source_runtime"] = payload["style_trace_source"]

    routed = _route_style_owner(
        model,
        query=query,
        fast_trace=payload["fast_style_decoder_residual"],
        slow_trace=payload["slow_style_decoder_residual"],
        fast_mask=ret.get("style_trace_mask"),
        slow_mask=ret.get("slow_style_trace_mask"),
        router_enabled=bool(getattr(controls, "style_router_enabled", True)),
    )
    payload["style_decoder_residual"] = routed.get("residual")
    payload["style_decoder_residual_mask"] = routed.get("mask")
    payload["style_router_gate"] = routed.get("gate")
    payload["style_burst_score"] = routed.get("burst_score")
    payload["style_owner_source"] = routed.get("source", "missing")

    if isinstance(payload["style_decoder_residual"], torch.Tensor):
        ret["style_decoder_residual_mask"] = payload["style_decoder_residual_mask"]
    if isinstance(payload["style_router_gate"], torch.Tensor):
        ret["style_router_gate"] = payload["style_router_gate"]
    if isinstance(payload["style_burst_score"], torch.Tensor):
        ret["style_burst_score"] = payload["style_burst_score"]
    ret["style_owner_source"] = payload["style_owner_source"]

    blended_trace = payload["style_decoder_residual"]
    blended_mask = payload["style_decoder_residual_mask"]
    if blended_trace is None:
        blended_trace, blended_mask = resolve_combined_style_trace(
            {
                "style_trace": payload["fast_style_decoder_residual"],
                "slow_style_trace": payload["slow_style_decoder_residual"],
                "style_trace_mask": ret.get("style_trace_mask"),
                "slow_style_trace_mask": ret.get("slow_style_trace_mask"),
            }
        )
        if blended_trace is None:
            blended_trace = combine_style_traces(
                payload["fast_style_decoder_residual"],
                payload["slow_style_decoder_residual"],
            )
            blended_mask = ret.get("style_trace_mask", ret.get("slow_style_trace_mask"))

    payload["global_style_summary_runtime"], payload["global_style_summary_runtime_source"] = _resolve_global_summary_runtime(
        model,
        global_style_summary,
        blended_trace,
        blended_mask,
        blend=float(getattr(controls, "global_style_trace_blend", 0.0)),
    )
    if isinstance(payload["global_style_summary_runtime"], torch.Tensor):
        ret["global_style_summary_runtime"] = payload["global_style_summary_runtime"]
    ret["global_style_summary_runtime_source"] = payload["global_style_summary_runtime_source"]
    return payload


__all__ = [
    "build_style_realization_payload",
]
