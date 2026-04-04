from __future__ import annotations

from typing import Mapping

import torch

from modules.Conan.common_utils import expand_sequence_like
from modules.Conan.decoder_style_runtime import build_runtime_decoder_style_bundle
from modules.Conan.style_mainline import (
    resolve_style_profile_defaults,
    resolve_style_runtime_value,
)
from modules.Conan.style_realization_builder import build_style_realization_payload
from modules.Conan.style_trace_utils import combine_style_traces, resolve_combined_style_trace


def _merge_runtime_masks(*masks, reference=None):
    normalized = []
    for mask in masks:
        if not isinstance(mask, torch.Tensor):
            continue
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        if mask.dim() != 2:
            continue
        if isinstance(reference, torch.Tensor) and tuple(mask.shape) != tuple(reference.shape[:2]):
            continue
        device = reference.device if isinstance(reference, torch.Tensor) else mask.device
        normalized.append(mask.bool().to(device))
    if not normalized:
        return None
    merged = normalized[0]
    for mask in normalized[1:]:
        merged = merged | mask.to(device=merged.device)
    return merged


def _sequence_has_signal(
    value,
    *,
    reference=None,
    abs_eps: float = 1.0e-4,
    relative_eps: float = 0.02,
) -> bool:
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        return False
    signal = float(value.detach().abs().mean().item())
    threshold = float(abs_eps)
    if isinstance(reference, torch.Tensor) and reference.numel() > 0:
        reference_signal = float(reference.detach().abs().mean().item())
        if reference_signal > float(abs_eps):
            threshold = max(threshold, float(relative_eps) * reference_signal)
    return bool(signal > threshold)


def _resolve_style_owner_residual(style_payload, ret):
    fast_style_decoder_residual = style_payload.get("fast_style_decoder_residual")
    slow_style_decoder_residual = style_payload.get("slow_style_decoder_residual")
    style_decoder_residual = style_payload.get("style_decoder_residual")
    style_decoder_residual_mask = style_payload.get("style_decoder_residual_mask")
    if style_decoder_residual is None:
        style_decoder_residual, style_decoder_residual_mask = resolve_combined_style_trace(
            {
                "style_trace": fast_style_decoder_residual,
                "slow_style_trace": slow_style_decoder_residual,
                "style_trace_mask": ret.get("style_trace_mask"),
                "slow_style_trace_mask": ret.get("slow_style_trace_mask"),
            }
        )
    if style_decoder_residual is None:
        style_decoder_residual = combine_style_traces(
            fast_style_decoder_residual,
            slow_style_decoder_residual,
        )
        style_decoder_residual_mask = _merge_runtime_masks(
            ret.get("style_trace_mask"),
            ret.get("slow_style_trace_mask"),
            reference=style_decoder_residual,
        )
    if isinstance(style_decoder_residual, torch.Tensor) and not isinstance(
        style_decoder_residual_mask,
        torch.Tensor,
    ):
        style_decoder_residual_mask = _merge_runtime_masks(
            ret.get("style_trace_mask"),
            ret.get("slow_style_trace_mask"),
            reference=style_decoder_residual,
        )
    slow_style_decoder_residual_mask = ret.get("slow_style_trace_mask")
    style_owner_base_residual = (
        slow_style_decoder_residual
        if isinstance(slow_style_decoder_residual, torch.Tensor)
        else None
    )
    style_owner_base_mask = _merge_runtime_masks(
        slow_style_decoder_residual_mask,
        reference=style_owner_base_residual,
    )
    style_owner_innovation_residual = None
    style_owner_innovation_mask = None
    if isinstance(style_decoder_residual, torch.Tensor):
        if (
            isinstance(style_owner_base_residual, torch.Tensor)
            and tuple(style_owner_base_residual.shape) == tuple(style_decoder_residual.shape)
        ):
            style_owner_innovation_residual = style_decoder_residual - style_owner_base_residual
            style_owner_innovation_mask = _merge_runtime_masks(
                style_decoder_residual_mask,
                style_owner_base_mask,
                reference=style_decoder_residual,
            )
        else:
            style_owner_innovation_residual = style_decoder_residual
            style_owner_innovation_mask = _merge_runtime_masks(
                style_decoder_residual_mask,
                reference=style_decoder_residual,
            )
    return {
        "fast_style_decoder_residual": fast_style_decoder_residual,
        "slow_style_decoder_residual": slow_style_decoder_residual,
        "style_decoder_residual": style_decoder_residual,
        "style_decoder_residual_mask": style_decoder_residual_mask,
        "style_owner_base_residual": style_owner_base_residual,
        "style_owner_base_mask": style_owner_base_mask,
        "style_owner_innovation_residual": style_owner_innovation_residual,
        "style_owner_innovation_mask": style_owner_innovation_mask,
    }


def prepare_style_query_runtime_inputs(
    model,
    *,
    content,
    content_embed,
    condition_embed,
    global_timbre_anchor,
    global_style_summary,
    global_style_anchor_strength,
    style_mainline,
    kwargs,
    ret,
):
    global_timbre_anchor_runtime = global_timbre_anchor * global_style_anchor_strength
    if not style_mainline.apply_global_style_anchor:
        global_timbre_anchor_runtime = torch.zeros_like(global_timbre_anchor_runtime)
    ret["global_timbre_anchor_runtime"] = global_timbre_anchor_runtime

    base_condition_inp = content_embed + condition_embed
    ret["query_condition_inp"] = base_condition_inp
    pitch_inp = base_condition_inp
    if style_mainline.apply_global_style_anchor and style_mainline.global_timbre_to_pitch:
        pitch_inp = pitch_inp + global_timbre_anchor_runtime
    ret["pitch_condition_inp"] = pitch_inp
    ret["global_timbre_to_pitch_applied"] = bool(
        style_mainline.apply_global_style_anchor and style_mainline.global_timbre_to_pitch
    )

    style_profile_defaults = resolve_style_profile_defaults(kwargs, hparams=model.hparams)
    global_style_query_prior = expand_sequence_like(
        global_style_summary,
        content_embed.size(1),
        device=content_embed.device,
        dtype=content_embed.dtype,
    )
    style_query_global_summary_scale = float(
        resolve_style_runtime_value(
            "style_query_global_summary_scale",
            overrides=kwargs,
            hparams=model.hparams,
            profile_defaults=style_profile_defaults,
            default=0.0,
        )
    )
    style_query_base = base_condition_inp
    if (
        isinstance(global_style_query_prior, torch.Tensor)
        and style_query_global_summary_scale != 0.0
    ):
        style_query_base = style_query_base + style_query_global_summary_scale * global_style_query_prior
    if model.style_query_norm is not None:
        style_query_base = model.style_query_norm(style_query_base)
    ret["global_style_query_prior"] = global_style_query_prior
    ret["style_query_global_summary_scale"] = style_query_global_summary_scale
    ret["style_query_base"] = style_query_base
    ret["query_anchor_split_applied"] = True
    style_query_inp = (
        model.style_query_proj(style_query_base)
        if model.style_query_proj is not None
        else style_query_base
    )
    ret["style_query_inp"] = style_query_inp
    return {
        "base_condition_inp": base_condition_inp,
        "pitch_inp": pitch_inp,
        "style_profile_defaults": style_profile_defaults,
        "style_query_inp": style_query_inp,
        "global_timbre_anchor_runtime": global_timbre_anchor_runtime,
    }


def _prepare_timbre_query_runtime_inputs(
    model,
    *,
    content,
    base_condition_inp,
    style_profile_defaults,
    kwargs,
    style_decoder_residual,
    ret,
):
    dynamic_timbre_coarse_style_context_scale = float(
        resolve_style_runtime_value(
            "dynamic_timbre_coarse_style_context_scale",
            overrides=kwargs,
            hparams=model.hparams,
            profile_defaults=style_profile_defaults,
            default=0.0,
        )
    )
    dynamic_timbre_style_context_stopgrad = bool(
        resolve_style_runtime_value(
            "dynamic_timbre_style_context_stopgrad",
            overrides=kwargs,
            hparams=model.hparams,
            profile_defaults=style_profile_defaults,
            default=True,
        )
    )
    content_padding_mask = content.eq(model.content_padding_idx)
    dynamic_timbre_style_context = model._prepare_dynamic_timbre_style_context(
        style_decoder_residual,
        padding_mask=content_padding_mask,
        stopgrad=dynamic_timbre_style_context_stopgrad,
    )
    ret["dynamic_timbre_style_context_raw"] = style_decoder_residual
    ret["dynamic_timbre_style_context"] = dynamic_timbre_style_context
    ret["dynamic_timbre_coarse_style_context_scale"] = dynamic_timbre_coarse_style_context_scale
    ret["dynamic_timbre_coarse_style_context_scale_requested"] = dynamic_timbre_coarse_style_context_scale
    ret["dynamic_timbre_coarse_style_context_applied"] = False
    ret["timbre_query_style_context_applied"] = False
    ret["dynamic_timbre_style_context_stopgrad"] = dynamic_timbre_style_context_stopgrad
    ret["dynamic_timbre_style_context_owner_safe"] = isinstance(dynamic_timbre_style_context, torch.Tensor)
    ret["dynamic_timbre_style_context_bridge"] = (
        "layernorm_stopgrad" if dynamic_timbre_style_context_stopgrad else "layernorm"
    )
    query_style_scale_override = resolve_style_runtime_value(
        "dynamic_timbre_query_style_condition_scale",
        overrides=kwargs,
        hparams=model.hparams,
        profile_defaults=style_profile_defaults,
        default=0.0,
    )
    if query_style_scale_override is None:
        query_style_scale_override = 0.0
    ret["dynamic_timbre_query_style_condition_scale_requested"] = float(
        query_style_scale_override
    )
    ret["dynamic_timbre_query_style_condition_scale"] = float(query_style_scale_override)
    if float(query_style_scale_override) != 0.0:
        timbre_query_style_scale = float(query_style_scale_override)
        timbre_query_style_scale_source = "query_style_condition"
    elif float(dynamic_timbre_coarse_style_context_scale) != 0.0:
        timbre_query_style_scale = float(dynamic_timbre_coarse_style_context_scale)
        timbre_query_style_scale_source = "coarse_style_context"
    else:
        timbre_query_style_scale = 0.0
        timbre_query_style_scale_source = "disabled"
    ret["timbre_query_style_scale"] = timbre_query_style_scale
    ret["timbre_query_style_scale_source"] = timbre_query_style_scale_source
    timbre_query_base = base_condition_inp
    if (
        isinstance(dynamic_timbre_style_context, torch.Tensor)
        and timbre_query_style_scale != 0.0
    ):
        timbre_query_base = timbre_query_base + timbre_query_style_scale * dynamic_timbre_style_context
        ret["timbre_query_style_context_applied"] = True
        ret["dynamic_timbre_coarse_style_context_applied"] = bool(
            timbre_query_style_scale_source == "coarse_style_context"
        )
    if model.timbre_query_norm is not None:
        timbre_query_base = model.timbre_query_norm(timbre_query_base)
    timbre_query_inp = (
        model.timbre_query_proj(timbre_query_base)
        if model.timbre_query_proj is not None
        else timbre_query_base
    )
    ret["timbre_query_base"] = timbre_query_base
    ret["timbre_query_inp"] = timbre_query_inp
    ret["timbre_query_follows_style_owner"] = True
    return {
        "dynamic_timbre_style_context": dynamic_timbre_style_context,
        "timbre_query_inp": timbre_query_inp,
    }


def realize_style_timbre_decoder_runtime(
    model,
    *,
    content,
    base_condition_inp,
    style_query_inp,
    style_profile_defaults,
    reference_cache,
    reference_contract,
    ref_style,
    ref_dynamic_timbre,
    style_mainline,
    global_timbre_anchor,
    global_timbre_anchor_runtime,
    global_style_summary,
    infer,
    global_steps,
    kwargs,
    ret,
    expressive_upper_bound_progress,
):
    has_cached_timbre = bool(
        isinstance(reference_cache, Mapping)
        and (
            reference_cache.get("timbre_memory") is not None
            or reference_cache.get("timbre_memory_slow") is not None
        )
    )
    style_trace_source = "disabled_by_mode" if not style_mainline.apply_style_trace else "missing"
    dynamic_timbre_source = (
        "disabled_by_mode" if not style_mainline.apply_dynamic_timbre else "missing"
    )
    style_strength = model._resolve_strength(
        style_mainline.style_strength,
        batch_size=content.size(0),
        device=content.device,
    )
    dynamic_timbre_strength = model._resolve_strength(
        style_mainline.dynamic_timbre_strength,
        batch_size=content.size(0),
        device=content.device,
    )

    style_payload = build_style_realization_payload(
        model,
        query=style_query_inp,
        ret=ret,
        reference_cache=reference_cache,
        ref_style=ref_style,
        infer=infer,
        global_steps=global_steps,
        controls=style_mainline,
        global_style_summary=global_style_summary,
        global_style_summary_source=(
            reference_cache.get("global_style_summary_source", "reference_summary")
            if isinstance(reference_cache, Mapping)
            else "reference_summary"
        ),
        style_strength=style_strength,
        forcing_schedule_state=kwargs.get("forcing_schedule_state"),
        upper_bound_progress=expressive_upper_bound_progress,
    )
    style_trace_available = bool(style_payload.get("style_trace_available", False))
    style_trace_source = str(style_payload.get("style_trace_source", style_trace_source))
    style_owner_payload = _resolve_style_owner_residual(style_payload, ret)
    fast_style_decoder_residual = style_owner_payload["fast_style_decoder_residual"]
    slow_style_decoder_residual = style_owner_payload["slow_style_decoder_residual"]
    style_decoder_residual = style_owner_payload["style_decoder_residual"]
    style_decoder_residual_mask = style_owner_payload["style_decoder_residual_mask"]
    style_owner_base_residual = style_owner_payload["style_owner_base_residual"]
    style_owner_base_mask = style_owner_payload["style_owner_base_mask"]
    style_owner_innovation_residual = style_owner_payload["style_owner_innovation_residual"]
    style_owner_innovation_mask = style_owner_payload["style_owner_innovation_mask"]
    global_style_summary_runtime = style_payload.get(
        "global_style_summary_runtime",
        global_style_summary,
    )
    global_style_summary_runtime_source = style_payload.get(
        "global_style_summary_runtime_source",
        reference_cache.get("global_style_summary_source", "reference_cache")
        if isinstance(reference_cache, Mapping)
        else "reference_cache",
    )
    ret["main_style_owner_residual"] = style_decoder_residual
    ret["style_owner_source"] = style_payload.get(
        "style_owner_source",
        ret.get("style_owner_source", "missing"),
    )
    if style_payload.get("style_trace_skip_reason") is not None:
        ret["style_trace_skip_reason"] = style_payload.get("style_trace_skip_reason")

    timbre_style_context_residual = style_owner_innovation_residual
    timbre_style_context_source = "style_owner_innovation"
    if not _sequence_has_signal(
        timbre_style_context_residual,
        reference=style_decoder_residual,
    ):
        timbre_style_context_residual = style_decoder_residual
        timbre_style_context_source = "style_owner"
    ret["dynamic_timbre_style_context_source"] = timbre_style_context_source

    timbre_query_payload = _prepare_timbre_query_runtime_inputs(
        model,
        content=content,
        base_condition_inp=base_condition_inp,
        style_profile_defaults=style_profile_defaults,
        kwargs=kwargs,
        style_decoder_residual=timbre_style_context_residual,
        ret=ret,
    )
    dynamic_timbre_style_context = timbre_query_payload["dynamic_timbre_style_context"]
    timbre_query_inp = timbre_query_payload["timbre_query_inp"]

    dynamic_timbre_decoder_residual = None
    dynamic_timbre_prebudget = None
    dynamic_timbre_available = False
    if (
        model.use_dynamic_timbre
        and style_mainline.apply_dynamic_timbre
        and (ref_dynamic_timbre is not None or has_cached_timbre)
    ):
        dynamic_timbre = model.get_dynamic_timbre(
            timbre_query_inp,
            ref_dynamic_timbre,
            ret,
            infer=infer,
            global_steps=global_steps,
            reference_cache=reference_cache,
            memory_mode=style_mainline.dynamic_timbre_memory_mode,
            timbre_temperature=style_mainline.dynamic_timbre_temperature,
            style_context=dynamic_timbre_style_context,
            style_condition_scale=style_mainline.dynamic_timbre_style_condition_scale,
            gate_scale=style_mainline.dynamic_timbre_gate_scale,
            gate_bias=style_mainline.dynamic_timbre_gate_bias,
            boundary_suppress_strength=style_mainline.dynamic_timbre_boundary_suppress_strength,
            boundary_radius=style_mainline.dynamic_timbre_boundary_radius,
            anchor_preserve_strength=style_mainline.dynamic_timbre_anchor_preserve_strength,
            style_context_prepared=True,
            tvt_prior_scale=float(style_mainline.dynamic_timbre_tvt_prior_scale),
            use_tvt=bool(style_mainline.dynamic_timbre_use_tvt),
            upper_bound_progress=expressive_upper_bound_progress,
        )
        dynamic_timbre_prebudget = dynamic_timbre * dynamic_timbre_strength
        ret["dynamic_timbre_decoder_residual_prebudget"] = dynamic_timbre_prebudget
        dynamic_timbre_decoder_residual = model._apply_runtime_dynamic_timbre_budget(
            dynamic_timbre_prebudget,
            style_decoder_residual=style_decoder_residual,
            slow_style_decoder_residual=slow_style_decoder_residual,
            style_owner_base_residual=style_owner_base_residual,
            style_owner_innovation_residual=style_owner_innovation_residual,
            content=content,
            kwargs=kwargs,
            ret=ret,
        )
        dynamic_timbre_available = True
        dynamic_timbre_source = "reference_cache" if has_cached_timbre else "reference_audio"
    if model.use_dynamic_timbre and not style_mainline.apply_dynamic_timbre:
        ret["dynamic_timbre_skip_reason"] = "decoder_style_condition_mode"
    elif (
        model.use_dynamic_timbre
        and not dynamic_timbre_available
        and not (ref_dynamic_timbre is not None or has_cached_timbre)
    ):
        ret["dynamic_timbre_skip_reason"] = "reference_missing"

    ret["style_trace_applied"] = bool(style_trace_available)
    ret["dynamic_timbre_applied"] = bool(dynamic_timbre_available)
    ret["fast_style_decoder_residual"] = fast_style_decoder_residual
    ret["style_decoder_residual"] = style_decoder_residual
    ret["style_decoder_residual_mask"] = style_decoder_residual_mask
    ret["slow_style_decoder_residual"] = slow_style_decoder_residual
    ret["style_owner_base_residual"] = style_owner_base_residual
    ret["style_owner_base_mask"] = style_owner_base_mask
    ret["style_owner_innovation_residual"] = style_owner_innovation_residual
    ret["style_owner_innovation_mask"] = style_owner_innovation_mask
    ret["dynamic_timbre_decoder_residual"] = dynamic_timbre_decoder_residual
    if dynamic_timbre_prebudget is None:
        ret.setdefault("dynamic_timbre_decoder_residual_prebudget", None)
    ret["global_style_summary_runtime"] = global_style_summary_runtime
    ret["global_style_summary_runtime_source"] = global_style_summary_runtime_source

    decoder_style_bundle, decoder_signal_eps = build_runtime_decoder_style_bundle(
        decoder_style_adapter=model.decoder_style_adapter,
        reference_cache=reference_cache,
        reference_contract=reference_contract,
        style_mainline=style_mainline,
        global_timbre_anchor=global_timbre_anchor,
        global_timbre_anchor_runtime=global_timbre_anchor_runtime,
        global_style_summary_runtime=global_style_summary_runtime,
        global_style_summary_runtime_source=global_style_summary_runtime_source,
        slow_style_trace=slow_style_decoder_residual,
        slow_style_trace_mask=ret.get("slow_style_trace_mask"),
        slow_style_source=str(
            ret.get(
                "slow_style_trace_source_runtime",
                style_payload.get("slow_style_source_runtime", "missing"),
            )
        ),
        M_style=style_decoder_residual,
        M_style_mask=style_decoder_residual_mask,
        M_style_source=str(ret.get("style_owner_source", "combined_style_owner")),
        M_timbre=dynamic_timbre_decoder_residual,
        M_timbre_mask=ret.get("dynamic_timbre_mask"),
        M_timbre_source=dynamic_timbre_source,
    )
    ret["decoder_style_bundle"] = decoder_style_bundle
    ret["decoder_style_bundle_effective_signal_epsilon"] = decoder_signal_eps
    return {
        "style_trace_available": bool(style_trace_available),
        "dynamic_timbre_available": bool(dynamic_timbre_available),
        "style_trace_source": style_trace_source,
        "dynamic_timbre_source": dynamic_timbre_source,
        "global_style_summary_runtime": global_style_summary_runtime,
        "global_style_summary_runtime_source": global_style_summary_runtime_source,
    }


__all__ = [
    "prepare_style_query_runtime_inputs",
    "realize_style_timbre_decoder_runtime",
]
