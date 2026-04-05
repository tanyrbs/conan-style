"""Shared single-reference Conan inference request schema helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

CONDITION_FIELDS = ("emotion", "accent")

PUBLIC_CONTROL_KEYS = (
    "style_profile",
    "style_strength",
)

PUBLIC_CONTROL_ALIASES = {
    "style_strength": ("style_strengths",),
}

ADVANCED_CONDITION_CONTROL_KEYS = (
    "emotion",
    "emotion_id",
    "emotion_strength",
    "accent",
    "accent_id",
    "accent_strength",
    "arousal",
    "valence",
    "energy",
)

ADVANCED_MODEL_CONTROL_KEYS = (
    "dynamic_timbre_strength",
)

ADVANCED_STYLE_RUNTIME_KEYS = (
    "allow_mainline_profile_research_overrides",
    "allow_explicit_dynamic_timbre_strength",
    "decoder_style_condition_mode",
    "global_timbre_to_pitch",
    "style_to_pitch_residual",
    "style_to_pitch_residual_include_timbre",
    "style_to_pitch_residual_mode",
    "style_to_pitch_residual_scale",
    "style_to_pitch_residual_max_semitones",
    "style_to_pitch_residual_smooth_factor",
    "global_style_anchor_strength",
    "style_trace_mode",
    "style_memory_mode",
    "fast_style_strength_scale",
    "slow_style_strength_scale",
    "style_router_enabled",
    "style_temperature",
    "global_style_trace_blend",
    "dynamic_timbre_memory_mode",
    "dynamic_timbre_style_condition_scale",
    "dynamic_timbre_temperature",
    "dynamic_timbre_gate_scale",
    "dynamic_timbre_gate_bias",
    "dynamic_timbre_boundary_suppress_strength",
    "dynamic_timbre_boundary_radius",
    "dynamic_timbre_anchor_preserve_strength",
    "dynamic_timbre_use_tvt",
    "dynamic_timbre_tvt_prior_scale",
    "style_query_global_summary_scale",
    "dynamic_timbre_coarse_style_context_scale",
    "dynamic_timbre_query_style_condition_scale",
    "dynamic_timbre_style_context_stopgrad",
    "runtime_dynamic_timbre_style_budget_enabled",
    "runtime_dynamic_timbre_style_budget_ratio",
    "runtime_dynamic_timbre_style_budget_margin",
    "runtime_dynamic_timbre_style_budget_slow_style_weight",
    "runtime_dynamic_timbre_style_budget_epsilon",
)

ADVANCED_CONTROL_KEYS = (
    ADVANCED_CONDITION_CONTROL_KEYS
    + ADVANCED_MODEL_CONTROL_KEYS
    + ADVANCED_STYLE_RUNTIME_KEYS
)

UNSUPPORTED_INTERNAL_REQUEST_KEYS = (
    "style_condition_strength",
    "enforce_decoder_no_timing_writeback",
    "style_to_pitch_residual_apply_during_teacher_forcing",
    "upper_bound_curriculum_enabled",
    "expressive_upper_bound_curriculum_enabled",
    "upper_bound_curriculum_start_steps",
    "expressive_upper_bound_curriculum_start_steps",
    "upper_bound_curriculum_end_steps",
    "expressive_upper_bound_curriculum_end_steps",
)

ADVANCED_CONTROL_ALIASES = {
    "dynamic_timbre_strength": ("dynamic_timbre_strengths",),
    "decoder_style_condition_mode": ("style_condition_mode", "style_mainline_mode"),
    "global_timbre_to_pitch": ("global_style_anchor_to_pitch", "style_anchor_to_pitch"),
    "style_to_pitch_residual_mode": ("pitch_residual_mode",),
    "style_memory_mode": ("style_reference_memory_mode",),
    "dynamic_timbre_memory_mode": ("dynamic_timbre_reference_memory_mode",),
    "global_style_anchor_strength": ("global_timbre_strength", "style_anchor_strength"),
    "style_to_pitch_residual": ("use_style_to_pitch_residual",),
    "fast_style_strength_scale": ("fast_style_scale",),
    "slow_style_strength_scale": ("slow_style_scale",),
}

SPLIT_REFERENCE_KEYS = (
    "ref_timbre_wav",
    "ref_style_wav",
    "ref_dynamic_timbre_wav",
    "ref_emotion_wav",
    "ref_accent_wav",
)


def has_distinct_split_reference_inputs(
    source: Mapping[str, Any],
    *,
    ref_key: str = "ref_wav",
) -> bool:
    ref_value = source.get(ref_key)
    return any(source.get(key) not in (None, "", ref_value) for key in SPLIT_REFERENCE_KEYS)


def build_mainline_request_input(
    source: Mapping[str, Any],
    *,
    allow_advanced_controls: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    request = {
        "ref_wav": source["ref_wav"],
        "src_wav": source["src_wav"],
    }
    for key in PUBLIC_CONTROL_KEYS:
        value = source.get(key)
        alias_keys = PUBLIC_CONTROL_ALIASES.get(key, ())
        if value is None:
            for alias_key in alias_keys:
                alias_value = source.get(alias_key)
                if alias_value is not None:
                    value = alias_value
                    break
        if value is not None:
            request[key] = value

    ignored_advanced_keys: List[str] = []
    if allow_advanced_controls:
        for key in ADVANCED_CONTROL_KEYS:
            value = source.get(key)
            alias_keys = ADVANCED_CONTROL_ALIASES.get(key, ())
            if value is None:
                for alias_key in alias_keys:
                    alias_value = source.get(alias_key)
                    if alias_value is not None:
                        value = alias_value
                        break
            elif any(source.get(alias_key) is not None for alias_key in alias_keys):
                ignored_advanced_keys.extend(
                    alias_key
                    for alias_key in alias_keys
                    if source.get(alias_key) is not None and alias_key not in ignored_advanced_keys
                )
            if value is not None:
                request[key] = value
    else:
        ignored_advanced_keys = []
        for key in ADVANCED_CONTROL_KEYS:
            if source.get(key) is not None and key not in ignored_advanced_keys:
                ignored_advanced_keys.append(key)
            for alias_key in ADVANCED_CONTROL_ALIASES.get(key, ()):
                if source.get(alias_key) is not None and alias_key not in ignored_advanced_keys:
                    ignored_advanced_keys.append(alias_key)
    ignored_advanced_keys.extend(
        key
        for key in UNSUPPORTED_INTERNAL_REQUEST_KEYS
        if source.get(key) is not None and key not in ignored_advanced_keys
    )
    return request, ignored_advanced_keys


__all__ = [
    "ADVANCED_CONTROL_KEYS",
    "ADVANCED_CONTROL_ALIASES",
    "ADVANCED_CONDITION_CONTROL_KEYS",
    "ADVANCED_MODEL_CONTROL_KEYS",
    "ADVANCED_STYLE_RUNTIME_KEYS",
    "CONDITION_FIELDS",
    "PUBLIC_CONTROL_ALIASES",
    "PUBLIC_CONTROL_KEYS",
    "SPLIT_REFERENCE_KEYS",
    "UNSUPPORTED_INTERNAL_REQUEST_KEYS",
    "build_mainline_request_input",
    "has_distinct_split_reference_inputs",
]
