import warnings

from modules.Conan.style_mainline import (
    derive_dynamic_timbre_strength,
    normalize_decoder_style_condition_mode,
    normalize_style_trace_mode,
)


STYLE_PROFILE_KEYS = (
    "decoder_style_condition_mode",
    "global_timbre_to_pitch",
    "global_style_anchor_strength",
    "style_trace_mode",
    "style_memory_mode",
    "style_strength",
    "fast_style_strength_scale",
    "slow_style_strength_scale",
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
)


STYLE_PROFILES = {
    "strong_style": {
        "track": "mainline",
        "decoder_style_condition_mode": "mainline_full",
        "global_timbre_to_pitch": False,
        "global_style_anchor_strength": 1.0,
        "style_trace_mode": "slow",
        "style_memory_mode": "slow",
        "style_strength": 1.35,
        "fast_style_strength_scale": 1.0,
        "slow_style_strength_scale": 1.45,
        "style_temperature": 1.2,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "slow",
        "dynamic_timbre_style_condition_scale": 0.35,
        "dynamic_timbre_temperature": 1.0,
        "dynamic_timbre_gate_scale": 1.0,
        "dynamic_timbre_gate_bias": 0.0,
        "dynamic_timbre_boundary_suppress_strength": 0.5,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.2,
        "dynamic_timbre_use_tvt": True,
        "dynamic_timbre_tvt_prior_scale": 1.0,
        "style_query_global_summary_scale": 0.0,
        "dynamic_timbre_coarse_style_context_scale": 0.0,
        "dynamic_timbre_query_style_condition_scale": 0.0,
        "dynamic_timbre_style_context_stopgrad": True,
        "runtime_dynamic_timbre_style_budget_enabled": True,
        "runtime_dynamic_timbre_style_budget_ratio": 0.40,
        "runtime_dynamic_timbre_style_budget_margin": 0.0,
    },
    "research_dual": {
        "track": "research",
        "decoder_style_condition_mode": "mainline_full",
        "global_timbre_to_pitch": False,
        "global_style_anchor_strength": 1.0,
        "style_trace_mode": "dual",
        "style_memory_mode": "slow",
        "style_strength": 1.35,
        "fast_style_strength_scale": 1.0,
        "slow_style_strength_scale": 1.35,
        "style_temperature": 1.2,
        "global_style_trace_blend": 0.25,
        "dynamic_timbre_memory_mode": "slow",
        "dynamic_timbre_style_condition_scale": 0.5,
        "dynamic_timbre_temperature": 1.0,
        "dynamic_timbre_gate_scale": 1.0,
        "dynamic_timbre_gate_bias": 0.0,
        "dynamic_timbre_boundary_suppress_strength": 0.5,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.2,
        "dynamic_timbre_use_tvt": True,
        "dynamic_timbre_tvt_prior_scale": 1.0,
        "style_query_global_summary_scale": 0.0,
        "dynamic_timbre_coarse_style_context_scale": 0.0,
        "dynamic_timbre_query_style_condition_scale": 0.0,
        "dynamic_timbre_style_context_stopgrad": True,
        "runtime_dynamic_timbre_style_budget_enabled": True,
        "runtime_dynamic_timbre_style_budget_ratio": 0.55,
        "runtime_dynamic_timbre_style_budget_margin": 0.02,
    },
    "extreme": {
        "track": "mainline",
        "decoder_style_condition_mode": "mainline_full",
        "global_timbre_to_pitch": False,
        "global_style_anchor_strength": 1.15,
        "style_trace_mode": "slow",
        "style_memory_mode": "slow",
        "style_strength": 1.55,
        "fast_style_strength_scale": 1.0,
        "slow_style_strength_scale": 1.45,
        "style_temperature": 1.3,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "slow",
        "dynamic_timbre_style_condition_scale": 0.6,
        "dynamic_timbre_temperature": 1.05,
        "dynamic_timbre_gate_scale": 1.0,
        "dynamic_timbre_gate_bias": 0.0,
        "dynamic_timbre_boundary_suppress_strength": 0.45,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.18,
        "dynamic_timbre_use_tvt": True,
        "dynamic_timbre_tvt_prior_scale": 1.05,
        "style_query_global_summary_scale": 0.0,
        "dynamic_timbre_coarse_style_context_scale": 0.0,
        "dynamic_timbre_query_style_condition_scale": 0.0,
        "dynamic_timbre_style_context_stopgrad": True,
        "runtime_dynamic_timbre_style_budget_enabled": True,
        "runtime_dynamic_timbre_style_budget_ratio": 0.50,
        "runtime_dynamic_timbre_style_budget_margin": 0.0,
    },
}


def _style_profile_track(name, profile=None):
    if isinstance(profile, dict) and profile.get("track") is not None:
        return str(profile.get("track"))
    return "research" if str(name).startswith("research_") else "mainline"


def available_style_profiles():
    return sorted(STYLE_PROFILES.keys())


def available_mainline_style_profiles():
    return [
        name for name in available_style_profiles()
        if _style_profile_track(name, STYLE_PROFILES.get(name)) == "mainline"
    ]


def resolve_style_profile(
    overrides=None,
    *,
    preset=None,
    default_preset="strong_style",
):
    overrides = overrides or {}
    preset_name = overrides.get(
        "style_profile",
        overrides.get("style_runtime_preset", preset if preset is not None else default_preset),
    )
    if preset_name not in STYLE_PROFILES:
        warnings.warn(
            f"Unknown style_profile '{preset_name}'. Falling back to '{default_preset}'. "
            f"Available profiles: {', '.join(available_style_profiles())}.",
            stacklevel=2,
        )
        preset_name = default_preset

    resolved = dict(STYLE_PROFILES[preset_name])
    profile_track = _style_profile_track(preset_name, resolved)
    allow_mainline_profile_research_overrides = bool(
        overrides.get("allow_mainline_profile_research_overrides", False)
    )
    for key in STYLE_PROFILE_KEYS:
        value = overrides.get(key, None)
        if value is not None:
            resolved[key] = value
    normalized_mode = normalize_decoder_style_condition_mode(
        resolved.get("decoder_style_condition_mode", "mainline_full"),
        default="mainline_full",
    )
    normalized_trace_mode = normalize_style_trace_mode(
        resolved.get("style_trace_mode", "slow"),
        default="slow",
    )
    if profile_track == "mainline":
        research_like_override = (
            normalized_mode != "mainline_full"
            or normalized_trace_mode == "dual"
            or float(resolved.get("global_style_trace_blend", 0.0)) > 0.0
            or float(resolved.get("style_query_global_summary_scale", 0.0)) != 0.0
            or str(resolved.get("style_memory_mode", "slow")).strip().lower() != "slow"
            or str(resolved.get("dynamic_timbre_memory_mode", "slow")).strip().lower() != "slow"
            or float(resolved.get("dynamic_timbre_coarse_style_context_scale", 0.0)) != 0.0
            or float(resolved.get("dynamic_timbre_query_style_condition_scale", 0.0)) != 0.0
            or not bool(resolved.get("dynamic_timbre_style_context_stopgrad", True))
            or not bool(resolved.get("dynamic_timbre_use_tvt", True))
            or float(resolved.get("dynamic_timbre_tvt_prior_scale", 1.0)) != 1.0
            or float(resolved.get("runtime_dynamic_timbre_style_budget_ratio", 0.40)) != 0.40
            or float(resolved.get("runtime_dynamic_timbre_style_budget_margin", 0.0)) != 0.0
            or not bool(resolved.get("runtime_dynamic_timbre_style_budget_enabled", True))
        )
        if research_like_override and not allow_mainline_profile_research_overrides:
            warnings.warn(
                f"Mainline style_profile '{preset_name}' received research-style overrides; coercing back to canonical mainline defaults. "
                "Use 'research_dual' or set allow_mainline_profile_research_overrides=True to keep them.",
                stacklevel=2,
            )
            resolved["decoder_style_condition_mode"] = "mainline_full"
            resolved["style_trace_mode"] = "slow"
            resolved["global_style_trace_blend"] = 0.0
            resolved["style_query_global_summary_scale"] = 0.0
            resolved["style_memory_mode"] = "slow"
            resolved["dynamic_timbre_memory_mode"] = "slow"
            resolved["dynamic_timbre_coarse_style_context_scale"] = 0.0
            resolved["dynamic_timbre_query_style_condition_scale"] = 0.0
            resolved["dynamic_timbre_style_context_stopgrad"] = True
            resolved["dynamic_timbre_use_tvt"] = True
            resolved["dynamic_timbre_tvt_prior_scale"] = 1.0
            resolved["runtime_dynamic_timbre_style_budget_enabled"] = True
            resolved["runtime_dynamic_timbre_style_budget_ratio"] = 0.40
            resolved["runtime_dynamic_timbre_style_budget_margin"] = 0.0
            normalized_mode = "mainline_full"
            normalized_trace_mode = "slow"
        elif research_like_override:
            warnings.warn(
                f"Mainline style_profile '{preset_name}' is being used with research-style overrides. "
                "Prefer the dedicated research profile if this is intentional.",
                stacklevel=2,
            )
            profile_track = "research_override"
    allow_explicit_dynamic_timbre_strength = bool(
        overrides.get("allow_explicit_dynamic_timbre_strength", False)
    )
    explicit_dynamic_timbre_strength = overrides.get("dynamic_timbre_strength", None)
    if allow_explicit_dynamic_timbre_strength and explicit_dynamic_timbre_strength is not None:
        resolved["dynamic_timbre_strength"] = explicit_dynamic_timbre_strength
        resolved["dynamic_timbre_strength_source"] = "explicit_runtime_override"
    else:
        resolved["dynamic_timbre_strength"] = derive_dynamic_timbre_strength(
            resolved.get("style_strength", 1.0)
        )
        resolved["dynamic_timbre_strength_source"] = "derived_from_style_strength"
    resolved["style_profile"] = preset_name
    resolved["style_profile_track"] = profile_track
    return resolved


def style_profile_to_runtime_kwargs(
    overrides=None,
    *,
    preset=None,
    default_preset="strong_style",
):
    resolved = resolve_style_profile(
        overrides=overrides,
        preset=preset,
        default_preset=default_preset,
    )
    runtime_kwargs = {
        key: resolved.get(key)
        for key in STYLE_PROFILE_KEYS
        if resolved.get(key) is not None
    }
    runtime_kwargs["style_profile"] = resolved["style_profile"]
    return runtime_kwargs
