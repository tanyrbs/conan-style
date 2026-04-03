import warnings

from modules.Conan.style_mainline import (
    derive_dynamic_timbre_strength,
    normalize_decoder_style_condition_mode,
    sanitize_mainline_style_strength,
    sanitize_scalar_range,
)


STYLE_PROFILE_KEYS = (
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
    "style_strength",
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
)


STYLE_PROFILES = {
    "strong_style": {
        "track": "mainline",
        "decoder_style_condition_mode": "mainline_full",
        "global_timbre_to_pitch": False,
        "style_to_pitch_residual": True,
        "style_to_pitch_residual_include_timbre": False,
        "style_to_pitch_residual_mode": "auto",
        "style_to_pitch_residual_scale": 1.0,
        "style_to_pitch_residual_max_semitones": 2.5,
        "style_to_pitch_residual_smooth_factor": 0.35,
        "global_style_anchor_strength": 1.0,
        "style_trace_mode": "dual",
        "style_memory_mode": "slow",
        "style_strength": 1.35,
        "fast_style_strength_scale": 1.10,
        "slow_style_strength_scale": 1.20,
        "style_router_enabled": True,
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
        "style_to_pitch_residual": False,
        "style_to_pitch_residual_include_timbre": False,
        "style_to_pitch_residual_mode": "auto",
        "style_to_pitch_residual_scale": 1.0,
        "style_to_pitch_residual_max_semitones": 2.5,
        "style_to_pitch_residual_smooth_factor": 0.35,
        "global_style_anchor_strength": 1.0,
        "style_trace_mode": "dual",
        "style_memory_mode": "slow",
        "style_strength": 1.35,
        "fast_style_strength_scale": 1.0,
        "slow_style_strength_scale": 1.35,
        "style_router_enabled": True,
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
        "style_to_pitch_residual": True,
        "style_to_pitch_residual_include_timbre": False,
        "style_to_pitch_residual_mode": "auto",
        "style_to_pitch_residual_scale": 1.20,
        "style_to_pitch_residual_max_semitones": 4.0,
        "style_to_pitch_residual_smooth_factor": 0.25,
        "global_style_anchor_strength": 1.15,
        "style_trace_mode": "dual",
        "style_memory_mode": "slow",
        "style_strength": 1.55,
        "fast_style_strength_scale": 1.20,
        "slow_style_strength_scale": 1.25,
        "style_router_enabled": True,
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

    base_profile = dict(STYLE_PROFILES[preset_name])
    resolved = dict(base_profile)
    profile_track = _style_profile_track(preset_name, resolved)
    allow_mainline_profile_research_overrides = bool(
        overrides.get("allow_mainline_profile_research_overrides", False)
    )
    explicit_override_keys = {
        key for key in STYLE_PROFILE_KEYS
        if overrides.get(key, None) is not None
    }
    for key in STYLE_PROFILE_KEYS:
        value = overrides.get(key, None)
        if value is not None:
            resolved[key] = value
    requested_style_strength = resolved.get("style_strength", base_profile.get("style_strength", 1.0))
    resolved["style_strength_requested"] = requested_style_strength
    resolved["style_strength"] = sanitize_mainline_style_strength(
        requested_style_strength,
        default=float(base_profile.get("style_strength", 1.0)),
    )
    resolved["style_strength_effective"] = resolved["style_strength"]
    try:
        style_strength_was_clamped = (
            abs(float(requested_style_strength) - float(resolved["style_strength"])) > 1e-8
        )
    except (TypeError, ValueError):
        style_strength_was_clamped = False
    resolved["style_strength_was_clamped"] = bool(style_strength_was_clamped)
    if style_strength_was_clamped:
        warnings.warn(
            f"style_strength={requested_style_strength} is outside the approved mainline range "
            f"and was clamped to {resolved['style_strength']:.3f}.",
            stacklevel=2,
        )
    normalized_mode = normalize_decoder_style_condition_mode(
        resolved.get("decoder_style_condition_mode", "mainline_full"),
        default="mainline_full",
    )
    if profile_track == "mainline":
        allowed_mainline_override_keys = {"style_strength"}
        research_override_keys = sorted(
            key
            for key in (explicit_override_keys - allowed_mainline_override_keys)
            if resolved.get(key) != base_profile.get(key)
        )
        research_like_override = len(research_override_keys) > 0
        if research_like_override and not allow_mainline_profile_research_overrides:
            warnings.warn(
                f"Mainline style_profile '{preset_name}' received research-style overrides ({', '.join(research_override_keys)}); "
                "coercing back to the profile's approved mainline defaults. "
                "Use 'research_dual' or set allow_mainline_profile_research_overrides=True to keep them.",
                stacklevel=2,
            )
            for key in research_override_keys:
                if key in base_profile:
                    resolved[key] = base_profile[key]
            normalized_mode = normalize_decoder_style_condition_mode(
                resolved.get("decoder_style_condition_mode", "mainline_full"),
                default="mainline_full",
            )
        elif research_like_override:
            warnings.warn(
                f"Mainline style_profile '{preset_name}' is being used with research-style overrides ({', '.join(research_override_keys)}). "
                "Prefer the dedicated research profile if this is intentional.",
                stacklevel=2,
            )
            profile_track = "research_override"
    allow_explicit_dynamic_timbre_strength = bool(
        overrides.get("allow_explicit_dynamic_timbre_strength", False)
    )
    explicit_dynamic_timbre_strength = overrides.get("dynamic_timbre_strength", None)
    if allow_explicit_dynamic_timbre_strength and explicit_dynamic_timbre_strength is not None:
        resolved["dynamic_timbre_strength"] = sanitize_scalar_range(
            explicit_dynamic_timbre_strength,
            default=float(resolved.get("dynamic_timbre_strength", 1.0)),
            min_value=0.50,
            max_value=1.05,
        )
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
