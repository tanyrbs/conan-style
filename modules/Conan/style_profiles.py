import warnings


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
    "dynamic_timbre_strength",
    "dynamic_timbre_memory_mode",
    "dynamic_timbre_style_condition_scale",
    "dynamic_timbre_temperature",
    "dynamic_timbre_gate_scale",
    "dynamic_timbre_gate_bias",
    "dynamic_timbre_boundary_suppress_strength",
    "dynamic_timbre_boundary_radius",
    "dynamic_timbre_anchor_preserve_strength",
    "style_query_global_summary_scale",
    "dynamic_timbre_coarse_style_context_scale",
    "dynamic_timbre_style_context_stopgrad",
    "runtime_dynamic_timbre_style_budget_enabled",
    "runtime_dynamic_timbre_style_budget_ratio",
    "runtime_dynamic_timbre_style_budget_margin",
)


STYLE_PROFILES = {
    "strong_style": {
        "decoder_style_condition_mode": "mainline_full",
        "global_timbre_to_pitch": False,
        "global_style_anchor_strength": 1.0,
        "style_trace_mode": "dual",
        "style_memory_mode": "slow",
        "style_strength": 1.35,
        "fast_style_strength_scale": 1.15,
        "slow_style_strength_scale": 1.3,
        "style_temperature": 1.2,
        "global_style_trace_blend": 0.25,
        "dynamic_timbre_strength": 1.0,
        "dynamic_timbre_memory_mode": "slow",
        "dynamic_timbre_style_condition_scale": 0.5,
        "dynamic_timbre_temperature": 1.0,
        "dynamic_timbre_gate_scale": 1.0,
        "dynamic_timbre_gate_bias": 0.0,
        "dynamic_timbre_boundary_suppress_strength": 0.5,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.2,
        "style_query_global_summary_scale": 0.35,
        "dynamic_timbre_coarse_style_context_scale": 0.10,
        "dynamic_timbre_style_context_stopgrad": True,
        "runtime_dynamic_timbre_style_budget_enabled": True,
        "runtime_dynamic_timbre_style_budget_ratio": 0.90,
        "runtime_dynamic_timbre_style_budget_margin": 0.04,
    },
    "extreme": {
        "decoder_style_condition_mode": "mainline_full",
        "global_timbre_to_pitch": False,
        "global_style_anchor_strength": 1.15,
        "style_trace_mode": "dual",
        "style_memory_mode": "slow",
        "style_strength": 1.55,
        "fast_style_strength_scale": 1.15,
        "slow_style_strength_scale": 1.3,
        "style_temperature": 1.3,
        "global_style_trace_blend": 0.05,
        "dynamic_timbre_strength": 1.1,
        "dynamic_timbre_memory_mode": "slow",
        "dynamic_timbre_style_condition_scale": 0.6,
        "dynamic_timbre_temperature": 1.05,
        "dynamic_timbre_gate_scale": 1.0,
        "dynamic_timbre_gate_bias": 0.0,
        "dynamic_timbre_boundary_suppress_strength": 0.45,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.18,
        "style_query_global_summary_scale": 0.35,
        "dynamic_timbre_coarse_style_context_scale": 0.10,
        "dynamic_timbre_style_context_stopgrad": True,
        "runtime_dynamic_timbre_style_budget_enabled": True,
        "runtime_dynamic_timbre_style_budget_ratio": 0.95,
        "runtime_dynamic_timbre_style_budget_margin": 0.05,
    },
}


def available_style_profiles():
    return sorted(STYLE_PROFILES.keys())


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
    for key in STYLE_PROFILE_KEYS:
        value = overrides.get(key, None)
        if value is not None:
            resolved[key] = value
    resolved["style_profile"] = preset_name
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
