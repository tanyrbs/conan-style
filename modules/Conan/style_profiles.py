STYLE_PROFILE_KEYS = (
    "decoder_style_condition_mode",
    "global_style_anchor_strength",
    "style_trace_mode",
    "style_memory_mode",
    "style_strength",
    "style_temperature",
    "global_style_trace_blend",
    "dynamic_timbre_memory_mode",
    "dynamic_timbre_strength",
    "dynamic_timbre_temperature",
    "dynamic_timbre_gate_scale",
    "dynamic_timbre_gate_bias",
    "dynamic_timbre_boundary_suppress_strength",
    "dynamic_timbre_boundary_radius",
    "dynamic_timbre_anchor_preserve_strength",
)


STYLE_PROFILES = {
    "balanced": {
        "decoder_style_condition_mode": "legacy_full",
        "global_style_anchor_strength": 1.0,
        "style_trace_mode": "fast",
        "style_memory_mode": "fast",
        "style_strength": 1.0,
        "style_temperature": 1.0,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.0,
        "dynamic_timbre_temperature": 1.0,
        "dynamic_timbre_gate_scale": 1.0,
        "dynamic_timbre_gate_bias": 0.0,
        "dynamic_timbre_boundary_suppress_strength": 0.0,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.0,
    },
    "strong_style": {
        "decoder_style_condition_mode": "legacy_full",
        "global_style_anchor_strength": 1.0,
        "style_trace_mode": "fast",
        "style_memory_mode": "fast",
        "style_strength": 1.35,
        "style_temperature": 1.2,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.0,
        "dynamic_timbre_temperature": 1.0,
        "dynamic_timbre_gate_scale": 1.0,
        "dynamic_timbre_gate_bias": 0.0,
        "dynamic_timbre_boundary_suppress_strength": 0.0,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.0,
    },
    "strong_timbre": {
        "decoder_style_condition_mode": "legacy_full",
        "global_style_anchor_strength": 1.15,
        "style_trace_mode": "fast",
        "style_memory_mode": "fast",
        "style_strength": 1.0,
        "style_temperature": 1.0,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.35,
        "dynamic_timbre_temperature": 1.15,
        "dynamic_timbre_gate_scale": 1.15,
        "dynamic_timbre_gate_bias": 0.04,
        "dynamic_timbre_boundary_suppress_strength": 0.12,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.12,
    },
    "strong_style_timbre": {
        "decoder_style_condition_mode": "legacy_full",
        "global_style_anchor_strength": 1.1,
        "style_trace_mode": "fast",
        "style_memory_mode": "fast",
        "style_strength": 1.35,
        "style_temperature": 1.2,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.35,
        "dynamic_timbre_temperature": 1.15,
        "dynamic_timbre_gate_scale": 1.15,
        "dynamic_timbre_gate_bias": 0.04,
        "dynamic_timbre_boundary_suppress_strength": 0.12,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.12,
    },
    "extreme": {
        "decoder_style_condition_mode": "legacy_full",
        "global_style_anchor_strength": 1.15,
        "style_trace_mode": "fast",
        "style_memory_mode": "fast",
        "style_strength": 1.55,
        "style_temperature": 1.3,
        "global_style_trace_blend": 0.05,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.55,
        "dynamic_timbre_temperature": 1.22,
        "dynamic_timbre_gate_scale": 1.25,
        "dynamic_timbre_gate_bias": 0.08,
        "dynamic_timbre_boundary_suppress_strength": 0.18,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.2,
    },
    "mainline_dynamic_timbre": {
        "decoder_style_condition_mode": "global_style_dynamic_timbre",
        "global_style_anchor_strength": 1.1,
        "style_trace_mode": "none",
        "style_memory_mode": "fast",
        "style_strength": 1.0,
        "style_temperature": 1.0,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.45,
        "dynamic_timbre_temperature": 1.18,
        "dynamic_timbre_gate_scale": 1.18,
        "dynamic_timbre_gate_bias": 0.05,
        "dynamic_timbre_boundary_suppress_strength": 0.18,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.15,
    },
    "mainline_strong_style_timbre": {
        "decoder_style_condition_mode": "mainline_full",
        "global_style_anchor_strength": 1.08,
        "style_trace_mode": "slow",
        "style_memory_mode": "slow",
        "style_strength": 1.45,
        "style_temperature": 1.22,
        "global_style_trace_blend": 0.35,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.35,
        "dynamic_timbre_temperature": 1.15,
        "dynamic_timbre_gate_scale": 1.16,
        "dynamic_timbre_gate_bias": 0.04,
        "dynamic_timbre_boundary_suppress_strength": 0.16,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.12,
    },
    "dynamic_timbre_only": {
        "decoder_style_condition_mode": "dynamic_timbre_only",
        "global_style_anchor_strength": 0.0,
        "style_trace_mode": "none",
        "style_memory_mode": "fast",
        "style_strength": 0.0,
        "style_temperature": 1.0,
        "global_style_trace_blend": 0.0,
        "dynamic_timbre_memory_mode": "fast",
        "dynamic_timbre_strength": 1.6,
        "dynamic_timbre_temperature": 1.22,
        "dynamic_timbre_gate_scale": 1.22,
        "dynamic_timbre_gate_bias": 0.08,
        "dynamic_timbre_boundary_suppress_strength": 0.2,
        "dynamic_timbre_boundary_radius": 2,
        "dynamic_timbre_anchor_preserve_strength": 0.2,
    },
}


def available_style_profiles():
    return sorted(STYLE_PROFILES.keys())


def resolve_style_profile(
    overrides=None,
    *,
    preset=None,
    default_preset="balanced",
):
    overrides = overrides or {}
    preset_name = overrides.get(
        "style_profile",
        overrides.get("style_runtime_preset", preset if preset is not None else default_preset),
    )
    if preset_name not in STYLE_PROFILES:
        preset_name = default_preset

    resolved = dict(STYLE_PROFILES[preset_name])
    for key in STYLE_PROFILE_KEYS:
        value = overrides.get(key, None)
        if value is not None:
            resolved[key] = value
    if (
        preset_name == "balanced"
        and overrides.get("style_strength", None) is not None
        and overrides.get("dynamic_timbre_strength", None) is None
    ):
        resolved["dynamic_timbre_strength"] = resolved["style_strength"]
    resolved["style_profile"] = preset_name
    return resolved


def style_profile_to_runtime_kwargs(
    overrides=None,
    *,
    preset=None,
    default_preset="balanced",
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
