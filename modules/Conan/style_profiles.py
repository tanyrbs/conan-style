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
