STYLE_REGULARIZATION_LAMBDAS = (
    "lambda_style_trace_consistency",
)

TIMBRE_REGULARIZATION_LAMBDAS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_gate",
    "lambda_dynamic_timbre_budget",
    "lambda_dynamic_timbre_boundary",
    "lambda_dynamic_timbre_anchor",
    "lambda_pitch_residual_safe",
    "lambda_gate_rank",
    "lambda_decoder_late_owner",
    "lambda_decoder_late_anchor_budget",
)

DEFAULT_SCHEDULED_CONTROL_LAMBDAS = (
    "lambda_style_trace_consistency",
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_gate",
    "lambda_dynamic_timbre_budget",
    "lambda_dynamic_timbre_boundary",
    "lambda_dynamic_timbre_anchor",
    "lambda_pitch_residual_safe",
    "lambda_gate_rank",
    "lambda_decoder_late_owner",
    "lambda_decoder_late_anchor_budget",
)

MAINLINE_MINIMAL_CONTROL_LAMBDAS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_budget",
    "lambda_pitch_residual_safe",
    "lambda_decoder_late_owner",
)

VALID_CONTROL_LOSS_PROFILES = (
    "mainline_minimal",
    "full",
)


def resolve_control_loss_profile(config, default="mainline_minimal"):
    profile = str((config or {}).get("control_loss_profile", default) or default).strip().lower()
    if profile in {"minimal", "core", "mainline"}:
        profile = "mainline_minimal"
    if profile not in VALID_CONTROL_LOSS_PROFILES:
        return str(default or "mainline_minimal")
    return profile


def linear_schedule_scale(
    global_step,
    *,
    start_steps=0,
    warmup_steps=1,
    init_scale=1.0,
    final_scale=1.0,
):
    global_step = max(int(global_step), 0)
    start_steps = max(int(start_steps), 0)
    warmup_steps = int(warmup_steps)
    init_scale = float(init_scale)
    final_scale = float(final_scale)

    if global_step <= start_steps:
        return init_scale
    if warmup_steps <= 0:
        return final_scale

    progress = min(1.0, float(global_step - start_steps) / float(warmup_steps))
    return init_scale + (final_scale - init_scale) * progress


def _apply_group_schedule(base_config, scheduled_config, global_step, group_name, lambda_keys):
    scale = linear_schedule_scale(
        global_step,
        start_steps=base_config.get(f"{group_name}_start_steps", 0),
        warmup_steps=base_config.get(f"{group_name}_warmup_steps", 1),
        init_scale=base_config.get(f"{group_name}_init_scale", 1.0),
        final_scale=base_config.get(f"{group_name}_final_scale", 1.0),
    )
    scheduled_config[f"{group_name}_effective_scale"] = float(scale)
    for key in lambda_keys:
        scheduled_config[key] = float(base_config.get(key, 0.0)) * float(scale)


def build_scheduled_control_config(config, global_step):
    scheduled_config = dict(config)
    _apply_group_schedule(
        config,
        scheduled_config,
        global_step,
        "style_reg",
        STYLE_REGULARIZATION_LAMBDAS,
    )
    _apply_group_schedule(
        config,
        scheduled_config,
        global_step,
        "timbre_reg",
        TIMBRE_REGULARIZATION_LAMBDAS,
    )
    return scheduled_config


def _schedule_progress(global_step, start, warmup):
    if global_step <= start:
        return 0.0
    if warmup <= 0:
        return 1.0
    return max(0.0, min(float(global_step - start) / float(warmup), 1.0))


def _resolve_scheduled_lambda(base_value, schedule_spec, global_step, default_type="linear"):
    if not isinstance(schedule_spec, dict):
        return float(base_value)

    schedule_type = str(schedule_spec.get("type", default_type)).lower()
    start = int(schedule_spec.get("start", 0))
    warmup = int(schedule_spec.get("warmup", 0))
    initial = float(schedule_spec.get("initial", 0.0))
    if "target" in schedule_spec:
        target = float(schedule_spec["target"])
    elif "value" in schedule_spec:
        target = float(schedule_spec["value"])
    elif float(base_value) != 0.0:
        target = float(base_value)
    elif any(key in schedule_spec for key in ("start", "warmup", "initial")):
        raise ValueError(
            "Scheduled control lambda with base value 0.0 now requires an explicit `target` or `value`."
        )
    else:
        target = float(base_value)
    progress = _schedule_progress(int(global_step), start, warmup)

    if schedule_type == "linear":
        return initial + (target - initial) * progress
    if schedule_type == "step":
        return target if progress >= 1.0 else initial
    return float(base_value)


def resolve_control_regularization_config(config, global_step, *, schedule_key="control_regularization_schedule"):
    resolved = build_scheduled_control_config(config, global_step)
    schedule_config = config.get(schedule_key, {})
    if not isinstance(schedule_config, dict):
        schedule_config = {}
    default_type = str(schedule_config.get("type", "linear")).lower()
    for key in DEFAULT_SCHEDULED_CONTROL_LAMBDAS:
        if key not in resolved:
            continue
        resolved[key] = _resolve_scheduled_lambda(
            resolved[key],
            schedule_config.get(key),
            global_step,
            default_type=default_type,
        )
    profile = resolve_control_loss_profile(config)
    resolved["control_loss_profile"] = profile
    if profile == "mainline_minimal":
        allowed = set(MAINLINE_MINIMAL_CONTROL_LAMBDAS)
        for key in DEFAULT_SCHEDULED_CONTROL_LAMBDAS:
            if key not in allowed:
                resolved[key] = 0.0
    return resolved
