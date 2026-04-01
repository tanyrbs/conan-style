import warnings
from typing import Any, Dict, Mapping, Optional


REFERENCE_BUNDLE_KEYS = (
    "ref",
    "ref_timbre",
    "ref_style",
    "ref_dynamic_timbre",
    "ref_emotion",
    "ref_accent",
)

STYLE_RUNTIME_KEYS = (
    "decoder_style_condition_mode",
    "global_timbre_to_pitch",
    "global_style_anchor_strength",
    "style_trace_mode",
    "style_memory_mode",
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

REFERENCE_CONTRACT_MODES = (
    "collapsed_reference",
)


def first_present(mapping: Optional[Mapping[str, Any]], *keys: str, default=None):
    if mapping is None:
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def normalize_reference_contract_mode(mode, default: str = "collapsed_reference") -> str:
    normalized_default = str(default or "collapsed_reference").strip().lower() or "collapsed_reference"
    normalized = str(mode or normalized_default).strip().lower() or normalized_default
    alias_map = {
        "collapsed": "collapsed_reference",
        "single_reference": "collapsed_reference",
        "single": "collapsed_reference",
        "one_reference": "collapsed_reference",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in REFERENCE_CONTRACT_MODES:
        warnings.warn(
            f"Unknown reference_contract_mode '{mode}'. Falling back to '{normalized_default}'. "
            f"Supported modes: {', '.join(REFERENCE_CONTRACT_MODES)}.",
            stacklevel=2,
        )
        return normalized_default
    return normalized


def _build_reference_contract_metadata(
    *,
    contract_mode: str,
    explicit_ref: bool,
    explicit_timbre: bool,
    explicit_style: bool,
    explicit_dynamic_timbre: bool,
):
    collapsed_fields = []
    if not explicit_timbre:
        collapsed_fields.append("ref_timbre")
    if not explicit_style:
        collapsed_fields.append("ref_style")
    if not explicit_dynamic_timbre:
        collapsed_fields.append("ref_dynamic_timbre")
    factorization_guaranteed = False
    return {
        "mode": contract_mode,
        "factorization_semantics": "single_reference_weak_internal_factorization",
        "explicit_ref": bool(explicit_ref),
        "explicit_timbre": bool(explicit_timbre),
        "explicit_style": bool(explicit_style),
        "explicit_dynamic_timbre": bool(explicit_dynamic_timbre),
        "collapsed_fields": collapsed_fields,
        "collapsed": len(collapsed_fields) > 0,
        "factorization_guaranteed": bool(factorization_guaranteed),
        "factorization_not_guaranteed": not bool(factorization_guaranteed),
    }


def canonicalize_reference_bundle(
    bundle: Optional[Mapping[str, Any]] = None,
    *,
    default_ref=None,
    prompt_fallback_to_style: bool = False,
    contract_mode: Optional[str] = None,
):
    bundle = bundle or {}
    existing_contract = first_present(bundle, "reference_contract", default={}) or {}
    contract_mode = normalize_reference_contract_mode(
        first_present(bundle, "reference_contract_mode", default=contract_mode),
        default="collapsed_reference",
    )
    resolved_ref = first_present(bundle, "ref", default=default_ref)
    explicit_ref = resolved_ref is not None
    explicit_timbre_value = first_present(bundle, "ref_timbre", "timbre", default=None)
    if "explicit_timbre" in existing_contract:
        explicit_timbre = bool(existing_contract.get("explicit_timbre"))
    else:
        explicit_timbre = explicit_timbre_value is not None
    resolved_timbre = explicit_timbre_value if explicit_timbre_value is not None else resolved_ref
    if not explicit_timbre:
        resolved_timbre = resolved_ref

    explicit_style_value = first_present(bundle, "ref_style", "style", default=None)
    explicit_dynamic_timbre_value = first_present(bundle, "ref_dynamic_timbre", "dynamic_timbre", default=None)
    if "explicit_style" in existing_contract:
        explicit_style = bool(existing_contract.get("explicit_style"))
    else:
        explicit_style = explicit_style_value is not None
    if "explicit_dynamic_timbre" in existing_contract:
        explicit_dynamic_timbre = bool(existing_contract.get("explicit_dynamic_timbre"))
    else:
        explicit_dynamic_timbre = explicit_dynamic_timbre_value is not None

    resolved_style = explicit_style_value if explicit_style_value is not None else resolved_timbre
    if not explicit_style:
        resolved_style = resolved_timbre
    resolved_dynamic_timbre = (
        explicit_dynamic_timbre_value
        if explicit_dynamic_timbre_value is not None
        else resolved_style
    )
    if not explicit_dynamic_timbre:
        resolved_dynamic_timbre = resolved_style

    prompt_default = resolved_style if prompt_fallback_to_style else None
    resolved_emotion = first_present(bundle, "ref_emotion", "emotion", default=prompt_default)
    resolved_accent = first_present(bundle, "ref_accent", "accent", default=prompt_default)
    contract = _build_reference_contract_metadata(
        contract_mode=contract_mode,
        explicit_ref=explicit_ref,
        explicit_timbre=explicit_timbre,
        explicit_style=explicit_style,
        explicit_dynamic_timbre=explicit_dynamic_timbre,
    )

    return {
        "ref": resolved_ref,
        "ref_timbre": resolved_timbre,
        "ref_style": resolved_style,
        "ref_dynamic_timbre": resolved_dynamic_timbre,
        "ref_emotion": resolved_emotion,
        "ref_accent": resolved_accent,
        "summary_source": first_present(
            bundle,
            "summary_source",
            default=resolved_style if resolved_style is not None else resolved_timbre,
        ),
        "reference_contract_mode": contract_mode,
        "reference_contract": contract,
    }


def resolve_reference_bundle(
    source: Optional[Mapping[str, Any]] = None,
    *,
    fallback_ref=None,
    prompt_fallback_to_style: bool = False,
    contract_mode: Optional[str] = None,
):
    source = source or {}
    bundle = first_present(source, "reference_bundle", default=None)
    if isinstance(bundle, Mapping):
        return canonicalize_reference_bundle(
            bundle,
            default_ref=fallback_ref,
            prompt_fallback_to_style=prompt_fallback_to_style,
            contract_mode=first_present(source, "reference_contract_mode", default=contract_mode),
        )
    return canonicalize_reference_bundle(
        {
            "ref": first_present(source, "ref", default=fallback_ref),
            "ref_timbre": first_present(
                source,
                "ref_timbre",
                "ref_timbre_mels",
                "timbre_ref_mels",
            ),
            "ref_style": first_present(
                source,
                "ref_style",
                "ref_style_mels",
            ),
            "ref_dynamic_timbre": first_present(
                source,
                "ref_dynamic_timbre",
                "ref_dynamic_timbre_mels",
            ),
            "ref_emotion": first_present(
                source,
                "ref_emotion",
                "ref_emotion_mels",
                "emotion_ref_mels",
            ),
            "ref_accent": first_present(
                source,
                "ref_accent",
                "ref_accent_mels",
                "accent_ref_mels",
            ),
            "reference_contract_mode": first_present(
                source,
                "reference_contract_mode",
                default=contract_mode,
            ),
        },
        default_ref=fallback_ref,
        prompt_fallback_to_style=prompt_fallback_to_style,
        contract_mode=contract_mode,
    )


def build_reference_bundle_from_batch(sample: Mapping[str, Any], default_ref=None):
    raw_bundle = {
        "ref": first_present(sample, "ref", "ref_mels", default=default_ref),
        "ref_timbre": first_present(sample, "ref_timbre_mels", "timbre_ref_mels"),
        "ref_style": first_present(sample, "ref_style_mels"),
        "ref_dynamic_timbre": first_present(sample, "ref_dynamic_timbre_mels"),
        "ref_emotion": first_present(sample, "ref_emotion_mels", "emotion_ref_mels"),
        "ref_accent": first_present(sample, "ref_accent_mels", "accent_ref_mels"),
        "reference_contract_mode": first_present(sample, "reference_contract_mode"),
    }
    return canonicalize_reference_bundle(
        raw_bundle,
        default_ref=default_ref,
        contract_mode=first_present(sample, "reference_contract_mode"),
    )


def build_reference_bundle_from_inputs(
    *,
    ref=None,
    ref_timbre=None,
    ref_style=None,
    ref_dynamic_timbre=None,
    ref_emotion=None,
    ref_accent=None,
    prompt_fallback_to_style: bool = False,
    reference_contract_mode: Optional[str] = None,
):
    raw_bundle = {
        "ref": ref,
        "ref_timbre": ref_timbre,
        "ref_style": ref_style,
        "ref_dynamic_timbre": ref_dynamic_timbre,
        "ref_emotion": ref_emotion,
        "ref_accent": ref_accent,
        "reference_contract_mode": reference_contract_mode,
    }
    return canonicalize_reference_bundle(
        raw_bundle,
        default_ref=ref,
        prompt_fallback_to_style=prompt_fallback_to_style,
        contract_mode=reference_contract_mode,
    )


def build_control_kwargs(source: Mapping[str, Any], style_strength_default=1.0):
    return {
        "emotion_id": first_present(source, "emotion_id", "emotion_ids"),
        "accent_id": first_present(source, "accent_id", "accent_ids"),
        "arousal": first_present(source, "arousal"),
        "valence": first_present(source, "valence"),
        "energy": first_present(source, "energy"),
        "style_strength": first_present(
            source,
            "style_strength",
            "style_strengths",
            default=style_strength_default,
        ),
        "dynamic_timbre_strength": first_present(
            source,
            "dynamic_timbre_strength",
            "dynamic_timbre_strengths",
            default=1.0,
        ),
        "emotion_strength": first_present(
            source,
            "emotion_strength",
            "emotion_strengths",
            default=1.0,
        ),
        "accent_strength": first_present(
            source,
            "accent_strength",
            "accent_strengths",
            default=1.0,
        ),
    }


def build_style_runtime_kwargs(source: Mapping[str, Any]):
    return {
        "decoder_style_condition_mode": first_present(
            source,
            "decoder_style_condition_mode",
            "style_condition_mode",
            "style_mainline_mode",
        ),
        "global_timbre_to_pitch": first_present(
            source,
            "global_timbre_to_pitch",
            "global_style_anchor_to_pitch",
            "style_anchor_to_pitch",
        ),
        "global_style_anchor_strength": first_present(
            source,
            "global_style_anchor_strength",
            "global_timbre_strength",
            "style_anchor_strength",
        ),
        "style_trace_mode": first_present(source, "style_trace_mode"),
        "style_memory_mode": first_present(
            source,
            "style_memory_mode",
            "style_reference_memory_mode",
        ),
        "fast_style_strength_scale": first_present(source, "fast_style_strength_scale"),
        "slow_style_strength_scale": first_present(source, "slow_style_strength_scale"),
        "style_temperature": first_present(source, "style_temperature"),
        "global_style_trace_blend": first_present(source, "global_style_trace_blend"),
        "dynamic_timbre_memory_mode": first_present(
            source,
            "dynamic_timbre_memory_mode",
            "dynamic_timbre_reference_memory_mode",
        ),
        "dynamic_timbre_style_condition_scale": first_present(
            source,
            "dynamic_timbre_style_condition_scale",
        ),
        "dynamic_timbre_temperature": first_present(source, "dynamic_timbre_temperature"),
        "dynamic_timbre_gate_scale": first_present(source, "dynamic_timbre_gate_scale"),
        "dynamic_timbre_gate_bias": first_present(source, "dynamic_timbre_gate_bias"),
        "dynamic_timbre_boundary_suppress_strength": first_present(
            source,
            "dynamic_timbre_boundary_suppress_strength",
        ),
        "dynamic_timbre_boundary_radius": first_present(
            source,
            "dynamic_timbre_boundary_radius",
        ),
        "dynamic_timbre_anchor_preserve_strength": first_present(
            source,
            "dynamic_timbre_anchor_preserve_strength",
        ),
        "style_query_global_summary_scale": first_present(
            source,
            "style_query_global_summary_scale",
        ),
        "dynamic_timbre_coarse_style_context_scale": first_present(
            source,
            "dynamic_timbre_coarse_style_context_scale",
        ),
        "dynamic_timbre_style_context_stopgrad": first_present(
            source,
            "dynamic_timbre_style_context_stopgrad",
        ),
        "runtime_dynamic_timbre_style_budget_enabled": first_present(
            source,
            "runtime_dynamic_timbre_style_budget_enabled",
        ),
        "runtime_dynamic_timbre_style_budget_ratio": first_present(
            source,
            "runtime_dynamic_timbre_style_budget_ratio",
        ),
        "runtime_dynamic_timbre_style_budget_margin": first_present(
            source,
            "runtime_dynamic_timbre_style_budget_margin",
        ),
    }


def bundle_to_model_kwargs(reference_bundle: Optional[Mapping[str, Any]], **extra_kwargs):
    model_kwargs: Dict[str, Any] = {}
    if reference_bundle is not None:
        normalized_bundle = canonicalize_reference_bundle(reference_bundle)
        model_kwargs["reference_bundle"] = normalized_bundle
        contract = normalized_bundle.get("reference_contract", {}) or {}
        explicit_field_flags = {
            "ref_timbre": bool(contract.get("explicit_timbre", False)),
            "ref_style": bool(contract.get("explicit_style", False)),
            "ref_dynamic_timbre": bool(contract.get("explicit_dynamic_timbre", False)),
        }
        for key in REFERENCE_BUNDLE_KEYS:
            if key == "ref":
                continue
            if key in explicit_field_flags and not explicit_field_flags[key]:
                continue
            value = normalized_bundle.get(key, None)
            if value is not None:
                model_kwargs[key] = value
    for key, value in extra_kwargs.items():
        if value is not None:
            model_kwargs[key] = value
    return model_kwargs


def reference_bundle_to_model_kwargs(reference_bundle: Optional[Mapping[str, Any]], **extra_kwargs):
    return bundle_to_model_kwargs(reference_bundle, **extra_kwargs)
