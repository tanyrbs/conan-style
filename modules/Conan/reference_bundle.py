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
    "global_style_anchor_strength",
    "style_memory_mode",
    "style_temperature",
    "dynamic_timbre_memory_mode",
    "dynamic_timbre_temperature",
    "dynamic_timbre_gate_scale",
    "dynamic_timbre_gate_bias",
)

REFERENCE_CONTRACT_MODES = (
    "collapsed_reference",
    "strict_factorized",
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
        "strict": "strict_factorized",
        "factorized": "strict_factorized",
        "strict_factorized_reference": "strict_factorized",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in REFERENCE_CONTRACT_MODES:
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
    factorization_guaranteed = (
        contract_mode == "strict_factorized"
        and explicit_ref
        and explicit_timbre
        and explicit_style
        and explicit_dynamic_timbre
    )
    return {
        "mode": contract_mode,
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
    contract_mode = normalize_reference_contract_mode(
        first_present(bundle, "reference_contract_mode", default=contract_mode),
        default="collapsed_reference",
    )
    resolved_ref = first_present(bundle, "ref", default=default_ref)
    explicit_ref = resolved_ref is not None
    explicit_timbre_value = first_present(bundle, "ref_timbre", "timbre", default=None)
    explicit_timbre = explicit_timbre_value is not None
    resolved_timbre = explicit_timbre_value if explicit_timbre_value is not None else resolved_ref

    explicit_style_value = first_present(bundle, "ref_style", "style", default=None)
    explicit_dynamic_timbre_value = first_present(bundle, "ref_dynamic_timbre", "dynamic_timbre", default=None)
    explicit_style = explicit_style_value is not None
    explicit_dynamic_timbre = explicit_dynamic_timbre_value is not None

    if contract_mode == "strict_factorized":
        if resolved_ref is None:
            raise ValueError("strict_factorized reference contract requires `ref`.")
        if explicit_timbre_value is None:
            raise ValueError("strict_factorized reference contract requires explicit `ref_timbre`.")
        if explicit_style_value is None:
            raise ValueError("strict_factorized reference contract requires explicit `ref_style`.")
        if explicit_dynamic_timbre_value is None:
            raise ValueError("strict_factorized reference contract requires explicit `ref_dynamic_timbre`.")
        resolved_timbre = explicit_timbre_value
        resolved_style = explicit_style_value
        resolved_dynamic_timbre = explicit_dynamic_timbre_value
    else:
        resolved_style = explicit_style_value if explicit_style_value is not None else resolved_timbre
        resolved_dynamic_timbre = (
            explicit_dynamic_timbre_value
            if explicit_dynamic_timbre_value is not None
            else resolved_style
        )

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
                "style_ref_mels",
            ),
            "ref_dynamic_timbre": first_present(
                source,
                "ref_dynamic_timbre",
                "ref_dynamic_timbre_mels",
                "dynamic_timbre_ref_mels",
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
        "ref_style": first_present(sample, "ref_style_mels", "style_ref_mels"),
        "ref_dynamic_timbre": first_present(
            sample,
            "ref_dynamic_timbre_mels",
            "dynamic_timbre_ref_mels",
        ),
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
        "style_id": first_present(source, "style_id", "style_ids"),
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
        "dynamic_timbre_strength": first_present(
            source,
            "dynamic_timbre_strength",
            "dynamic_timbre_strengths",
            default=style_strength_default,
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
        "global_style_anchor_strength": first_present(
            source,
            "global_style_anchor_strength",
            "global_timbre_strength",
            "style_anchor_strength",
        ),
        "style_memory_mode": first_present(
            source,
            "style_memory_mode",
            "style_reference_memory_mode",
        ),
        "style_temperature": first_present(source, "style_temperature"),
        "dynamic_timbre_memory_mode": first_present(
            source,
            "dynamic_timbre_memory_mode",
            "dynamic_timbre_reference_memory_mode",
        ),
        "dynamic_timbre_temperature": first_present(source, "dynamic_timbre_temperature"),
        "dynamic_timbre_gate_scale": first_present(source, "dynamic_timbre_gate_scale"),
        "dynamic_timbre_gate_bias": first_present(source, "dynamic_timbre_gate_bias"),
    }


def bundle_to_model_kwargs(reference_bundle: Optional[Mapping[str, Any]], **extra_kwargs):
    model_kwargs: Dict[str, Any] = {}
    if reference_bundle is not None:
        normalized_bundle = canonicalize_reference_bundle(reference_bundle)
        model_kwargs["reference_bundle"] = normalized_bundle
        for key in REFERENCE_BUNDLE_KEYS:
            if key == "ref":
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
