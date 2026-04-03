"""Shared single-reference Conan inference request schema helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

CONDITION_FIELDS = ("emotion", "accent")

PUBLIC_CONTROL_KEYS = (
    "style_profile",
    "style_strength",
)

ADVANCED_CONTROL_KEYS = (
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
        if value is not None:
            request[key] = value

    ignored_advanced_keys: List[str] = []
    if allow_advanced_controls:
        for key in ADVANCED_CONTROL_KEYS:
            value = source.get(key)
            if value is not None:
                request[key] = value
    else:
        ignored_advanced_keys = [key for key in ADVANCED_CONTROL_KEYS if source.get(key) is not None]
    return request, ignored_advanced_keys


__all__ = [
    "ADVANCED_CONTROL_KEYS",
    "CONDITION_FIELDS",
    "PUBLIC_CONTROL_KEYS",
    "SPLIT_REFERENCE_KEYS",
    "build_mainline_request_input",
    "has_distinct_split_reference_inputs",
]
