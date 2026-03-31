from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


STYLE_MAINLINE_OWNER = "global_style_plus_dynamic_timbre"
STYLE_MAINLINE_SURFACE = "ConanStrongStyleMainlineSurface"
STYLE_MAINLINE_MEMORY = "ConanStrongStyleMainlineMemory"

VALID_DECODER_STYLE_CONDITION_MODES = (
    "legacy_full",
    "mainline_full",
    "global_style_dynamic_timbre",
    "global_only",
    "dynamic_timbre_only",
)

VALID_STYLE_TRACE_MODES = (
    "none",
    "fast",
    "slow",
    "dual",
)


def _first_present(source: Optional[Mapping[str, Any]], *keys: str, default=None):
    if not isinstance(source, Mapping):
        return default
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def normalize_decoder_style_condition_mode(mode, default: str = "legacy_full") -> str:
    normalized_default = str(default or "legacy_full").strip().lower() or "legacy_full"
    normalized = str(mode or normalized_default).strip().lower() or normalized_default
    alias_map = {
        "default": "legacy_full",
        "full": "legacy_full",
        "legacy": "legacy_full",
        "legacy_full_conditioning": "legacy_full",
        "mainline_full_stack": "mainline_full",
        "global_style_trace_dynamic_timbre": "mainline_full",
        "global_style_plus_trace_dynamic_timbre": "mainline_full",
        "gstdt": "mainline_full",
        "mainline": "global_style_dynamic_timbre",
        "bridge": "global_style_dynamic_timbre",
        "gsdt": "global_style_dynamic_timbre",
        "global_dynamic_timbre": "global_style_dynamic_timbre",
        "global_style_plus_dynamic_timbre": "global_style_dynamic_timbre",
        "global_anchor_only": "global_only",
        "anchor_only": "global_only",
        "timbre_only": "dynamic_timbre_only",
        "dynamic_only": "dynamic_timbre_only",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in VALID_DECODER_STYLE_CONDITION_MODES:
        return normalized_default
    return normalized


def normalize_style_trace_mode(mode, default: str = "fast") -> str:
    normalized_default = str(default or "fast").strip().lower() or "fast"
    normalized = str(mode or normalized_default).strip().lower() or normalized_default
    alias_map = {
        "off": "none",
        "disabled": "none",
        "false": "none",
        "default": normalized_default,
        "on": normalized_default,
        "true": normalized_default,
        "local": "fast",
        "full": "fast",
        "mainline": "slow",
        "coarse": "slow",
        "hybrid": "dual",
        "slow_fast": "dual",
        "dual_path": "dual",
        "dual_memory": "dual",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in VALID_STYLE_TRACE_MODES:
        return normalized_default
    return normalized


@dataclass(frozen=True)
class StyleMainlineControls:
    mode: str = "legacy_full"
    apply_global_style_anchor: bool = True
    apply_style_trace: bool = True
    apply_dynamic_timbre: bool = True
    global_timbre_to_pitch: bool = True
    global_style_anchor_strength: float = 1.0
    style_strength: float = 1.0
    dynamic_timbre_strength: float = 1.0
    style_trace_mode: str = "fast"
    style_memory_mode: str = "fast"
    dynamic_timbre_memory_mode: str = "fast"
    style_temperature: float = 1.0
    global_style_trace_blend: float = 0.0
    dynamic_timbre_temperature: float = 1.0
    dynamic_timbre_gate_scale: float = 1.0
    dynamic_timbre_gate_bias: float = 0.0
    dynamic_timbre_boundary_suppress_strength: float = 0.0
    dynamic_timbre_boundary_radius: int = 2
    dynamic_timbre_anchor_preserve_strength: float = 0.0
    enforce_decoder_no_timing_writeback: bool = True
    mainline_owner: str = STYLE_MAINLINE_OWNER

    def as_dict(self):
        return asdict(self)


def resolve_style_mainline_controls(
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    hparams: Optional[Mapping[str, Any]] = None,
    default_mode: str = "legacy_full",
) -> StyleMainlineControls:
    mode = normalize_decoder_style_condition_mode(
        _first_present(
            overrides,
            "decoder_style_condition_mode",
            "style_condition_mode",
            "style_mainline_mode",
            default=_first_present(
                hparams,
                "decoder_style_condition_mode",
                "style_condition_mode",
                "style_mainline_mode",
                default=default_mode,
            ),
        ),
        default=default_mode,
    )

    mode_flags = {
        "legacy_full": (True, True, True),
        "mainline_full": (True, True, True),
        "global_style_dynamic_timbre": (True, False, True),
        "global_only": (True, False, False),
        "dynamic_timbre_only": (False, False, True),
    }
    apply_global_style_anchor, apply_style_trace, apply_dynamic_timbre = mode_flags[mode]
    default_style_trace_mode = {
        "legacy_full": "fast",
        "mainline_full": "slow",
        "global_style_dynamic_timbre": "none",
        "global_only": "none",
        "dynamic_timbre_only": "none",
    }[mode]
    default_global_timbre_to_pitch = {
        "legacy_full": True,
        "mainline_full": False,
        "global_style_dynamic_timbre": False,
        "global_only": False,
        "dynamic_timbre_only": False,
    }[mode]

    def _value(*keys: str, default=None):
        return _first_present(
            overrides,
            *keys,
            default=_first_present(hparams, *keys, default=default),
        )

    def _raw_or_float(*keys: str, default=None):
        value = _value(*keys, default=default)
        if isinstance(value, (int, float)):
            return float(value)
        return value

    return StyleMainlineControls(
        mode=mode,
        apply_global_style_anchor=apply_global_style_anchor,
        apply_style_trace=apply_style_trace,
        apply_dynamic_timbre=apply_dynamic_timbre,
        global_timbre_to_pitch=bool(
            _value(
                "global_timbre_to_pitch",
                "global_style_anchor_to_pitch",
                "style_anchor_to_pitch",
                default=default_global_timbre_to_pitch,
            )
        ),
        global_style_anchor_strength=_raw_or_float(
            "global_style_anchor_strength",
            "global_timbre_strength",
            "style_anchor_strength",
            default=1.0,
        ),
        style_strength=_raw_or_float("style_strength", "style_strengths", default=1.0),
        dynamic_timbre_strength=_raw_or_float(
            "dynamic_timbre_strength", "dynamic_timbre_strengths", default=1.0
        ),
        style_trace_mode=normalize_style_trace_mode(
            _value("style_trace_mode", default=default_style_trace_mode),
            default=default_style_trace_mode,
        ),
        style_memory_mode=str(_value("style_memory_mode", "style_reference_memory_mode", default="fast")),
        dynamic_timbre_memory_mode=str(
            _value(
                "dynamic_timbre_memory_mode",
                "dynamic_timbre_reference_memory_mode",
                default="fast",
            )
        ),
        style_temperature=float(_value("style_temperature", default=1.0)),
        global_style_trace_blend=float(_value("global_style_trace_blend", default=0.0)),
        dynamic_timbre_temperature=float(_value("dynamic_timbre_temperature", default=1.0)),
        dynamic_timbre_gate_scale=float(_value("dynamic_timbre_gate_scale", default=1.0)),
        dynamic_timbre_gate_bias=float(_value("dynamic_timbre_gate_bias", default=0.0)),
        dynamic_timbre_boundary_suppress_strength=float(
            _value("dynamic_timbre_boundary_suppress_strength", default=0.0)
        ),
        dynamic_timbre_boundary_radius=max(0, int(_value("dynamic_timbre_boundary_radius", default=2))),
        dynamic_timbre_anchor_preserve_strength=float(
            _value("dynamic_timbre_anchor_preserve_strength", default=0.0)
        ),
        enforce_decoder_no_timing_writeback=bool(
            _value("enforce_decoder_no_timing_writeback", default=True)
        ),
    )


def build_style_mainline_surface_payload(
    controls: StyleMainlineControls,
    *,
    style_trace_available: bool = False,
    dynamic_timbre_available: bool = False,
    style_trace_source: str = "disabled",
    dynamic_timbre_source: str = "disabled",
) -> dict[str, Any]:
    def _surface_value(value):
        if isinstance(value, (int, float)):
            return float(value)
        return "batched"

    return {
        "surface": STYLE_MAINLINE_SURFACE,
        "mainline_owner": controls.mainline_owner,
        "decoder_side_only": True,
        "timing_writeback_allowed": False,
        "enforce_decoder_no_timing_writeback": bool(controls.enforce_decoder_no_timing_writeback),
        "planner_writeback_allowed": False,
        "projector_writeback_allowed": False,
        "timing_authority": "pitch_content_only",
        "mode": controls.mode,
        "apply_global_style_anchor": bool(controls.apply_global_style_anchor),
        "apply_style_trace": bool(controls.apply_style_trace),
        "apply_dynamic_timbre": bool(controls.apply_dynamic_timbre),
        "global_timbre_to_pitch": bool(controls.global_timbre_to_pitch),
        "global_style_anchor_strength": _surface_value(controls.global_style_anchor_strength),
        "style_strength": _surface_value(controls.style_strength),
        "dynamic_timbre_strength": _surface_value(controls.dynamic_timbre_strength),
        "style_trace_mode": str(controls.style_trace_mode),
        "style_memory_mode": str(controls.style_memory_mode),
        "dynamic_timbre_memory_mode": str(controls.dynamic_timbre_memory_mode),
        "style_temperature": float(controls.style_temperature),
        "global_style_trace_blend": float(controls.global_style_trace_blend),
        "dynamic_timbre_temperature": float(controls.dynamic_timbre_temperature),
        "dynamic_timbre_gate_scale": float(controls.dynamic_timbre_gate_scale),
        "dynamic_timbre_gate_bias": float(controls.dynamic_timbre_gate_bias),
        "dynamic_timbre_boundary_suppress_strength": float(
            controls.dynamic_timbre_boundary_suppress_strength
        ),
        "dynamic_timbre_boundary_radius": int(controls.dynamic_timbre_boundary_radius),
        "dynamic_timbre_anchor_preserve_strength": float(
            controls.dynamic_timbre_anchor_preserve_strength
        ),
        "style_trace_available": bool(style_trace_available),
        "dynamic_timbre_available": bool(dynamic_timbre_available),
        "style_trace_source": str(style_trace_source),
        "dynamic_timbre_source": str(dynamic_timbre_source),
    }


def build_style_mainline_memory_payload(reference_cache: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    cache = dict(reference_cache or {})
    return {
        "surface": STYLE_MAINLINE_MEMORY,
        "mainline_owner": STYLE_MAINLINE_OWNER,
        "global_timbre_anchor_available": cache.get("global_timbre_anchor", cache.get("global_style_anchor")) is not None,
        "global_style_summary_available": cache.get("global_style_summary") is not None,
        "prosody_memory_available": cache.get("prosody_memory") is not None,
        "prosody_memory_slow_available": cache.get("prosody_memory_slow") is not None,
        "timbre_memory_available": cache.get("timbre_memory") is not None,
        "timbre_memory_slow_available": cache.get("timbre_memory_slow") is not None,
        "summary_source": cache.get("summary_source", None),
        "global_style_summary_source": cache.get("global_style_summary_source", None),
    }


__all__ = [
    "STYLE_MAINLINE_MEMORY",
    "STYLE_MAINLINE_OWNER",
    "STYLE_MAINLINE_SURFACE",
    "StyleMainlineControls",
    "VALID_DECODER_STYLE_CONDITION_MODES",
    "VALID_STYLE_TRACE_MODES",
    "build_style_mainline_memory_payload",
    "build_style_mainline_surface_payload",
    "normalize_decoder_style_condition_mode",
    "normalize_style_trace_mode",
    "resolve_style_mainline_controls",
]
