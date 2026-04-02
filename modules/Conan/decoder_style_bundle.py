from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import torch

from modules.Conan.effective_signal import tensor_has_effective_signal


DECODER_STYLE_TIMING_AUTHORITY = "decoder_only_no_timing_writeback"

DECODER_STYLE_BUNDLE_FORBIDDEN_TIMING_KEYS = (
    "mel2ph",
    "dur",
    "pause_after",
    "unit_dur_exec",
    "frame2unit",
    "execution_authority",
    "unit_clock_tokens",
)

VALID_DECODER_STYLE_BUNDLE_VARIANTS = (
    "legacy_full",
    "mainline_full",
    "global_style_dynamic_timbre",
    "global_only",
    "dynamic_timbre_only",
)


def _is_sequence_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() == 3 and value.size(1) > 0


def _normalize_single(value: Any):
    if not isinstance(value, torch.Tensor):
        return value
    if value.dim() == 2:
        return value.unsqueeze(1)
    if value.dim() == 3 and value.size(-1) == 1:
        return value.transpose(1, 2)
    return value


def _coerce_mask(mask: Any, reference: Optional[torch.Tensor]):
    if not isinstance(mask, torch.Tensor) or not _is_sequence_tensor(reference):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 2 or tuple(mask.shape[:2]) != tuple(reference.shape[:2]):
        return None
    return mask.bool().to(device=reference.device)


def normalize_decoder_style_bundle_variant(variant: Any, default: str = "mainline_full") -> str:
    normalized_default = str(default or "mainline_full").strip().lower() or "mainline_full"
    normalized = str(variant or normalized_default).strip().lower() or normalized_default
    alias_map = {
        "default": "mainline_full",
        "legacy": "legacy_full",
        "full": "mainline_full",
        "mainline": "mainline_full",
        "gsdt": "mainline_full",
        "global_dynamic_timbre": "mainline_full",
        "global_only_anchor": "global_only",
        "dynamic_only": "dynamic_timbre_only",
        "timbre_only": "dynamic_timbre_only",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in VALID_DECODER_STYLE_BUNDLE_VARIANTS:
        return normalized_default
    return normalized


@dataclass(frozen=True)
class DecoderStyleBundle:
    bundle_variant: str = "mainline_full"
    bundle_source: str = "decoder_style_mainline"
    mainline_owner: str = "M_style_owner_plus_bounded_M_timbre"
    timing_authority: str = DECODER_STYLE_TIMING_AUTHORITY
    decoder_only: bool = True
    planner_writeback_allowed: bool = False
    projector_writeback_allowed: bool = False
    timing_writeback_allowed: bool = False
    enforce_no_timing_writeback: bool = True
    factorization_guaranteed: bool = False
    effective_signal_epsilon: float = 1e-8
    global_timbre_anchor: Optional[torch.Tensor] = None
    global_timbre_anchor_runtime: Optional[torch.Tensor] = None
    global_timbre_source: str = "none"
    global_timbre_role: str = "global_timbre_anchor"
    global_style_summary: Optional[torch.Tensor] = None
    global_style_summary_source: str = "none"
    slow_style_trace: Optional[torch.Tensor] = None
    slow_style_trace_mask: Optional[torch.Tensor] = None
    slow_style_source: str = "none"
    M_style: Optional[torch.Tensor] = None
    M_style_mask: Optional[torch.Tensor] = None
    M_style_source: str = "none"
    M_timbre: Optional[torch.Tensor] = None
    M_timbre_mask: Optional[torch.Tensor] = None
    M_timbre_source: str = "none"

    def as_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


def ensure_decoder_style_bundle_respects_timing(
    bundle: Optional[Mapping[str, Any]],
    *,
    forbidden_keys=DECODER_STYLE_BUNDLE_FORBIDDEN_TIMING_KEYS,
):
    if not isinstance(bundle, Mapping):
        return
    if bool(bundle.get("timing_writeback_allowed", False)):
        return
    if not bool(bundle.get("enforce_no_timing_writeback", True)):
        return
    for forbidden in forbidden_keys:
        if bundle.get(forbidden) is not None:
            raise RuntimeError(
                f"Decoder style bundle must not write timing controls. Forbidden key detected: {forbidden}."
            )


def canonicalize_decoder_style_bundle(bundle: Optional[Mapping[str, Any]] = None):
    if not isinstance(bundle, Mapping):
        return None
    normalized = dict(bundle)
    normalized["bundle_variant"] = normalize_decoder_style_bundle_variant(
        normalized.get("bundle_variant", normalized.get("decoder_style_condition_mode", "mainline_full"))
    )
    normalized["bundle_source"] = str(normalized.get("bundle_source", "decoder_style_mainline"))
    normalized["mainline_owner"] = str(normalized.get("mainline_owner", "M_style_owner_plus_bounded_M_timbre"))
    normalized["timing_authority"] = str(
        normalized.get("timing_authority", DECODER_STYLE_TIMING_AUTHORITY)
    )
    normalized["decoder_only"] = bool(normalized.get("decoder_only", True))
    normalized["planner_writeback_allowed"] = bool(normalized.get("planner_writeback_allowed", False))
    normalized["projector_writeback_allowed"] = bool(normalized.get("projector_writeback_allowed", False))
    normalized["timing_writeback_allowed"] = bool(normalized.get("timing_writeback_allowed", False))
    normalized["enforce_no_timing_writeback"] = bool(normalized.get("enforce_no_timing_writeback", True))
    normalized["factorization_guaranteed"] = bool(normalized.get("factorization_guaranteed", False))
    normalized["effective_signal_epsilon"] = float(normalized.get("effective_signal_epsilon", 1e-8))
    effective_signal_epsilon = max(0.0, float(normalized["effective_signal_epsilon"]))

    for key in ("global_timbre_anchor", "global_timbre_anchor_runtime", "global_style_summary"):
        value = _normalize_single(normalized.get(key))
        if _tensor_has_effective_signal(value, eps=effective_signal_epsilon):
            normalized[key] = value
        else:
            normalized[key] = None
    if normalized.get("global_timbre_anchor_runtime") is None:
        normalized["global_timbre_anchor_runtime"] = normalized.get("global_timbre_anchor")
    normalized["global_timbre"] = normalized.get("global_timbre_anchor_runtime")

    branch_specs = (
        ("slow_style_trace", "slow_style_trace_mask"),
        ("M_style", "M_style_mask"),
        ("M_timbre", "M_timbre_mask"),
    )
    for key, mask_key in branch_specs:
        value = normalized.get(key)
        if _is_sequence_tensor(value) and _tensor_has_effective_signal(value, eps=effective_signal_epsilon):
            normalized[mask_key] = _coerce_mask(normalized.get(mask_key), value)
        else:
            normalized[key] = None
            normalized[mask_key] = None

    normalized["style_trace"] = normalized.get("M_style")
    normalized["style_trace_mask"] = normalized.get("M_style_mask")
    normalized["dynamic_timbre"] = normalized.get("M_timbre")
    normalized["dynamic_timbre_mask"] = normalized.get("M_timbre_mask")

    ensure_decoder_style_bundle_respects_timing(normalized)
    return normalized


def validate_decoder_style_bundle(bundle: Optional[Mapping[str, Any]] = None):
    normalized = canonicalize_decoder_style_bundle(bundle)
    if normalized is None:
        return
    if bool(normalized.get("planner_writeback_allowed", False)):
        raise ValueError("decoder_style_bundle must keep `planner_writeback_allowed=False`.")
    if bool(normalized.get("projector_writeback_allowed", False)):
        raise ValueError("decoder_style_bundle must keep `projector_writeback_allowed=False`.")
    for key in ("global_timbre_anchor", "global_timbre_anchor_runtime", "global_style_summary"):
        value = normalized.get(key)
        if value is not None and (not isinstance(value, torch.Tensor) or value.dim() != 3):
            raise ValueError(f"decoder_style_bundle['{key}'] must be a [B, 1, H] tensor.")
    for key, mask_key in (
        ("slow_style_trace", "slow_style_trace_mask"),
        ("M_style", "M_style_mask"),
        ("M_timbre", "M_timbre_mask"),
    ):
        value = normalized.get(key)
        if value is None:
            continue
        if not _is_sequence_tensor(value):
            raise ValueError(f"decoder_style_bundle['{key}'] must be a [B, T, H] tensor.")
        mask = normalized.get(mask_key)
        if mask is not None and tuple(mask.shape[:2]) != tuple(value.shape[:2]):
            raise ValueError(
                f"decoder_style_bundle['{mask_key}'] shape mismatch for '{key}': "
                f"value={tuple(value.shape)}, mask={tuple(mask.shape)}"
            )


def build_decoder_style_bundle(
    *,
    global_timbre_anchor=None,
    global_timbre_anchor_runtime=None,
    global_timbre_source: str = "none",
    global_timbre_role: str = "global_timbre_anchor",
    global_style_summary=None,
    global_style_summary_source: str = "none",
    slow_style_trace=None,
    slow_style_trace_mask=None,
    slow_style_source: str = "none",
    M_style=None,
    M_style_mask=None,
    M_style_source: str = "none",
    M_timbre=None,
    M_timbre_mask=None,
    M_timbre_source: str = "none",
    bundle_variant: str = "mainline_full",
    bundle_source: str = "decoder_style_mainline",
    mainline_owner: str = "M_style_owner_plus_bounded_M_timbre",
    timing_authority: str = DECODER_STYLE_TIMING_AUTHORITY,
    decoder_only: bool = True,
    planner_writeback_allowed: bool = False,
    projector_writeback_allowed: bool = False,
    timing_writeback_allowed: bool = False,
    enforce_no_timing_writeback: bool = True,
    factorization_guaranteed: bool = False,
    effective_signal_epsilon: float = 1e-8,
):
    bundle = DecoderStyleBundle(
        bundle_variant=normalize_decoder_style_bundle_variant(bundle_variant),
        bundle_source=str(bundle_source),
        mainline_owner=str(mainline_owner),
        timing_authority=str(timing_authority),
        decoder_only=bool(decoder_only),
        planner_writeback_allowed=bool(planner_writeback_allowed),
        projector_writeback_allowed=bool(projector_writeback_allowed),
        timing_writeback_allowed=bool(timing_writeback_allowed),
        enforce_no_timing_writeback=bool(enforce_no_timing_writeback),
        factorization_guaranteed=bool(factorization_guaranteed),
        effective_signal_epsilon=float(effective_signal_epsilon),
        global_timbre_anchor=global_timbre_anchor,
        global_timbre_anchor_runtime=global_timbre_anchor_runtime,
        global_timbre_source=str(global_timbre_source),
        global_timbre_role=str(global_timbre_role),
        global_style_summary=global_style_summary,
        global_style_summary_source=str(global_style_summary_source),
        slow_style_trace=slow_style_trace,
        slow_style_trace_mask=slow_style_trace_mask,
        slow_style_source=str(slow_style_source),
        M_style=M_style,
        M_style_mask=M_style_mask,
        M_style_source=str(M_style_source),
        M_timbre=M_timbre,
        M_timbre_mask=M_timbre_mask,
        M_timbre_source=str(M_timbre_source),
    )
    normalized = canonicalize_decoder_style_bundle(bundle.as_dict())
    validate_decoder_style_bundle(normalized)
    return normalized


__all__ = [
    "DECODER_STYLE_BUNDLE_FORBIDDEN_TIMING_KEYS",
    "DECODER_STYLE_TIMING_AUTHORITY",
    "DecoderStyleBundle",
    "build_decoder_style_bundle",
    "canonicalize_decoder_style_bundle",
    "ensure_decoder_style_bundle_respects_timing",
    "normalize_decoder_style_bundle_variant",
    "validate_decoder_style_bundle",
]
