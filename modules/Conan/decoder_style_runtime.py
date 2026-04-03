from __future__ import annotations

from typing import Any, Mapping, Optional

from modules.Conan.decoder_style_bundle import (
    DECODER_STYLE_TIMING_AUTHORITY,
    build_decoder_style_bundle,
    validate_decoder_style_bundle,
)
from modules.Conan.effective_signal import maybe_effective_sequence, maybe_effective_singleton


def build_runtime_decoder_style_bundle(
    *,
    decoder_style_adapter,
    reference_cache: Optional[Mapping[str, Any]],
    reference_contract: Optional[Mapping[str, Any]],
    style_mainline,
    global_timbre_anchor=None,
    global_timbre_anchor_runtime=None,
    global_style_summary_runtime=None,
    global_style_summary_runtime_source: str = "reference_cache",
    M_style=None,
    M_style_mask=None,
    M_timbre=None,
    M_timbre_mask=None,
    M_timbre_source: str = "missing",
):
    decoder_signal_eps = float(
        getattr(decoder_style_adapter, "effective_signal_epsilon", 1e-8)
        if decoder_style_adapter is not None
        else 1e-8
    )
    decoder_global_timbre_anchor = (
        global_timbre_anchor if style_mainline.apply_global_style_anchor else None
    )
    decoder_global_timbre_anchor_runtime = (
        global_timbre_anchor_runtime if style_mainline.apply_global_style_anchor else None
    )
    bundle = build_decoder_style_bundle(
        global_timbre_anchor=maybe_effective_singleton(
            decoder_global_timbre_anchor,
            eps=decoder_signal_eps,
        ),
        global_timbre_anchor_runtime=maybe_effective_singleton(
            decoder_global_timbre_anchor_runtime,
            eps=decoder_signal_eps,
        ),
        global_timbre_source=(
            reference_cache.get("global_timbre_anchor_source", "reference_cache")
            if isinstance(reference_cache, Mapping)
            else "reference_cache"
        ),
        global_style_summary=maybe_effective_singleton(
            global_style_summary_runtime,
            eps=decoder_signal_eps,
        ),
        global_style_summary_source=global_style_summary_runtime_source,
        slow_style_trace=None,
        slow_style_trace_mask=None,
        slow_style_source="retained_in_logs_only",
        M_style=maybe_effective_sequence(M_style, eps=decoder_signal_eps),
        M_style_mask=M_style_mask,
        M_style_source="combined_style_owner",
        M_timbre=maybe_effective_sequence(M_timbre, eps=decoder_signal_eps),
        M_timbre_mask=M_timbre_mask,
        M_timbre_source=M_timbre_source,
        factorization_guaranteed=bool(
            (reference_contract or {}).get("factorization_guaranteed", False)
        ),
        mainline_owner=style_mainline.mainline_owner,
        bundle_variant=style_mainline.mode,
        bundle_source="decoder_style_mainline",
        timing_authority=DECODER_STYLE_TIMING_AUTHORITY,
        enforce_no_timing_writeback=style_mainline.enforce_decoder_no_timing_writeback,
        effective_signal_epsilon=decoder_signal_eps,
    )
    validate_decoder_style_bundle(bundle)
    return bundle, decoder_signal_eps


__all__ = ["build_runtime_decoder_style_bundle"]
