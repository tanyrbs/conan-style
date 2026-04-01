"""Legacy compatibility wrapper for the current single-reference inference engine."""

import warnings

from inference.Conan import StreamingVoiceConversion as _StreamingVoiceConversion

__all__ = ["StreamingVoiceConversion"]


class StreamingVoiceConversion(_StreamingVoiceConversion):
    """Backwards-compatible alias for older scripts.

    This module intentionally keeps the old import path working, but the
    canonical single-reference inference entrypoint is now ``inference.Conan``.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "inference.Conan_previous is legacy; use inference.Conan.StreamingVoiceConversion instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
