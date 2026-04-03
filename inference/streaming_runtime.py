import math
from typing import Dict, Mapping, Optional, Tuple

import torch


STREAMING_VOCODER_LEFT_CONTEXT_KEYS = (
    "vocoder_left_context_frames",
    "streaming_vocoder_left_context_frames",
    "vocoder_stream_context",
)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def resolve_vocoder_left_context_frames(source: Mapping) -> Tuple[int, str]:
    for key in STREAMING_VOCODER_LEFT_CONTEXT_KEYS:
        value = source.get(key, None)
        if value is None:
            continue
        try:
            return max(0, int(value)), key
        except (TypeError, ValueError):
            continue
    return 48, "default"


def resolve_streaming_layout(source: Mapping) -> Dict:
    sample_rate = max(1, _safe_int(source.get("audio_sample_rate", 16000), 16000))
    hop_size = max(1, _safe_int(source.get("hop_size", 320), 320))
    frame_ms = 1000.0 * hop_size / float(sample_rate)
    configured_chunk_ms = max(1, _safe_int(source.get("chunk_size", 20), 20))
    chunk_frames = max(1, int(configured_chunk_ms / frame_ms))
    right_context_frames = max(0, _safe_int(source.get("right_context", 0), 0))
    vocoder_left_context_frames, vocoder_left_context_source = resolve_vocoder_left_context_frames(source)
    effective_chunk_ms = chunk_frames * frame_ms
    right_context_ms = right_context_frames * frame_ms
    steady_state_vocoder_window_frames = vocoder_left_context_frames + chunk_frames
    return {
        "audio_sample_rate": int(sample_rate),
        "hop_size": int(hop_size),
        "mel_frame_ms": float(frame_ms),
        "chunk_size_ms_config": int(configured_chunk_ms),
        "chunk_frames": int(chunk_frames),
        "chunk_ms_effective": float(effective_chunk_ms),
        "right_context_frames": int(right_context_frames),
        "right_context_ms": float(right_context_ms),
        "first_packet_algorithmic_latency_ms": float(effective_chunk_ms + right_context_ms),
        "vocoder_left_context_frames": int(vocoder_left_context_frames),
        "vocoder_left_context_source": str(vocoder_left_context_source),
        "steady_state_vocoder_window_frames": int(steady_state_vocoder_window_frames),
        "steady_state_vocoder_window_ms": float(steady_state_vocoder_window_frames * frame_ms),
        "steady_state_vocoder_recompute_multiplier": float(
            steady_state_vocoder_window_frames / float(chunk_frames)
        ),
    }


def estimate_streaming_num_chunks(total_frames: int, chunk_frames: int) -> int:
    total_frames = max(0, int(total_frames))
    chunk_frames = max(1, int(chunk_frames))
    if total_frames <= 0:
        return 0
    return int(math.ceil(total_frames / float(chunk_frames)))


def cumulative_prefix_recompute_multiplier(num_chunks: int) -> float:
    num_chunks = max(0, int(num_chunks))
    if num_chunks <= 0:
        return 0.0
    return float((num_chunks + 1) / 2.0)


def build_streaming_latency_report(
    source: Mapping,
    *,
    duration_seconds: Optional[float] = None,
    total_frames: Optional[int] = None,
) -> Dict:
    layout = resolve_streaming_layout(source)
    report = dict(layout)
    if total_frames is None and duration_seconds is not None:
        duration_seconds = max(0.0, float(duration_seconds))
        total_frames = int(
            math.ceil(duration_seconds * layout["audio_sample_rate"] / float(layout["hop_size"]))
        )
    if total_frames is not None:
        total_frames = max(0, int(total_frames))
        num_chunks = estimate_streaming_num_chunks(total_frames, layout["chunk_frames"])
        report.update(
            {
                "estimated_mel_frames": int(total_frames),
                "estimated_duration_seconds": float(
                    total_frames * layout["hop_size"] / float(layout["audio_sample_rate"])
                ),
                "estimated_num_chunks": int(num_chunks),
                "acoustic_prefix_recompute_multiplier": float(
                    cumulative_prefix_recompute_multiplier(num_chunks)
                ),
            }
        )
    return report


class PrefixCodeBuffer:
    def __init__(self, max_length: int):
        self.max_length = max(1, int(max_length))
        self._buffer = None
        self.length = 0

    def _grow(self, required_length: int):
        if self._buffer is None:
            raise RuntimeError("PrefixCodeBuffer cannot grow before initialization.")
        if required_length <= self._buffer.size(0):
            return
        new_capacity = max(required_length, self._buffer.size(0) * 2)
        new_buffer = self._buffer.new_empty((new_capacity, *self._buffer.shape[1:]))
        new_buffer[: self.length].copy_(self._buffer[: self.length])
        self._buffer = new_buffer

    def append(self, chunk: torch.Tensor) -> torch.Tensor:
        if not isinstance(chunk, torch.Tensor):
            raise TypeError("PrefixCodeBuffer.append expects a torch.Tensor.")
        if chunk.dim() == 0:
            chunk = chunk.reshape(1)
        chunk_length = int(chunk.size(0))
        if chunk_length <= 0:
            return self.prefix()
        if self._buffer is None:
            self._buffer = chunk.new_empty((self.max_length, *chunk.shape[1:]))
        required_length = self.length + chunk_length
        if required_length > self._buffer.size(0):
            self._grow(required_length)
        self._buffer[self.length:required_length].copy_(chunk)
        self.length = required_length
        return self.prefix()

    def prefix(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("PrefixCodeBuffer has not received any content yet.")
        return self._buffer[: self.length]


class RollingMelContextBuffer:
    def __init__(self, left_context_frames: int):
        self.left_context_frames = max(0, int(left_context_frames))
        self._tail = None
        self._chunks = []

    def append(self, mel_chunk: torch.Tensor):
        if not isinstance(mel_chunk, torch.Tensor):
            raise TypeError("RollingMelContextBuffer.append expects a torch.Tensor.")
        if mel_chunk.dim() < 1:
            raise ValueError("mel_chunk must have at least one dimension.")
        if mel_chunk.size(0) <= 0:
            empty_context = 0 if self._tail is None else int(self._tail.size(0))
            return self.current_window(), empty_context
        if self._tail is not None and self._tail.size(0) > 0:
            mel_window = torch.cat([self._tail, mel_chunk], dim=0)
            context_frames = int(self._tail.size(0))
        else:
            mel_window = mel_chunk
            context_frames = 0
        self._chunks.append(mel_chunk)
        if self.left_context_frames > 0:
            self._tail = mel_window[-self.left_context_frames:]
        else:
            self._tail = mel_chunk.new_empty((0, *mel_chunk.shape[1:]))
        return mel_window, context_frames

    def current_window(self) -> Optional[torch.Tensor]:
        return self._tail

    def full_sequence(self) -> Optional[torch.Tensor]:
        if len(self._chunks) <= 0:
            return None
        if len(self._chunks) == 1:
            return self._chunks[0]
        return torch.cat(self._chunks, dim=0)
