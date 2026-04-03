# Conan streaming / low-latency mainline note

Updated: 2026-04-03

## Scope

This refactor does **not** switch the shipped Conan mainline to a true full-chain native stateful streaming stack.

That was intentional.

The current repo already has a stateful streaming front-end through Emformer, but the downstream acoustic decoder
and shipped vocoder wrappers still run in prefix/window recompute form. Forcing a default transition to native
incremental acoustic decoding or a different streaming vocoder would change runtime behavior without a matching
quality-regression loop.

So this round only does the safe part:

- consolidate streaming runtime math / helper glue
- make capability boundaries explicit
- reduce Python-side prefix concatenation overhead
- add latency instrumentation
- keep default acoustic / vocoder output semantics unchanged

## Current shipped streaming shape

Mainline status:

- front-end encoder: **stateful streaming**
- acoustic model: **prefix recompute**
- vocoder: **stateless window recompute**

That means the shipped path is best described as:

> streaming-oriented evaluation path, not fully native end-to-end stateful deployment

## New code surface

### 1) `inference/streaming_runtime.py`

This is now the single home for:

- `resolve_vocoder_left_context_frames(...)`
- `resolve_streaming_layout(...)`
- `build_streaming_latency_report(...)`
- `cumulative_prefix_recompute_multiplier(...)`
- `PrefixCodeBuffer`
- `RollingMelContextBuffer`

This keeps chunk / context / latency formulas out of `inference/Conan.py`.

### 2) explicit vocoder capability surface

The shipped vocoder wrappers now expose:

- `supports_native_streaming()`
- `reset_stream()`
- `spec2wav_stream()`

Current shipped wrappers explicitly report **no native streaming support**.

That avoids semantic drift where a wrapper looks “streaming-like” but still actually does full-window recompute.

### 3) metadata instrumentation

Each inference request now surfaces:

- `streaming_capabilities`
- `streaming_latency_report`
- `theoretical_first_packet_latency_ms`
- `steady_state_vocoder_window_ms`
- `steady_state_vocoder_recompute_multiplier`
- actual `acoustic_prefix_recompute_multiplier`

## Current theoretical latency

Using the shipped Conan mainline config:

- `audio_sample_rate = 16000`
- `hop_size = 320`
- one mel frame = `20 ms`
- `chunk_size = 80` -> `4` mel frames
- `right_context = 2` -> `40 ms`
- `vocoder_left_context_frames = 48`

So:

- first-packet algorithmic latency = `80 + 40 = 120 ms`
- steady-state vocoder window = `48 + 4 = 52` frames
- steady-state vocoder window time = `52 * 20 = 1040 ms`
- steady-state vocoder recompute multiplier = `52 / 4 = 13x`

For an estimated 5-second utterance:

- mel frames ≈ `250`
- chunks ≈ `ceil(250 / 4) = 63`
- cumulative acoustic prefix-recompute multiplier ≈ `(63 + 1) / 2 = 32x`

## CLI

You can inspect the current theoretical budget without loading checkpoints:

```bash
python inference/run_streaming_latency_report.py
python inference/run_streaming_latency_report.py --duration_seconds 5
```

## Why the default path still stays conservative

Not changed in this round:

- no native incremental Conan decoder path by default
- no default switch to a streaming vocoder implementation
- no default vocoder replacement

Those are the right next-stage optimizations only after:

- real quality regression
- long-form parity checks
- latency / quality trade-off measurement on the target deployment surface
