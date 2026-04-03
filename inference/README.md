# Conan inference mainline

This repo now keeps only the Conan mainline inference surface.

## Canonical files

- config: `egs/conan_mainline_infer.yaml`
- model entry: `checkpoints/Conan/model_ckpt_steps_200000.ckpt`
- Emformer: `checkpoints/Emformer/model_ckpt_steps_700000.ckpt`
- vocoder: `checkpoints/hifigan_vc/model_ckpt_steps_1000000.ckpt`

## Public runtime contract

The retained Conan mainline surface is:

- `src_wav`
- `ref_wav`
- optional `style_profile`
- optional `style_strength`

Mainline semantics stay fixed to:

> single reference -> stable identity -> strong style owner -> bounded dynamic timbre enhancement

Notes:

- `collapsed_reference` is the only shipped contract
- `style_profile: strong_style` is the canonical default surface
- `style_strength` is clamped to the shipped mainline range `0.50 .. 1.80`
- `style_to_pitch_residual_include_timbre: false` is part of the shipped contract
- the shipped path does **not** currently promise a fully wired post-rhythm pitch canvas; `style_to_pitch_residual_mode=post_rhythm` is best-effort and will fall back to source-aligned when no rhythm frame-index canvas is emitted
- training ramps fast style / TVT timbre / pitch residual by the `upper_bound_curriculum`, but inference always runs with the full ceiling
- split-reference / factorized payloads are **not part of the canonical public runner surface**; the default CLI path still rejects them even though some internal/evaluator code paths retain research-oriented handling
- online/offline streaming parity is expected to match in mel length and wav length

## Streaming status on the shipped mainline

The current Conan inference path is **streaming-oriented**, but it is **not** yet a fully native stateful
end-to-end streaming stack:

- Emformer front-end: stateful streaming
- acoustic model: prefix recompute
- vocoder: stateless window recompute

That means the shipped path is suitable for streaming evaluation / instrumentation, while still preserving
the original acoustic + vocoder behavior.

### Theoretical latency report

You can inspect the current theoretical latency and recompute budget without loading checkpoints:

```bash
python inference/run_streaming_latency_report.py
python inference/run_streaming_latency_report.py --duration_seconds 5
```

The report surfaces:

- mel-frame duration
- chunk / right-context timing
- first-packet algorithmic latency
- steady-state vocoder window + recompute multiplier
- cumulative acoustic prefix-recompute multiplier for an estimated duration

### Runtime metadata

`inference/Conan.py` now writes the following streaming metadata into each inference record:

- `streaming_capabilities`
- `streaming_latency_report`
- `theoretical_first_packet_latency_ms`
- `steady_state_vocoder_window_ms`
- `steady_state_vocoder_recompute_multiplier`
- `acoustic_prefix_recompute_multiplier`

## Verified inference fixes as of 2026-04-03

- style-profile defaults now flow cleanly into runtime controls
- each inference request now resolves its style profile exactly once before building control/runtime kwargs
- `inference/conan_request.py` is now the canonical source for public request keys, advanced-control filtering, and split-reference detection
- request-helper advanced mode now forwards the runtime controls that the engine actually consumes, including explicit `dynamic_timbre_strength` and mainline research-override gating; internal budget stabilizers stay hparam-owned on the closed mainline path
- request helpers now also flag unsupported internal-only keys such as `style_condition_strength`, protected budget stabilizers, and upper-bound curriculum knobs instead of silently pretending they are valid request-surface controls
- canonical runners stay single-reference by default; internal split-reference handling remains for evaluator/research plumbing, not as a public closed-mainline promise
- resolved mainline controls stay authoritative for TVT prior/runtime flags and pitch-residual scale / semitone / smoothing
- inference metadata now reports requested/effective/clamped `style_strength`
- inference metadata now also reports requested pitch-canvas mode, realized canvas, and fallback reason when post-rhythm routing is unavailable
- runtime layout validation now checks for actual checkpoint artifacts, not just directories
- streaming prefix inference trims missing tail right-context correctly
- online/offline parity checks now match on both mel length and wav sample length
- direct library-style inference (`set_hparams(..., global_hparams=False)` + `StreamingVoiceConversion(...)`) no longer requires the global singleton hparams dict on the canonical style/prosody/timbre path, although some legacy fallback reads still remain outside that main path
- `EmformerDistillModel` respects `emformer_mode` as the canonical config key
- vocoder wrappers now expose an explicit streaming capability surface instead of silently implying native streaming support

## Runnable entrypoints

### Single-reference conversion

Edit `inference/conan_single_reference_demo.example.json`, then run:

```bash
python inference/run_voice_conversion.py   --pair_config inference/conan_single_reference_demo.example.json
```

### Style-profile sweep

Edit `inference/conan_style_profile_sweep.example.json`, then run:

```bash
python inference/run_style_profile_sweep.py   --sweep_config inference/conan_style_profile_sweep.example.json

python inference/run_style_profile_evaluation.py   --sweep_dir infer_out_profiles/conan_mainline_demo

python inference/run_style_profile_report.py   --sweep_dir infer_out_profiles/conan_mainline_demo
```

## Related training doc

- `docs/canonical_training_mainline_20260401.md`
