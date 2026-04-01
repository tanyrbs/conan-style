# Conan inference surface

## Canonical mainline

The Conan repo now has one canonical inference config and one canonical checkpoint entrypoint:

- config: `egs/conan_mainline_infer.yaml`
- checkpoint entry: `--exp_name Conan`
- checkpoint overlay: `checkpoints/Conan/config.yaml`

All mainline runners now default to that pair.

## Mainline product contract

The current Conan mainline only exposes this product-facing surface:

- `src_wav`
- `ref_wav`
- optional `style_profile`
- optional `style_strength`

In other words:

> single reference -> stable identity -> stronger style -> bounded material enhancement

## Canonical runnable entrypoints

### 1. Batch single-reference demo

- runner: `inference/run_voice_conversion.py`
- manifest example: `inference/conan_single_reference_demo.example.json`

```bash
python inference/run_voice_conversion.py \
  --pair_config inference/conan_single_reference_demo.example.json
```

### 2. Style profile sweep demo

- runner: `inference/run_style_profile_sweep.py`
- sweep example: `inference/conan_style_profile_sweep.example.json`

```bash
python inference/run_style_profile_sweep.py \
  --sweep_config inference/conan_style_profile_sweep.example.json

python inference/run_style_profile_evaluation.py \
  --sweep_dir infer_out_profiles/conan_mainline_demo

python inference/run_style_profile_report.py \
  --sweep_dir infer_out_profiles/conan_mainline_demo
```

### 3. Conan Gradio demo

- launcher: `inference/run_conan_gradio_demo.py`
- settings: `inference/conan_gradio/gradio_settings.yaml`

```bash
python inference/run_conan_gradio_demo.py
```

## Important policy

### 1. Split references are not part of the mainline surface

`ref_timbre_wav / ref_style_wav / ref_dynamic_timbre_wav` are no longer part of the canonical public surface.

The canonical runners will collapse them back to `ref_wav` unless you explicitly opt into split-reference / research behavior.

### 2. Advanced label controls are gated

`emotion / accent / arousal / valence / energy` are no longer default Conan mainline controls.

The canonical runners will ignore or gate them unless you explicitly opt into a research/ablation flow.

### 3. Factorized reporting is opt-in

`run_style_profile_evaluation.py` will only produce a factorized swap report when explicitly asked:

```bash
python inference/run_style_profile_evaluation.py \
  --sweep_dir <dir> \
  --enable_factorized_report
```

## Streaming status

Current streaming semantics are explicit:

- Emformer: stateful
- acoustic model: prefix recompute
- vocoder: bounded left-context re-synthesis

So the current path is suitable for streaming-oriented evaluation, but it is not yet a fully stateful end-to-end decoder/vocoder stack.

## Legacy / research-only surfaces

These files still exist, but they are not the Conan mainline:

- `Conan_previous.py`
- `run_voice_conversion_nvae.py`
- `factorized_swap_builder.py`
- `factorized_swap_report.py`
- `build_libritts_factorized_swap_matrix.py`
- `tts/gradio/*` (legacy NATSpeech/PortaSpeech TTS surface)
