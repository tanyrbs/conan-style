# Conan canonical training mainline

Updated: 2026-04-03

## 1. Scope

This cleaned repo snapshot keeps only the Conan single-reference strong-style mainline.

Mainline target:

> single reference -> stable identity -> strong style owner -> bounded dynamic timbre enhancement -> decoder-side fusion

## 2. Canonical configs

Only these two configs remain:

- training: `egs/conan_emformer.yaml`
- inference: `egs/conan_mainline_infer.yaml`

Both are aligned to the same mainline semantics, including `style_profile: strong_style`.

## 3. Mainline contract

Training / inference stay locked to:

- `reference_contract_mode: collapsed_reference`
- `decoder_style_condition_mode: mainline_full`
- `style_trace_mode: dual`
- `style_router_enabled: true`
- `style_to_pitch_residual: true`
- `global_timbre_to_pitch: false`
- `allow_split_reference_inputs: false`

Canonical control regularization stays on the 4-loss pack:

- `lambda_output_identity_cosine`
- `lambda_dynamic_timbre_budget`
- `lambda_pitch_residual_safe`
- `lambda_decoder_late_owner`

## 4. Data prerequisites

Default repo paths:

- `data/binary/libritts_single`
- `data/processed/libritts_single`

Current checked-in LibriTTS-single behavior:

- `valid_prefixes` / `test_prefixes` are intentionally empty
- binarization falls back to deterministic per-speaker utterance holdout
- binary `*_ref_indices.npy` use a stable hash, so reference pairing stays reproducible across Python processes

## 5. Training-prep gate

Before real training, run:

```bash
python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml
```

Expected result:

- `MAINLINE_TRAIN_PREP_OK`

This prep gate now also checks that style-profile defaults really control mainline runtime strengths instead of silently falling back to neutral values.

It also checks that binary train / valid / test splits exist and are non-empty.

## 6. Real training command

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineTrain
```

Notes:

- keep shipped inference checkpoint entry `Conan` untouched
- use a fresh experiment name for real training output

## 7. Mainline inference / evaluation after training

Single-reference demo:

```bash
python inference/run_voice_conversion.py   --pair_config inference/conan_single_reference_demo.example.json
```

Style-profile sweep:

```bash
python inference/run_style_profile_sweep.py   --sweep_config inference/conan_style_profile_sweep.example.json

python inference/run_style_profile_evaluation.py   --sweep_dir infer_out_profiles/conan_mainline_demo

python inference/run_style_profile_report.py   --sweep_dir infer_out_profiles/conan_mainline_demo
```

## 8. Verified implementation notes as of 2026-04-03

- canonical mainline uses `lambda_pitch_residual_safe`; `lambda_dynamic_timbre_boundary` remains `0.0`
- `style_to_pitch_residual` smoothing is applied after projection onto the final pitch canvas and is mask-aware
- streaming inference tail trimming has been aligned with offline decoding, so online/offline mel and wav lengths now match on parity checks
- `EmformerDistillModel` now reads `emformer_mode` correctly instead of only the legacy `mode` key
- Emformer distillation train / infer / validation paths now align targets to the actual streamed-logit length

## 9. Repo cleanup policy in this snapshot

Removed from the repo snapshot:

- old / research inference surfaces
- Gradio demo surface
- smoke, dry-run, and parity scripts
- redundant YAML configs
- non-core historical docs
- stale generated outputs and cache directories

Retained on purpose:

- Conan mainline training path
- Conan mainline inference path
- canonical prep gate
- latest Conan checkpoint plus required Emformer / vocoder checkpoints
