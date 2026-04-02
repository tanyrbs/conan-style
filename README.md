# Conan canonical mainline

Updated: 2026-04-03

This repository snapshot keeps only the Conan single-reference strong-style mainline.

## Canonical configs

- training: `egs/conan_emformer.yaml`
- inference: `egs/conan_mainline_infer.yaml`

Both are aligned to the same shipped surface:

- `reference_contract_mode: collapsed_reference`
- `style_profile: strong_style`
- `style_strength` is clamped to the shipped mainline range `0.50 .. 1.80`
- `style_to_pitch_residual: true`
- `style_to_pitch_residual_include_timbre: false`
- `global_timbre_to_pitch: false`

## Canonical 4-loss pack

The canonical training mainline keeps exactly these active control losses:

- `lambda_output_identity_cosine`
- `lambda_dynamic_timbre_budget`
- `lambda_pitch_residual_safe`
- `lambda_decoder_late_owner`

`lambda_dynamic_timbre_boundary` stays off in the canonical mainline.

## Data layout

Default repo paths:

- processed: `data/processed/libritts_single`
- binary: `data/binary/libritts_single`

The current LibriTTS single-speaker metadata does not use old VCTK-style split prefixes, so the repo now uses:

- deterministic per-speaker valid/test holdout fallback
- deterministic reference pairing indices inside binary data

That removes a real reproduction risk where `ref_indices` could drift across Python processes.

## Verified current state

As of 2026-04-03, this repo snapshot has been checked with:

- real-data binarization
- prep gate
- real-data smoke training
- resume-from-checkpoint smoke
- online/offline inference parity

Key validated properties:

- training / inference strong-style semantics are aligned
- canonical config uses `lambda_pitch_residual_safe`, not `lambda_dynamic_timbre_boundary`
- mainline `style_strength` requests are now clamped and surfaced as requested/effective metadata
- dynamic timbre is not allowed to leak into `style_to_pitch_residual` on the shipped mainline path
- style-to-pitch residual smoothing is post-canvas and mask-aware
- streaming inference tail trimming now matches offline length exactly

## Commands

### 1) Rebuild binary data

```bash
$env:N_PROC='1'; python data_gen/tts/runs/binarize.py --config egs/conan_emformer.yaml
```

### 2) Mainline prep gate

```bash
python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml
```

Expected result:

- `MAINLINE_TRAIN_PREP_OK`

### 3) Real training

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineTrain
```

### 4) Mainline inference

```bash
python inference/run_voice_conversion.py --pair_config inference/conan_single_reference_demo.example.json
```

## Core docs

- `docs/canonical_training_mainline_20260401.md`
- `inference/README.md`
