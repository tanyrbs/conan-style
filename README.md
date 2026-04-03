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

Training now opens the mainline upper-bound modules by curriculum:

- fast style branch
- TVT dynamic timbre path
- bounded style-to-pitch residual

Canonical training ramps them from `20000 -> 80000`; inference always uses the full ceiling.

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
- resolved mainline controls now stay authoritative for TVT prior/runtime flags and pitch-residual scale / semitone / smoothing
- dynamic timbre is not allowed to leak into `style_to_pitch_residual` on the shipped mainline path
- fast style / TVT timbre / pitch residual now share one upper-bound curriculum instead of all starting at step 0
- dynamic timbre now follows a consistent residual semantic on both TVT and non-TVT paths
- dynamic timbre boundary suppression no longer collapses to a global mask on dense HuBERT-style unit sequences
- style-to-pitch residual smoothing is post-canvas and mask-aware
- internal runtime glue is now split by responsibility:
  - `modules/Conan/common_utils.py` owns shared lightweight helpers
  - `modules/Conan/pitch_canvas_utils.py` owns pitch-canvas projection / mask / smoothing semantics
  - `modules/Conan/decoder_style_runtime.py` owns decoder style bundle assembly
- Conan model assembly is now cleaner at layer 2:
  - `modules/Conan/pitch_runtime.py` owns pitch-generation and style-to-pitch runtime mixins
  - `modules/Conan/common.py` is the single source for lightweight mapping lookup helpers
  - `inference/conan_request.py` is the single source for canonical mainline request schema helpers
- that maintainability layer does not add new public controls; it only reduces drift and duplicated glue logic
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

Exact-step real-data smoke:

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineSmoke --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=100000,num_sanity_val_steps=0,max_updates=1,save_codes=[]"
```

`max_updates` now stops exactly at the requested batch budget instead of overshooting by one step.

### 4) Mainline inference

```bash
python inference/run_voice_conversion.py --pair_config inference/conan_single_reference_demo.example.json
```

## Core docs

- `docs/canonical_training_mainline_20260401.md`
- `inference/README.md`
