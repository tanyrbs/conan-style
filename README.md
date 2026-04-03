# Conan canonical mainline

Updated: 2026-04-03

This repository snapshot keeps only the Conan single-reference strong-style mainline.

Review-archive note: this lightweight code snapshot does **not** bundle large binary artifacts.
Before running prep / training / inference, stage the actual processed data, binary data, and Conan / Emformer / vocoder checkpoints at the configured paths.

## Canonical configs

- binarization: `egs/conan_binarize.yaml`
- training: `egs/conan_emformer.yaml`
- inference: `egs/conan_mainline_infer.yaml`

Both are aligned to the same shipped surface:

- `reference_contract_mode: collapsed_reference`
- `style_profile: strong_style`
- `style_strength` is clamped to the shipped mainline range `0.50 .. 1.80`
- `allow_item_style_strength_override: false` keeps the profile authoritative during training batches
- `dynamic_timbre_strength` is intentionally derived from `style_strength` on the canonical path, so the two are not treated as orthogonal free knobs
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

Implementation note for this 2026-04-03 closure pass:

- `lambda_dynamic_timbre_budget` now constrains the **pre-budget** dynamic-timbre residual instead of only observing the already-clipped runtime result
- the budget reference now includes the decoder slow-style residual in addition to the combined style-owner residual
- the decoder runtime bundle now receives the explicit `slow_style_trace`, so the "mid-stage coarse slow-style / late-stage owner-style" story is no longer only theoretical

Important: this mainline is a **bounded single-reference factorization contract**, not a proof of perfect disentanglement. The code explicitly surfaces `factorization_guaranteed: false`; the shipped losses constrain identity drift, timbre budget, pitch residual safety, and late-owner dominance, but they do not mathematically guarantee that every internal branch learns a unique semantic role.

## Data layout

Default repo paths:

- processed: `data/processed/libritts_single`
- binary: `data/binary/libritts_single`

The current LibriTTS single-speaker metadata does not use old VCTK-style split prefixes, so the repo now uses:

- deterministic per-speaker valid/test holdout fallback
- deterministic reference pairing indices inside binary data

That removes a real reproduction risk where `ref_indices` could drift across Python processes.

Current binarizer structure is also intentionally simplified:

- `data_gen.conan_binarizer.VCBinarizer` owns the shared VC item-processing path
- `ConanBinarizer` only adds cached offline `f0`
- `EmformerBinarizer` stays a thin subclass and no longer duplicates Conan item logic
- condition label normalization / id allocation / `*_map.json` + `*_set.json` emission now come from one source
- train/valid/test split selection is deterministic; optional shuffle only applies to the final train split

Mainline config layering is also now explicit:

- `egs/egs_bases/conan/mainline_shared.yaml`
- `egs/egs_bases/conan/mainline_data.yaml`
- `egs/egs_bases/conan/mainline_stats.yaml`
- `egs/egs_bases/conan/mainline_train.yaml`

So:

- `egs/conan_binarize.yaml` only carries the shared/data surface
- `egs/conan_emformer.yaml` is just the canonical training entry assembly
- `egs/conan_mainline_infer.yaml` is just the canonical inference entry assembly

## Audited mainline contract and environment caveat

This repo encodes the intended canonical mainline contract, but universal verification still depends on staged real data,
compatible `torchaudio` / CUDA runtime libraries, and running the prep gate plus smoke commands in your own environment.
Treat `tasks/Conan/mainline_train_prep.py` and a short real-data smoke run as the source of truth for train-readiness.

Key audited properties in the checked-in code:

- training / inference strong-style semantics are aligned
- canonical config uses `lambda_pitch_residual_safe`, not `lambda_dynamic_timbre_boundary`
- mainline `style_strength` requests are clamped and surfaced as requested/effective metadata
- canonical training batches now lock `style_strength` to the resolved `style_profile` unless `allow_item_style_strength_override` is explicitly enabled
- canonical mainline profile surface now only approves `style_strength` as a direct override; other profile-key deviations are treated as research overrides and coerced back unless explicitly allowed
- resolved mainline controls now stay authoritative for TVT prior/runtime flags and pitch-residual scale / semitone / smoothing
- dynamic timbre is not allowed to leak into `style_to_pitch_residual` on the shipped mainline path
- when no external speaker verifier is configured, the identity loss is evaluated in a frozen internal speaker space instead of a trainable one
- decoder runtime now consumes `slow_style_trace` directly instead of keeping it as a log-only byproduct
- runtime dynamic-timbre budgeting now uses both the owner-style residual and the slow-style residual as its reference, while diagnostics/losses observe the pre-budget residual
- fast style / TVT timbre / pitch residual now share one upper-bound curriculum instead of all starting at step 0
- dynamic timbre now follows a consistent residual semantic on both TVT and non-TVT paths
- dynamic timbre boundary suppression no longer collapses to a global mask on dense HuBERT-style unit sequences
- style-to-pitch residual smoothing is post-canvas and mask-aware
- binary indexed datasets produced under NumPy 2.x remain readable in the shipped NumPy 1.x conda environment
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
- `utils/extract_f0_rmvpe.py` now boots cleanly from the repo root, accepts raw RMVPE state-dict checkpoints, and can batch even when metadata lacks explicit duration fields
- task-side `load_ckpt` warm starts are now non-strict by default, so older Conan checkpoints can still be used for compatible fine-tune / smoke runs after mainline refactors

New control diagnostics from this closure pass include:

- `diag_dynamic_timbre_prebudget_norm`
- `diag_dynamic_timbre_postbudget_norm`
- `diag_dynamic_timbre_post_to_pre_budget_ratio`
- `diag_identity_backend_is_external`
- `diag_identity_encoder_frozen_for_loss`

Latest real-data regeneration / training run completed in this workspace:

- source wavs: `G:\streamVC\LibriTTS_local\LibriTTS`
- cached F0: regenerated with `utils/extract_f0_rmvpe.py`
- binary dataset: regenerated with `egs/conan_binarize.yaml`
- prep gate: `MAINLINE_TRAIN_PREP_OK`
- training: `500` real-data CPU steps finished successfully with `exp_name=ConanMainline500Cpu`

## Commands

### 1) Pre-extract F0

`egs/conan_binarize.yaml` uses `data_gen.conan_binarizer.ConanBinarizer`, so offline F0 extraction is required before binarization.

```bash
python utils/extract_f0_rmvpe.py --config egs/conan_binarize.yaml --pe-ckpt <path-to-rmvpe.pt> --batch-size 8 --max-tokens 40000
```

### 2) Rebuild binary data

```bash
$env:N_PROC='1'; python data_gen/tts/runs/binarize.py --config egs/conan_binarize.yaml
```

### 3) Mainline prep gate

```bash
python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml
```

Expected result after the processed/binary datasets are staged correctly:

- `MAINLINE_TRAIN_PREP_OK`

### 4) Real training

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineTrain
```

Exact-step real-data smoke:

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineSmoke --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=100000,num_sanity_val_steps=0,max_updates=1,save_codes=[]"
```

Recommended short audit smoke in a compatible `conda` env after staging real data:

```bash
conda run -n conan python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineAuditSmokeCpu --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=1,num_sanity_val_steps=0,max_updates=2,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[]"
```

`max_updates` now stops exactly at the requested batch budget instead of overshooting by one step.

500-step warm-start smoke from the shipped Conan checkpoint:

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainline500Cpu --hparams "load_ckpt=checkpoints/Conan/model_ckpt_steps_200000.ckpt,ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=1000000,num_sanity_val_steps=0,max_updates=500,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[],dataloader_persistent_workers=False,dataloader_pin_memory=False,tb_log_interval=200"
```

Because `slow_style_trace` now really enters the decoder runtime bundle, old checkpoints should be A/B checked if you are comparing exact forward behavior before vs. after this closure patch. For new training or resumed fine-tuning on the canonical path, this is the intended corrected behavior.

### 5) Mainline inference

```bash
python inference/run_voice_conversion.py --pair_config inference/conan_single_reference_demo.example.json
```

## Core docs

- `docs/canonical_training_mainline_20260401.md`
- `inference/README.md`
- `docs/streaming_low_latency_mainline_note_20260403.md`
