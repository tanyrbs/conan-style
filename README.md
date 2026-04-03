# Conan-derived canonical mainline

Updated: 2026-04-04

**Repository note:** this repo is our Conan-based modified implementation, not an official Conan upstream repository or the authoritative "official implementation". This README only describes the shipped contract in this codebase.

This repository snapshot keeps only our Conan-based single-reference strong-style mainline adaptation. Here, "canonical" means the supported mainline path in this repo, not an upstream-official release label.

Review-archive note: do **not** infer train-readiness from repository text alone.
Some checkouts may already have staged `data/*` and `checkpoints/*` artifacts at the canonical paths, while others may not.
Treat the prep gate plus short smoke runs as the source of truth for whether the current local checkout is actually ready.

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
- post-rhythm pitch-canvas projection remains best-effort; when no rhythm frame-index canvas is emitted, shipped runtime falls back to source-aligned residual application and surfaces the realized canvas in metadata

Training now opens the mainline upper-bound modules by curriculum:

- fast style branch
- TVT dynamic timbre path
- bounded style-to-pitch residual

Canonical training ramps them from `20000 -> 80000`; inference always uses the full ceiling.

## Canonical scheduled control-loss surface

The canonical training mainline keeps these five nonzero-configured control losses on the shipped path.
Their effective weights are still schedule-controlled during early training, so do not read this as
"all five fire from step 0 with full strength":

- `lambda_output_identity_cosine`
- `lambda_dynamic_timbre_budget`
- `lambda_pitch_residual_safe`
- `lambda_decoder_late_owner`
- `lambda_style_success_rank`

`lambda_dynamic_timbre_boundary` stays off in the canonical mainline.

Implementation note for this 2026-04-03 closure pass:

- `lambda_dynamic_timbre_budget` now constrains the **pre-budget** dynamic-timbre residual instead of only observing the already-clipped runtime result
- the budget reference now uses **owner-style energy + slow-style excess energy**, avoiding a direct slow-style double count when the routed owner is already carrying that authority
- the budget loss now supervises both the **pre-budget residual** and the **realized decoder-stage deltas**, so it constrains the actual effect space instead of only the pre-projection residual space
- `lambda_pitch_residual_safe` now uses robust target alignment (`SmoothL1`), adds a one-sided residual budget against the bounded target magnitude, and matches **target slope** when that bounded target exists; the missing-target branch stays conservative (`zero-anchor + raw smoothness`)
- the decoder runtime bundle now receives the explicit `slow_style_trace`, so the "mid-stage coarse slow-style / late-stage owner-style" story is no longer only theoretical

Important: this mainline is a **bounded single-reference factorization contract**, not a proof of perfect disentanglement. The code explicitly surfaces `factorization_guaranteed: false`; the shipped losses constrain identity drift, timbre budget, pitch residual safety, and late-owner dominance, but they do not mathematically guarantee that every internal branch learns a unique semantic role.

`lambda_style_success_rank` is intentionally training-side only. It does not open a new inference control surface; instead it adds a lower-bound style-success signal that combines paired self/reference alignment with weak-label batch ranking when metadata negatives are available. The target side is kept reference-derived only: self-derived runtime summaries such as `style_trace_pooled` / `style_trace_blended_with_reference` are excluded from that lower-bound target so the loss does not quietly grade the owner-style branch against its own pooled output. `tasks/Conan/mainline_train_prep.py` now reports whether the staged artifacts imply `paired_plus_weak_label_ranking` or only `paired_only`; if the staged condition artifacts expose no usable label buckets (`num_labels <= 1`), the label-driven weak-ranking branch naturally degenerates toward paired alignment rather than meaningful cross-label ranking. In the currently staged LibriTTS-single artifacts in this checkout, those weak-label buckets are empty, so the prep summary resolves to `paired_only`. On top of that, the shipped rank loss now has a conservative **proxy-negative fallback** for label-sparse batches: label negatives remain first-class, but rows that have no label support can backfill negatives from batch proxy features such as log-energy statistics, a text/content-length speaking-rate proxy, voiced ratio, and voiced-frame `f0` spread. The prep summary now also exposes a small-batch runtime preview of which negative path is actually available (`label`, `proxy`, or `label_plus_proxy_backfill`) instead of only the artifact-level label-map view.

An additional optional research regularizer is now wired but still shipped as `0.0` by default: `lambda_style_timbre_runtime_overlap`. It does not claim disentanglement; it simply measures and, when enabled, penalizes excessive frame-wise overlap between `style_decoder_residual` and `dynamic_timbre_decoder_residual_prebudget`. Importantly, explicit ablation runs can now enable it while staying on `control_loss_profile: mainline_minimal`; the schedule layer no longer silently zeroes that opt-in regularizer.

Treat `style_decoder_residual` as the public owner-style contract. Keep `fast_style_decoder_residual`, `slow_style_decoder_residual`, router gates, and burst scores as **internal realization / diagnostics variables**, not semantically identifiable public factors. The style-side regularizers and the style-success anchor now also follow that same contract: they prefer `style_decoder_residual` directly instead of supervising a raw fast+slow combination and accidentally overweighting the slow branch.

Evaluation readiness is now split more honestly as well:

- `inference/run_style_profile_evaluation.py` no longer hard-crashes at import time when the optional `inference/research/` package is absent; factorized swap reporting is now lazy / best-effort
- explicit timbre/style/dynamic-timbre/emotion/accent reference metrics are no longer hidden behind `include_research_metadata`; if a sweep row actually carries explicit reference wavs, the evaluator now computes the matching validation metrics even on the default canonical path

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

This repo encodes the intended canonical mainline contract, but universal verification still depends on the **current local environment** and on running the prep gate plus smoke commands in that environment.
Treat `tasks/Conan/mainline_train_prep.py` and a short real-data smoke run as the source of truth for train-readiness.
That prep gate validates the shipped canonical config; optional research regularizers such as `lambda_style_timbre_runtime_overlap` remain opt-in ablations, not part of the default prep pass. It also now checks exact `requirements.txt` pins for `torch` / `torchaudio` / `torchdyn` / `textgrid`, plus Python-vs-pinned-`torchaudio` compatibility. So `code_contract_ready: true` no longer implies `train_ready_now: true`.

Key audited properties in the checked-in code:

- training / inference strong-style semantics are aligned
- canonical config uses `lambda_pitch_residual_safe`, not `lambda_dynamic_timbre_boundary`
- mainline `style_strength` requests are clamped and surfaced as requested/effective metadata
- canonical training batches now lock `style_strength` to the resolved `style_profile` unless `allow_item_style_strength_override` is explicitly enabled
- canonical mainline profile surface now only approves `style_strength` as a direct override; other profile-key deviations are treated as research overrides and coerced back unless explicitly allowed
- resolved mainline controls now stay authoritative for TVT prior/runtime flags and pitch-residual scale / semitone / smoothing
- style conditioning now consumes the resolved mainline `style_strength` instead of a hidden raw-kwargs alias
- dynamic-timbre budget internals (`runtime_dynamic_timbre_style_budget_slow_style_weight` / `runtime_dynamic_timbre_style_budget_epsilon`) and upper-bound curriculum knobs are hparam-owned on closed mainline rather than per-request overrides
- dynamic timbre is not allowed to leak into `style_to_pitch_residual` on the shipped mainline path
- when no external speaker verifier is configured, the identity loss is evaluated in a frozen internal speaker space instead of a trainable one
- decoder runtime now consumes `slow_style_trace` directly instead of keeping it as a log-only byproduct
- runtime dynamic-timbre budgeting now uses owner-style energy plus only the **slow-style excess** beyond that owner reference, while losses/diagnostics observe both the pre-budget residual and the realized decoder-stage deltas
- control-loss/runtime glue is now split one layer further:
  - `modules/Conan/style_timbre_runtime.py` owns query-side style assembly, timbre-query preparation, runtime budget application, and decoder bundle realization
- fast style / TVT timbre / pitch residual now share one upper-bound curriculum instead of all starting at step 0
- dynamic timbre now follows a consistent residual semantic on both TVT and non-TVT paths
- dynamic timbre boundary suppression no longer collapses to a global mask on dense HuBERT-style unit sequences
- style-to-pitch residual smoothing is post-canvas and mask-aware, and the safe loss now combines **robust target alignment + one-sided safe budget + target-slope matching** when a bounded target residual is available instead of flattening any raw local movement
- direct library-style inference (`set_hparams(..., global_hparams=False)` + `StreamingVoiceConversion(...)`) no longer requires the global singleton hparams dict on the canonical style/timbre path, although some legacy fallback reads still exist outside that main path
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
- `diag_runtime_dynamic_timbre_style_budget_overflow_mean/std`
- `diag_runtime_dynamic_timbre_style_budget_relative_overflow_mean/std`
- `diag_runtime_dynamic_timbre_style_owner_energy_mean/std`
- `diag_runtime_dynamic_timbre_slow_style_excess_energy_mean/std`
- `diag_decoder_stage_dynamic_timbre_budget_overflow_mean/std`
- `diag_style_timbre_runtime_cos`
- `diag_style_timbre_runtime_abs_cos`
- `diag_style_timbre_runtime_overlap_margin_violation`
- `diag_style_timbre_runtime_postbudget_abs_cos`
- `diag_dynamic_timbre_prebudget_to_style_energy_ratio`
- `diag_dynamic_timbre_postbudget_to_style_energy_ratio`
- `diag_style_success_pair_cos`
- `diag_style_success_hard_negative_cos`
- `diag_style_success_pair_margin`
- `diag_style_success_uses_weak_negative_ranking`
- `diag_style_success_uses_proxy_negative_ranking`
- `diag_style_success_label_negative_row_frac`
- `diag_style_success_proxy_negative_row_frac`
- `diag_style_success_negative_row_frac`
- `diag_style_success_negative_source_label`
- `diag_style_success_negative_source_proxy`
- `diag_style_success_negative_source_label_plus_proxy_backfill`
- `diag_style_success_target_source_runtime_reference_derived`
- `diag_style_success_target_source_global_reference_derived`
- `diag_style_success_target_source_none`
- `diag_identity_backend_is_external`
- `diag_identity_encoder_frozen_for_loss`
- `diag_output_identity_target_cos`

Local-audit guidance:

- this repo does **not** claim universal trainability from the checked-in code alone
- the expected minimum local verification is: `compileall` -> prep gate -> short real-data smoke -> short inference smoke
- keep the smoke budget small during audit work; this closure pass uses `<= 50` update examples rather than long warm-start claims

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

If the current local environment still has dependency pin drift or missing staged artifacts, the honest result is `MAINLINE_TRAIN_PREP_NOT_READY`; in that case, treat the JSON summary fields `code_contract_ready` vs `ready` / `train_ready_now` as the source of truth rather than assuming the repo text implies train-readiness.

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

Short warm-start audit smoke from the shipped Conan checkpoint (`<= 50` updates):

```bash
conda run -n conan python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineAuditSmoke8 --hparams "load_ckpt=checkpoints/Conan/model_ckpt_steps_200000.ckpt,ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=1000000,num_sanity_val_steps=0,max_updates=8,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[],dataloader_persistent_workers=False,dataloader_pin_memory=False,tb_log_interval=1,lambda_style_timbre_runtime_overlap=0.005"
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
