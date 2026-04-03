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
- `style_strength` stays inside the shipped mainline range `0.50 .. 1.80`
- `style_to_pitch_residual: true`
- `style_to_pitch_residual_include_timbre: false`
- `global_timbre_to_pitch: false`
- `allow_split_reference_inputs: false`

Canonical training also keeps the aggressive upper-bound modules behind one shared curriculum:

- `upper_bound_curriculum_enabled: true`
- `upper_bound_curriculum_start_steps: 20000`
- `upper_bound_curriculum_end_steps: 80000`

This gates:

- the fast style branch inside dual trace
- the TVT dynamic timbre path
- the bounded style-to-pitch residual range

Inference always uses the fully opened ceiling.

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
- `VCBinarizer` owns the shared VC item path, while `ConanBinarizer` only adds cached offline `f0`
- `EmformerBinarizer` is intentionally a thin subclass and no longer duplicates Conan item logic

## 5. Offline F0 extraction

`egs/conan_emformer.yaml` uses `data_gen.conan_binarizer.ConanBinarizer`, so per-utterance offline F0 is required before binarization.

```bash
python utils/extract_f0_rmvpe.py --config egs/conan_emformer.yaml --pe-ckpt <path-to-rmvpe.pt> --batch-size 8 --max-tokens 40000
```

After F0 extraction, rebuild the binary dataset:

```bash
$env:N_PROC='1'; python data_gen/tts/runs/binarize.py --config egs/conan_emformer.yaml
```

## 6. Training-prep gate

Before real training, run:

```bash
python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml
```

Expected result:

- `MAINLINE_TRAIN_PREP_OK`

This prep gate now also checks that style-profile defaults really control mainline runtime strengths instead of silently falling back to neutral values.

It also checks that the upper-bound curriculum is enabled, lands on the expected progress points, and stays aligned with forcing / reference curriculum timing.

It also samples real training data to make sure dynamic-timbre boundary suppression is not degenerating into an all-ones global mask on dense unit sequences.

It also checks that binary train / valid / test splits exist and are non-empty.

Existing binary indexed datasets created under NumPy 2.x are also kept readable in the shipped NumPy 1.x conda env, so prep does not fail spuriously on `numpy._core` pickle references.

## 7. Real training command

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineTrain
```

Exact-step real-data smoke command:

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineSmoke --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=100000,num_sanity_val_steps=0,max_updates=1,save_codes=[]"
```

Short review-env smoke that was used during the latest audit:

```bash
conda run -n conan python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineAuditSmokeCpu --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=1,num_sanity_val_steps=0,max_updates=2,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[]"
```

Notes:

- keep shipped inference checkpoint entry `Conan` untouched
- use a fresh experiment name for real training output
- `max_updates` now stops exactly at the requested batch budget instead of overshooting by one step

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
- requested vs effective `style_strength` is now surfaced explicitly, so clamp events are observable
- resolved mainline controls now remain authoritative for TVT prior/runtime flags and pitch-residual scale / semitone / smoothing
- `style_to_pitch_residual` is style-led on the shipped path; dynamic timbre is not allowed to enter that residual unless a research override is enabled
- fast style / TVT timbre / pitch residual no longer start fully open at step 0; they share one `20000 -> 80000` upper-bound curriculum during training
- dynamic timbre now uses a unified residual semantic on both TVT and non-TVT paths (`local_delta` relative to the global timbre anchor)
- dynamic-timbre boundary suppression now detects dense HuBERT-like unit streams and avoids turning token-transition boundaries into a global mask
- `style_to_pitch_residual` smoothing is applied after projection onto the final pitch canvas and is mask-aware
- `VCBinarizer.process_item(...)` is now the single shared VC item-processing path; Conan only overrides frame-feature loading for cached `f0`
- binary indexed dataset loading is robust across the current NumPy 1.x/2.x artifact boundary
- runtime maintainability is now explicitly layered:
  - `modules/Conan/common_utils.py` for shared helper resolution / sequence expansion
  - `modules/Conan/pitch_canvas_utils.py` for pitch-canvas runtime semantics
  - `modules/Conan/decoder_style_runtime.py` for decoder style bundle contract assembly
- layer-2 maintainability refactor now also keeps:
  - `modules/Conan/pitch_runtime.py` as the mixin layer for pitch generation + style-to-pitch runtime logic
  - `modules/Conan/common.py` as the single source of truth for lightweight mapping lookup helpers
  - `inference/conan_request.py` as the single source of truth for canonical inference request schema helpers
- these internal modules do not add new public knobs; they only reduce duplicated glue code and contract drift
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
- configured Conan / Emformer / vocoder checkpoint paths (actual large checkpoint binaries may be omitted from lightweight review archives)
