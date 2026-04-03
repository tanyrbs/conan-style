# Conan canonical training mainline

Updated: 2026-04-04

## 1. Scope

This cleaned repo snapshot keeps only the Conan single-reference strong-style mainline.

Mainline target:

> single reference -> stable identity -> strong style owner -> bounded dynamic timbre enhancement -> decoder-side fusion

## 2. Canonical configs

Only these user-facing entrypoint configs remain:

- binarization: `egs/conan_binarize.yaml`
- training: `egs/conan_emformer.yaml`
- inference: `egs/conan_mainline_infer.yaml`

Both are aligned to the same mainline semantics, including `style_profile: strong_style`.

Config layering is now explicit:

- `egs/egs_bases/conan/mainline_shared.yaml`
- `egs/egs_bases/conan/mainline_data.yaml`
- `egs/egs_bases/conan/mainline_stats.yaml`
- `egs/egs_bases/conan/mainline_train.yaml`

## 3. Mainline contract

Training / inference stay locked to:

- `reference_contract_mode: collapsed_reference`
- `decoder_style_condition_mode: mainline_full`
- `style_trace_mode: dual`
- `style_router_enabled: true`
- `style_strength` stays inside the shipped mainline range `0.50 .. 1.80`
- `allow_item_style_strength_override: false`, so training batches keep the resolved profile strength unless you explicitly opt into a research override
- `dynamic_timbre_strength` is derived from `style_strength` on the canonical path, so these are partially coupled controls rather than fully independent free variables
- `style_to_pitch_residual: true`
- `style_to_pitch_residual_include_timbre: false`
- `global_timbre_to_pitch: false`
- post-rhythm pitch-canvas projection is still best-effort in the shipped path; if no rhythm frame-index canvas is emitted, runtime falls back to source-aligned residual application and reports the realized canvas/fallback reason
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

Canonical control regularization keeps five nonzero-configured losses on the shipped path.
Their effective strengths still follow the schedule, so this is the configured loss surface once the schedule opens, not a claim that all five are fully active from step 0:

- `lambda_output_identity_cosine`
- `lambda_dynamic_timbre_budget`
- `lambda_pitch_residual_safe`
- `lambda_decoder_late_owner`
- `lambda_style_success_rank`

This is intentionally a **bounded weak internal factorization** contract, not a proof of perfect disentanglement. The mainline explicitly reports `factorization_guaranteed: false`; the active losses constrain lower-level failure modes, but they do not prove that each latent/control branch has become uniquely interpretable.

The new `lambda_style_success_rank` is a training-only lower-bound signal rather than a new inference knob: it combines paired self/reference style alignment with weak-label batch ranking whenever metadata negatives are available, while leaving the shipped runtime surface closed. `tasks/Conan/mainline_train_prep.py` now reports whether the staged artifacts imply `paired_plus_weak_label_ranking` or only `paired_only`; if staged condition artifacts expose no usable label buckets (`num_labels <= 1`), the weak-label branch naturally collapses toward paired alignment instead of informative cross-label ranking. In the currently staged LibriTTS-single artifacts in this checkout, those weak-label buckets are empty, so the artifact-level preview resolves to `paired_only`.

Separately, the codebase now exposes an optional runtime-separation regularizer, `lambda_style_timbre_runtime_overlap`, but keeps it at `0.0` in the shipped canonical config. Its purpose is diagnostic/ablation-oriented: measure and optionally penalize excessive frame-wise overlap between `style_decoder_residual` and `dynamic_timbre_decoder_residual_prebudget`, not to assert that true disentanglement has been proved. Explicit ablation runs can now enable it without leaving `control_loss_profile: mainline_minimal`; the schedule layer no longer silently zeros that opt-in regularizer.

When `use_external_speaker_verifier: false`, the identity loss is evaluated through the model's own speaker encoder,
but the canonical config now freezes that internal encoder during the auxiliary identity-loss pass so the loss behaves like a fixed reference space rather than a moving target.

This closure pass also tightens two previously weakly-realized parts of the contract:

- `slow_style_trace` is now actually passed into the decoder runtime bundle, so the decoder's coarse slow-style path is no longer only described in theory
- runtime dynamic-timbre budgeting now references owner-style energy plus only the **slow-style excess** beyond that owner reference, while the training loss supervises both the **pre-budget** dynamic-timbre residual and the realized decoder-stage deltas
- `lambda_pitch_residual_safe` now uses robust target alignment (`SmoothL1`), adds a one-sided safe budget against the bounded target magnitude, and matches **target slope** when the bounded target exists; when the target is absent it keeps the conservative `zero-anchor + raw smoothness` fallback
- runtime style/timbre glue is now factored into `modules/Conan/style_timbre_runtime.py`, so `Conan.forward()` stays orchestration-oriented instead of mixing query preparation, owner resolution, timbre runtime, and decoder-bundle assembly in one block

Canonical mainline profile parsing is also stricter now:

- direct mainline overrides are intentionally limited to `style_strength`
- other profile-key deviations are treated as research overrides and coerced back to the approved mainline defaults unless `allow_mainline_profile_research_overrides: true`

## 4. Data prerequisites

Do **not** infer readiness from repository prose alone. Some checkouts may already have staged processed/binary datasets and checkpoints at the canonical paths, while others may not. The commands below describe the intended mainline flow; the prep gate and smoke runs decide whether the current local checkout is actually ready.

Default repo paths:

- `data/binary/libritts_single`
- `data/processed/libritts_single`

Current checked-in LibriTTS-single behavior:

- `valid_prefixes` / `test_prefixes` are intentionally empty
- binarization falls back to deterministic per-speaker utterance holdout
- binary `*_ref_indices.npy` use a stable hash, so reference pairing stays reproducible across Python processes
- `VCBinarizer` owns the shared VC item path, while `ConanBinarizer` only adds cached offline `f0`
- `EmformerBinarizer` is intentionally a thin subclass and no longer duplicates Conan item logic
- condition artifact generation now comes from one path and emits aligned `*_map.json` + `*_set.json`
- optional metadata shuffle no longer perturbs valid/test selection; it only shuffles the final train split order

## 5. Offline F0 extraction

`egs/conan_binarize.yaml` uses `data_gen.conan_binarizer.ConanBinarizer`, so per-utterance offline F0 is required before binarization.

```bash
python utils/extract_f0_rmvpe.py --config egs/conan_binarize.yaml --pe-ckpt <path-to-rmvpe.pt> --batch-size 8 --max-tokens 40000
```

After F0 extraction, rebuild the binary dataset:

```bash
$env:N_PROC='1'; python data_gen/tts/runs/binarize.py --config egs/conan_binarize.yaml
```

## 6. Training-prep gate

The prep gate is the authoritative train-readiness check for your local environment. Even when data/checkpoints are already present locally, do not treat this document as a universal claim that training has already been verified everywhere.
It validates the shipped canonical config; optional ablation knobs such as `lambda_style_timbre_runtime_overlap` are deliberately outside the default prep contract unless you opt into them yourself.

Before real training, run:

```bash
python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml
```

Expected result:

- `MAINLINE_TRAIN_PREP_OK`

This prep gate now also checks that style-profile defaults really control mainline runtime strengths instead of silently falling back to neutral values.
When the environment is incomplete, the JSON summary now separates `code_contract_ready` from the stricter `ready` / `train_ready_now` flags so mainline code-contract failures are not conflated with missing staged data or missing runtime libraries.
It also checks exact `requirements.txt` pins for `torch` / `torchaudio` / `torchdyn` / `textgrid`, plus Python-vs-pinned-`torchaudio` compatibility, so a local checkout can legitimately report `code_contract_ready: true` while still failing `train_ready_now`.

It now also checks that the shipped runtime dynamic-timbre budget contract matches the canonical mainline hparams, including the public budget ratio / margin plus the internal hparam-owned budget stabilization scalars:

- `runtime_dynamic_timbre_style_budget_ratio`
- `runtime_dynamic_timbre_style_budget_margin`
- `runtime_dynamic_timbre_style_budget_slow_style_weight`
- `runtime_dynamic_timbre_style_budget_epsilon`

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

Recommended short review-env smoke after staging data and a compatible runtime:

```bash
conda run -n conan python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineAuditSmokeCpu --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=1,num_sanity_val_steps=0,max_updates=2,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[]"
```

Short warm-start audit smoke from the shipped Conan checkpoint (`<= 50` updates):

```bash
conda run -n conan python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineAuditSmoke8 --hparams "load_ckpt=checkpoints/Conan/model_ckpt_steps_200000.ckpt,ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=1000000,num_sanity_val_steps=0,max_updates=8,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[],dataloader_persistent_workers=False,dataloader_pin_memory=False,tb_log_interval=1,lambda_style_timbre_runtime_overlap=0.005"
```

Notes:

- keep shipped inference checkpoint entry `Conan` untouched
- use a fresh experiment name for real training output
- `max_updates` now stops exactly at the requested batch budget instead of overshooting by one step
- task-side `load_ckpt` warm starts now default to non-strict loading, so older compatible Conan checkpoints can still seed current mainline smoke/fine-tune runs
- because `slow_style_trace` now really participates in decoder runtime fusion, old checkpoints should be A/B checked if you care about exact pre/post-closure forward parity

## 8. Mainline inference / evaluation after training

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

Notes:

- the evaluation entrypoint now lazy-loads the optional research-only factorized report writer, so the canonical evaluator no longer crashes just because `inference/research/` is absent
- explicit timbre/style/dynamic-timbre/emotion/accent reference metrics are computed whenever the sweep metadata actually provides those explicit reference wavs; they are no longer hidden behind the research-metadata flag

Streaming latency instrumentation:

```bash
python inference/run_streaming_latency_report.py
```

## 9. Audited implementation notes / intended invariants

- canonical mainline uses `lambda_pitch_residual_safe`; `lambda_dynamic_timbre_boundary` remains `0.0`
- requested vs effective `style_strength` is surfaced explicitly, so clamp events are observable
- canonical training batches keep `style_profile` authoritative unless `allow_item_style_strength_override` is deliberately enabled
- resolved mainline controls now remain authoritative for TVT prior/runtime flags and pitch-residual scale / semitone / smoothing
- `style_to_pitch_residual` is style-led on the shipped path; dynamic timbre is not allowed to enter that residual unless a research override is enabled
- when no external speaker verifier is configured, the identity loss is computed in a frozen internal speaker-embedding space rather than letting that auxiliary encoder drift during the same loss path
- `slow_style_trace` now enters the decoder runtime bundle directly, so decoder-stage authority separation better matches the intended mainline theory
- `lambda_dynamic_timbre_budget` now observes the pre-budget dynamic-timbre residual **and** the realized decoder-stage deltas, while the style-side reference is built from owner-style energy plus only the slow-style excess beyond that owner reference, avoiding a slow-style double count
- the dynamic-timbre budget loss no longer hard-drops unvoiced / weak frames; it uses a soft uv floor plus relative energy support so fricatives, breathy segments, and other low-periodicity regions remain partially supervised instead of becoming a blind spot
- fast style / TVT timbre / pitch residual no longer start fully open at step 0; they share one `20000 -> 80000` upper-bound curriculum during training
- dynamic timbre now uses a unified residual semantic on both TVT and non-TVT paths (`local_delta` relative to the global timbre anchor)
- dynamic-timbre boundary suppression now detects dense HuBERT-like unit streams and avoids turning token-transition boundaries into a global mask
- `style_to_pitch_residual` smoothing is applied after projection onto the final pitch canvas and is mask-aware; when a bounded target residual exists, the safe loss now combines **robust target alignment + one-sided magnitude budget + target-slope matching** instead of penalizing any raw residual variation equally
- direct library-style inference (`set_hparams(..., global_hparams=False)` + `StreamingVoiceConversion(...)`) no longer requires the global singleton hparams dict on the canonical style/timbre path, although some legacy fallback reads still exist outside that main path
- `VCBinarizer.process_item(...)` is now the single shared VC item-processing path; Conan only overrides frame-feature loading for cached `f0`
- binary indexed dataset loading is robust across the current NumPy 1.x/2.x artifact boundary
- runtime maintainability is now explicitly layered:
  - `modules/Conan/common_utils.py` for shared helper resolution / sequence expansion
  - `modules/Conan/pitch_canvas_utils.py` for pitch-canvas runtime semantics
  - `modules/Conan/decoder_style_runtime.py` for decoder style bundle contract assembly
  - `modules/Conan/style_timbre_runtime.py` for query-side style/timbre runtime realization
- layer-2 maintainability refactor now also keeps:
  - `modules/Conan/pitch_runtime.py` as the mixin layer for pitch generation + style-to-pitch runtime logic
  - `modules/Conan/common.py` as the single source of truth for lightweight mapping lookup helpers
  - `inference/conan_request.py` as the single source of truth for canonical inference request schema helpers
- these internal modules do not add new public knobs; they only reduce duplicated glue code and contract drift
- streaming inference tail trimming has been aligned with offline decoding, so online/offline mel and wav lengths now match on parity checks
- streaming runtime math / latency reporting is now centralized in `inference/streaming_runtime.py`
- shipped vocoder wrappers now expose an explicit non-native-streaming capability surface
- `utils/extract_f0_rmvpe.py` now boots from the repo root, accepts raw RMVPE state-dict checkpoints, and can batch even when metadata lacks explicit duration fields
- `EmformerDistillModel` now reads `emformer_mode` correctly instead of only the legacy `mode` key
- Emformer distillation train / infer / validation paths now align targets to the actual streamed-logit length
- closure-pass diagnostics now expose budget and identity semantics more directly:
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
  - `diag_identity_backend_is_external`
  - `diag_identity_encoder_frozen_for_loss`
  - `diag_output_identity_target_cos`

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
- streaming latency note: `docs/streaming_low_latency_mainline_note_20260403.md`
