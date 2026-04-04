# Project audit update - 2026-04-04

This note records the 2026-04-04 audit pass that focused on implementation risk,
performance headroom, dataset compatibility, and regression coverage.

## Audit lanes

The review was split across six lanes:

1. model/runtime correctness
2. training/task plumbing
3. inference/streaming path
4. data pipeline and binary artifacts
5. tests/docs/dependency surface
6. performance hotspots

## Checks run in this pass

- `python -m pytest -q`
- `python -m compileall modules tasks inference data_gen utils tests`
- `python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml`
- `python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanAuditStep1_20260404 --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=100000,num_sanity_val_steps=0,max_updates=1,save_codes=[]"`

The short one-step training smoke completed successfully in this environment.

## Shipped fixes

### 1) Training hot path no longer duplicates CUDA transfers per optimizer

File:

- `utils/commons/trainer.py`

What changed:

- the training loop now moves a batch to CUDA once per step instead of once per optimizer
- the already-prepared batch is shallow-copied per optimizer, preserving the old call shape
- optimizer clearing now uses `zero_grad(set_to_none=True)`

Why this matters:

- Conan training uses generator + discriminator optimizers
- the previous behavior paid the host->device transfer cost twice for the same batch
- the new path improves throughput headroom and reduces avoidable memory traffic

### 2) Inference request setup avoids recomputing identical reference mels

File:

- `inference/Conan.py`

What changed:

- split-reference mel loading now deduplicates repeated wav paths within one request
- inference-only sections now use `torch.inference_mode()`

Why this matters:

- repeated `ref_*_wav` aliases no longer trigger repeated feature extraction
- inference-mode reduces autograd bookkeeping on the streaming/offline front-end path

### 3) Indexed dataset sidecars are cheaper, safer, and backward compatible

File:

- `utils/commons/indexed_datasets.py`

What changed:

- new `.idx` files are written as a compact numeric `int64` offsets array
- the loader accepts both the new numeric format and the older dict-style payload
- the NumPy compatibility shim now prefers `numpy._core` before falling back to `numpy.core`
- `IndexedDataset` now exposes `close()` / context-manager support for explicit file-handle cleanup

Why this matters:

- avoids `allow_pickle=True` on the normal new-format read path
- removes noisy `numpy.core` deprecation warnings in typical environments
- keeps older binary artifacts readable while making new ones simpler and faster
- explicit close support helps Windows temp-dir cleanup and test hygiene

### 4) Data-loader and control helpers were tightened for determinism and edge cases

Files:

- `tasks/tts/dataset_utils.py`
- `modules/Conan/control/separation_metrics.py`
- `modules/Conan/control/style_success.py`
- `modules/Conan/pitch_runtime.py`
- `modules/Conan/prosody_util.py`

What changed:

- dataset bucket construction now uses deterministic, keyed RNG streams derived from `seed`
- several data-path tensor conversions now use `torch.as_tensor(...)` helpers to avoid extra copies
- dynamic-timbre energy scaling and style-success proxy backfill were vectorized
- a few masked reductions now clamp denominators to avoid divide-by-zero on pathological batches
- prosody forcing mask setup now uses `torch.arange(...)` instead of rebuilding Python lists

Why this matters:

- more reproducible sampling support across runs/workers
- lower Python overhead in the dataset path
- less per-row Python-loop overhead in control metrics
- safer behavior on all-masked / degenerate minibatches

## Regression coverage added/updated

Tests now cover:

- numeric `.idx` sidecar emission
- legacy dict-style `.idx` compatibility
- indexed-dataset roundtrip reads
- NumPy pickle-compat alias installation without the deprecated import path
- split-reference mel deduplication in `StreamingVoiceConversion._load_reference_mels`
- single CUDA transfer per batch in the multi-optimizer trainer path
- deterministic speaker/condition bucket construction

## Prep-gate result on this machine

`python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml` returned:

- `code_contract_ready: true`
- `environment_ready: false`
- `data_ready: true`
- `data_dependent_preview_ready: true`
- terminal status: `MAINLINE_TRAIN_PREP_NOT_READY`

The remaining blockers are environment drift, not a code-contract failure. The current prep report flags:

- installed `torch`: `2.7.0+cpu` vs pinned `2.5.1`
- installed `torchaudio`: `2.7.0+cpu` vs pinned `2.5.1`
- installed `nltk`: `3.9.1` vs pinned `3.8.1`
- Python `3.13` outside the pinned `torchaudio 2.5.1` compatibility window
- missing NLTK resource: `averaged_perceptron_tagger_eng`

## Remaining follow-up opportunities

Worth tracking next:

1. continue reducing legacy global-`hparams` coupling outside the already-audited mainline path
2. profile the streaming prefix-recompute path end-to-end and decide whether additional reference/runtime caching is worth the complexity
3. if inference latency matters on this machine, fix the pinned runtime mismatch first so performance comparisons are meaningful
