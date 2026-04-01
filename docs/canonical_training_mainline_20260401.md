# Conan canonical training / dry run / eval recipe

Updated: 2026-04-01

## 1. Scope

This document defines the **canonical Conan mainline training surface**.

Current goal is not multi-reference control research.  
Current goal is:

> **single reference -> stable identity -> strong style owner -> bounded timbre enhancer -> decoder-side fusion**

Mainline training therefore assumes:

- `reference_contract_mode: collapsed_reference`
- `decoder_style_condition_mode: mainline_full`
- `global_timbre_to_pitch: false`
- `style_trace_mode: slow`
- `allow_split_reference_inputs: false`
- `emit_collapsed_reference_aliases: false`

## 2. Canonical configs

### Training config

- `egs/conan_emformer.yaml`

### Inference / eval config

- `egs/conan_mainline_infer.yaml`

## 3. Data prerequisites

Expected mainline training data directories:

- `binary_data_dir`
- `processed_data_dir`

Default repo values point to:

- `data/binary/libritts_single`
- `data/processed/libritts_single`

For local smoke / CPU dry run, prefer a small smoke dataset such as:

- `data/binary/libritts_single_smoke`

## 4. Training-prep gate

Before the first real-dataset run:

```bash
python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml
```

Expected:

- `MAINLINE_TRAIN_PREP_OK`

This checks that the mainline contract is still locked to the intended defaults and that the required data dirs exist.

## 5. Canonical CPU local dry run

Before launching a real run, do one minimal CPU dry run:

```bash
python tasks/Conan/mainline_cpu_dry_run.py ^
  --config egs/conan_emformer.yaml ^
  --binary_data_dir data/binary/libritts_single_smoke ^
  --num_steps 2
```

Expected:

- `MAINLINE_CPU_DRY_RUN_OK`

This dry run performs:

1. mainline training-prep check
2. a tiny CPU training smoke on the canonical mainline config

## 6. Canonical real training command

Use the canonical mainline training config with a fresh experiment name:

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineTrain
```

Notes:

- do **not** overwrite the shipped inference checkpoint entry `Conan`
- use a new experiment name for actual training outputs

## 7. Canonical regression / eval commands

### Mainline contract smoke

```bash
python tasks/Conan/style_mainline_smoke.py --config egs/conan_mainline_infer.yaml
```

### Prefix-online parity smoke

```bash
python tasks/Conan/streaming_prefix_online_smoke.py --config egs/conan_mainline_infer.yaml
```

### Real wav parity smoke

```bash
python inference/streaming_parity_smoke.py ^
  --exp_name Conan ^
  --src_wav <src.wav> ^
  --ref_wav <ref.wav>
```

## 8. What “training ready” means in the current repo

At this stage, “training ready” means:

- mainline defaults are locked
- single-reference contract is enforced on the product path
- style/timbre query hierarchy is owner-aware
- training-prep check passes
- CPU dry run passes
- prefix-online parity smoke passes

It does **not** yet mean:

- fully stateful decoder streaming
- fully stateful vocoder streaming
- final product hyperparameters are frozen

## 9. Expected artifacts from a real run

For a normal mainline run you should expect:

- checkpoints under the experiment work dir
- tensorboard logs / scalar logs
- generated valid samples
- style mainline smoke results
- prefix-online parity smoke results

## 10. Recommended first real-dataset launch sequence

Recommended order:

1. `mainline_train_prep.py`
2. `mainline_cpu_dry_run.py`
3. short real-dataset launch
4. `style_mainline_smoke.py`
5. `streaming_prefix_online_smoke.py`

This keeps the current stage honest:

- **offline training first**
- **prefix-online parity as validation target**
- **no pretending strict streaming is already solved**
