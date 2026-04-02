# Conan inference mainline

This repo now keeps only the Conan mainline inference surface.

## Canonical files

- config: `egs/conan_mainline_infer.yaml`
- model entry: `checkpoints/Conan/model_ckpt_steps_200000.ckpt`
- Emformer: `checkpoints/Emformer/model_ckpt_steps_700000.ckpt`
- vocoder: `checkpoints/hifigan_vc/model_ckpt_steps_1000000.ckpt`

## Public runtime contract

The retained Conan mainline surface is:

- `src_wav`
- `ref_wav`
- optional `style_profile`
- optional `style_strength`

Mainline semantics stay fixed to:

> single reference -> stable identity -> strong style owner -> bounded dynamic timbre enhancement

Notes:

- `collapsed_reference` is the only shipped contract
- `style_profile: strong_style` is the canonical default surface
- split-reference / factorized research paths are no longer kept in this repo snapshot
- online/offline streaming parity is expected to match in mel length and wav length

## Verified inference fixes as of 2026-04-03

- style-profile defaults now flow cleanly into runtime controls
- streaming prefix inference trims missing tail right-context correctly
- online/offline parity checks now match on both mel length and wav sample length
- `EmformerDistillModel` respects `emformer_mode` as the canonical config key

## Runnable entrypoints

### Single-reference conversion

Edit `inference/conan_single_reference_demo.example.json`, then run:

```bash
python inference/run_voice_conversion.py   --pair_config inference/conan_single_reference_demo.example.json
```

### Style-profile sweep

Edit `inference/conan_style_profile_sweep.example.json`, then run:

```bash
python inference/run_style_profile_sweep.py   --sweep_config inference/conan_style_profile_sweep.example.json

python inference/run_style_profile_evaluation.py   --sweep_dir infer_out_profiles/conan_mainline_demo

python inference/run_style_profile_report.py   --sweep_dir infer_out_profiles/conan_mainline_demo
```

## Related training doc

- `docs/canonical_training_mainline_20260401.md`
