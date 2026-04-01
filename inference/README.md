# Inference surface

## Mainline product path

Use these for the current single-reference Conan mainline:

- `Conan.py`
- `run_voice_conversion.py`
- `run_style_profile_sweep.py`
- `style_profile_sweep.example.json`

Default contract:

- `src_wav`
- `ref_wav`
- optional `style_profile` (`strong_style` / `extreme`)

Split references are supported only as opt-in compatibility / ablation behavior.

## Current-stage focus

The current phase is **mainline consolidation**, not feature sprawl.

So inference-side priorities are:

1. keep the default path single-reference
2. make examples / runners tell the same story
3. keep research-only tooling from leaking into the product surface
4. improve streaming correctness without pretending prefix-recompute is already fully incremental

## Future inference extensions

Planned / acceptable extensions:

- true incremental decoder cache
- stateful vocoder
- clearer online/offline parity diagnostics
- stricter separation between mainline and research-only runners

## Research-only / ablation tooling

These files are not the default product surface:

- `Conan_previous.py`
- `run_voice_conversion_nvae.py`
- `build_libritts_factorized_swap_matrix.py`
- `factorized_swap_builder.py`
- `factorized_swap_report.py`
- `libritts_factorized_swap*.json`
- `run_style_profile_evaluation.py` when used with factorized swap rows

They remain useful for analysis, but they should not be treated as the canonical user path.
