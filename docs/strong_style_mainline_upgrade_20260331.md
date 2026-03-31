# Strong Style Mainline Upgrade Notes

Updated: 2026-03-31

## What changed in `Conan-master`

This round pushes the repo away from:

- one-reference silent collapse
- style-first query pollution
- pre-pitch style/timbre write-in

and closer to:

- explicit reference contract
- split `q_style / q_timbre`
- decoder-side realization
- `global_timbre_anchor` / `global_style_summary` separation

## Main structural upgrades

### 1. Reference contract is now explicit

Two modes are supported:

- `collapsed_reference`
- `strict_factorized`

`strict_factorized` now requires explicit `ref_style` and `ref_dynamic_timbre`. The batch/reference bundle now carries:

- `reference_contract_mode`
- `reference_contract`
- `factorization_guaranteed`

This prevents silent "three branches in name, one reference in reality".

### 2. Style no longer pollutes dynamic timbre query

The model now builds:

- `style_query_inp`
- `timbre_query_inp`

from the shared content/condition trunk, instead of letting `dynamic_timbre` read a state already contaminated by style residuals.

### 3. Global timbre anchor and global style summary are now separated

The repo no longer treats one tensor as both:

- utterance timbre anchor
- utterance style summary

The maintained reference cache now carries:

- `global_timbre_anchor`
- `global_style_summary`

`style_embed` is retained only as a compatibility alias of the timbre anchor.

### 4. Style/timbre moved into a decoder-side adapter

The strong-style branches are no longer written into the pitch branch first.

Current path:

`base_condition_inp -> pitch branch`

`base_condition_inp -> q_style / q_timbre`

`aligned style/timbre -> decoder_style_adapter -> causal decoder hidden`

The new adapter is lightweight but already stage-aware:

- `mid`: dynamic timbre
- `late`: global style summary + style trace + dynamic timbre

### 5. Mainline surface metadata was added

The model now exports a compact runtime surface:

- `decoder_style_condition_mode`
- `global_style_anchor_strength`
- `style_trace_applied`
- `dynamic_timbre_applied`
- `decoder_side_only = true`
- `timing_writeback_allowed = false`

### 6. Scheduled lambdas with base zero are no longer silently ignored

If a scheduled control lambda has base value `0.0`, it must now provide explicit `target` or `value`.

This fixes the old "warm up a zero, end at zero" failure mode.

## Why this direction

This upgrade is aligned with the cleaner local Conan style route:

- explicit contract first
- factor roles before regularizing them
- move style realization later
- keep timing ownership separate from style ownership

## Literature direction used for this round

- TVTSyn — https://arxiv.org/abs/2409.00623
- PromptVC — https://arxiv.org/abs/2309.15778
- StableVC — https://arxiv.org/abs/2412.04724
- PFlow-VC — https://arxiv.org/abs/2406.00865
- StyleStream — https://arxiv.org/abs/2507.05860
- Takin-VC — https://arxiv.org/abs/2411.13676
- Seed-VC — https://arxiv.org/abs/2411.09943
- Vevo — https://arxiv.org/abs/2410.10657

## Still not finished

The repo is still not a full fixed-clock CSSF system.

Remaining high-value steps:

1. split slow style memory vs fast local style replay more clearly
2. add stronger boundary-aware local timbre control
3. upgrade the lightweight decoder adapter toward richer mid/high-layer CSSF-style gating
4. add external evaluation: speaker / ASR / prosody encoders

## 2026-03-31 follow-up: dual style trace split

This round adds a cleaner dual-path style realization option:

- `style_trace_mode = dual`
- fast branch = local replay on fast prosody memory
- slow branch = posture / delivery on slow pooled prosody memory
- decoder adapter can now consume both branches without collapsing them into one trace
- `strict_factorized` training batches now require explicit `ref_timbre_mels`

This is still not full CSSF, but it pushes the style mainline closer to:

- slow style as utterance / phrase posture
- fast style as local expressive replay
- dynamic timbre as separate local voice-color residual
