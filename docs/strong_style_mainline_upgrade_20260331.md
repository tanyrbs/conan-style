# Conan Strong-Style (Single-Reference) Mainline

Updated: 2026-04-01

## Summary

This repo is now aligned to a **single-reference, strong-style transfer** objective:

- **External contract:** one reference audio in, one `style_strength` knob out.
- **Internal structure:** minimal 2.5-factor split for stability and clarity:
  - **z_spk**: global timbre anchor (speaker identity)
  - **g_style**: utterance-level style summary
  - **M_style**: local style memory for decoder-side realization

We intentionally **do not** expose multi-reference, combinatorial control.  
This is a **product-first** configuration: *strong timbre + strong style + listenability*.

## What changed

### 1) Reference contract is single-reference only

The contract is always `collapsed_reference`.  
Any `ref_timbre/ref_style/ref_dynamic_timbre` is treated as **internal-only** or
falls back to `ref` if missing.

This removes the “strict factorized” pathway and avoids user-facing complexity.

### 2) Decoder-side style is the mainline

The style path is realized *only* on the decoder side:

- `g_style` contributes to decoder conditioning
- `M_style` provides local expressive replay
- no timing writeback

This keeps pitch/timing stable and makes strong style more predictable.

### 3) Dynamic timbre is no longer a main path

`dynamic_timbre` is disabled by default.  
It may be re-enabled later as a **pure inference smoothing tool** (e.g., boundary-only).

### 4) External control is simplified

We keep a single public knob:

- `style_strength`

All other internal controls are not exposed by default.  
Emotion/accent condition paths are **disabled by default** and can be re-enabled later if needed.

`style_id` supervision is removed by default (single-knob interface only).

## Presets

Only strong-style presets are kept for simplicity:

- `strong_style` (default)
- `extreme`

## Gate warmup

To encourage early stability and later stronger style:

- `decoder_style_adapter_gate_bias_start: -1.0`
- `decoder_style_adapter_gate_bias_end: 0.0`
- `decoder_style_adapter_gate_bias_warmup: 80000`

## Config cleanup

Single-reference configs have been simplified:

- removed dynamic timbre training knobs
- removed emotion/accent control heads
- removed style_id supervision

## Recommended training loss (practical version)

We focus on listenability and transfer strength rather than strict disentanglement:

```
L =
  L_mel + L_mrstft
  + λ_gan L_gan + λ_fm L_fm
  + λ_c   L_content
  + λ_spk L_spk
  + λ_style (L_style_global + L_style_stats)
  + λ_stream L_stream
```

Key points:
- **L_content**: content preservation via HuBERT/ASR/CTC.
- **L_spk**: speaker similarity via speaker encoder cosine/AAM.
- **L_style_global**: global style embedding match.
- **L_style_stats**: prosody statistics (F0/energy/pace/voiced ratio).
- **L_stream**: chunk continuity for streaming stability.

## Training schedule (2-stage)

1) **Stage 1:** stabilize timbre + naturalness  
   - keep `λ_style` low
   - ensure intelligibility + identity

2) **Stage 2:** style-heavy finetune  
   - increase expressive samples
   - freeze parts of content encoder / vocoder
   - allow decoder adapter + style encoders to learn stronger expression

## Design intent

> **External single reference. Internal weak factorization. Strong decoder-side injection.**

This is not a research-style combinatorial VC system.  
It is a **production-style “strong style, strong timbre” Conan**.
