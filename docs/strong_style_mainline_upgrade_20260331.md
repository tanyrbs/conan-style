# Conan Strong-Style Mainline

Updated: 2026-04-01

## 0. Current phase

This repo is currently in a **mainline consolidation phase**, not an expansion phase.

That means the practical priority order is:

1. make the single-reference path cleaner
2. harden the owner hierarchy
3. stabilize training / inference / streaming behavior
4. only then consider further control or deployment expansion

So the near-term goal is **not** to expose more factors.  
It is to make the current mainline more internally consistent, more testable, and more deployment-ready.

## 1. System target

This branch is converging to a **single-reference, strong-style, strong-material VC** mainline:

> **single reference**  
> → **global timbre anchor** stabilizes identity  
> → **M_style** carries the main expression  
> → **M_timbre** adds bounded material/voice-quality variation  
> → **decoder-side fusion only**

We explicitly do **not** optimize for:

- multi-reference combinatorial control
- public factorized timbre/style control UI
- writing style/timbre back into timing or pitch authority

## 2. External contract

Default product surface:

- `src_wav`
- `ref_wav`
- `style_strength`
- optional `style_profile`

`ref_timbre/ref_style/ref_dynamic_timbre` remain available only for research ablation.  
The canonical reference contract is **`collapsed_reference`**.

## 3. Internal owner hierarchy

### 3.1 `global_timbre_anchor`

Identity owner:

- keeps speaker identity stable
- should not become a style carrier
- should not dominate pitch/timing paths

### 3.2 `M_style`

Expression owner:

- prosodic posture
- emotional tension
- local emphasis / delivery pattern
- the main source of “strong style”

### 3.3 `M_timbre`

Bounded material residual:

- breathiness / brightness / thickness
- onset / release texture
- local voice-quality activity
- should enhance style realization, **not replace it**

In other words:

> **M_style is the owner. M_timbre is the enhancer.**

## 4. What is already correct in the codebase

The current repo already contains the right backbone:

- single-reference normalization via `collapsed_reference`
- explicit `global_timbre_anchor / global_style_summary / M_style / M_timbre`
- `decoder_style_bundle`
- `decoder_style_adapter`
- `timing_writeback_allowed = false`
- decoder-side conditioning as the mainline

This means the repo is **not directionally wrong**.  
The main remaining work is to **tighten role boundaries**, not to redesign from scratch.

## 5. Critical assessment: remaining gaps

### 5.1 Streaming is not yet strictly incremental

Current `inference/Conan.py` does this:

- **Emformer** is stateful
- reference cache is prepared once and reused
- but the main model still re-runs on the entire **content prefix**
- the vocoder now only re-synthesizes a **bounded left-context mel window**, not the full mel prefix

So the current online path is:

> **stateful content encoder + prefix-recompute acoustic model + bounded-context vocoder re-synthesis**

This is useful for development, but it is **not** the final strict-incremental streaming form.

### 5.2 Owner hierarchy still needs to be enforced harder

Even with explicit branches, the model can still drift if:

- `M_timbre` budget becomes too large
- late-stage timbre injection competes with `M_style`
- anchor/query semantics remain mixed with older aliases

Recent patches now move the code in the right direction:

- dynamic timbre is enabled by default
- `M_timbre` is conditioned on style context
- late-stage timbre injection is attenuated
- a timbre budget regularization term is introduced

But the hierarchy still needs continued empirical tuning.

### 5.3 “Pure decoder-only semantics” is not fully complete yet

The code already guarantees:

- no style/timbre timing writeback
- no planner/projector writeback

However, there is still some pre-decoder query construction involving the anchor/query path.  
This is acceptable short-term, but the long-term target remains:

> style/timbre should influence **realization**, not seize timing authority

## 6. Current implementation policy

### 6.1 Single-reference first

Default behavior is product-oriented:

- split references are off by default
- single-knob interface is preferred
- emotion/accent public conditioning stays secondary

### 6.2 Decoder-side realization only

Required invariants:

- `timing_writeback_allowed = false`
- `global_timbre_to_pitch = false` by default
- style/timbre enter the system as decoder-side realization signals

### 6.3 M_style > M_timbre

Current defaults now reflect the intended order:

- `M_style` gets the larger late-stage expressive budget
- `M_timbre` is kept as a smaller, bounded residual
- if style ownership is weak/missing, late-stage timbre is further attenuated

### 6.4 Design items that should be treated as frozen for now

Until the mainline is fully stable, the following should remain fixed:

- external default = **single reference**
- canonical reference contract = **`collapsed_reference`**
- timing authority = **content/pitch only**
- `global_timbre_to_pitch = false` by default
- dynamic timbre = **constrained enhancer**, not independent owner

This avoids drifting back into a mixed product/research interface.

## 7. Training policy

### Stage 1 — stabilize identity and naturalness

- low style pressure
- stable speaker anchor
- intelligibility first

### Stage 2 — strengthen style, then material

- raise `M_style`
- open `M_timbre` progressively
- keep anchor preservation and boundary suppression active
- monitor speaker drift and boundary artifacts

## 8. Loss policy (practical)

The current codebase now supports a more explicit hierarchy-aware training view:

```
L =
  L_recon + L_f0 + L_adv
  + λ_id      L_id
  + λ_style   L_style
  + λ_mat     L_material
  + λ_dis     L_disentangle
  + λ_budget  L_timbre_budget
  + λ_stream  L_stream
```

Important practical pieces:

- identity consistency against the target speaker reference
- style-memory consistency
- timbre-anchor cosine / smoothness / anchor preservation
- style↔timbre disentanglement
- **timbre budget**: prevent `M_timbre` from overpowering `M_style`
- streaming consistency as a future first-class objective

## 9. Immediate roadmap

### Already started

- enable constrained dynamic timbre by default
- make `M_timbre` style-conditioned
- attenuate late-stage timbre relative to style
- add timbre regularization warmup
- add a timbre budget penalty
- simplify the inference product surface to a default single-reference path

### Next: current-stage tasks

#### P0 — consolidate the mainline

- keep public inference centered on `src_wav + ref_wav + style_profile`
- continue demoting legacy / factorized tools to research-only status
- keep docs and examples aligned to the same product story

#### P1 — complete the engineering path

- replace prefix-recompute decoder path with true incremental decoder state
- add stateful / chunkwise vocoder synthesis
- strengthen smoke / regression coverage for the single-reference path

#### P2 — complete the evaluation path

- benchmark online/offline parity at overlap boundaries
- monitor speaker drift under stronger style settings
- evaluate material activity separately from mere prosody movement

### Future extensions

Future work is welcome, but should obey the mainline contract:

#### Allowed future expansion directions

- stronger streaming implementation
- better identity/style/material metrics
- stricter online/offline consistency losses
- controlled emotion/accent expansion
- research-only factorized ablations behind separate tooling

#### Directions that should not become the default mainline

- public multi-reference combinatorial control
- letting `M_timbre` compete with `M_style` as a second expression owner
- reintroducing style/timbre timing writeback
- turning the mainline into a general-purpose factor-control playground

## 10. Design intent

> **External single reference. Internal three-role split. Strong decoder-side realization.**

This is not meant to be a public multi-factor control playground.  
It is a **product-oriented strong-style Conan** with:

- stable identity
- stronger expression
- stronger material activity
- controlled streaming-oriented deployment path
