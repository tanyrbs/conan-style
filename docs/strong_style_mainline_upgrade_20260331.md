# Conan Strong-Style Mainline

Updated: 2026-04-01

## 1. System target

当前 Conan 主线收敛为：

> **single reference → global_timbre_anchor → M_style → M_timbre → decoder-side fusion**

外部默认不是多参考分解控制；
内部仍然保留三角色分工。

需要明确的是：

- Conan mainline contract 只有 `collapsed_reference`
- 当前内部分工属于 **single-reference weak internal factorization**
- `reference_contract.factorization_guaranteed = false` 是设计语义，不是未完成状态

## 2. Canonical inference path

唯一 canonical inference config：

- `egs/conan_mainline_infer.yaml`

唯一 canonical checkpoint 入口：

- `checkpoints/Conan/config.yaml`
- `--exp_name Conan`

默认 runners：

- `python inference/run_voice_conversion.py`
- `python inference/run_style_profile_sweep.py`
- `python inference/run_style_profile_evaluation.py --sweep_dir ...`
- `python inference/run_style_profile_report.py --sweep_dir ...`
- `python inference/conan_gradio/app.py`

## 3. External contract

默认 public surface：

- `src_wav`
- `ref_wav`
- optional `style_profile`
- optional `style_strength`

split reference 仍可保留给 research-only tooling，
但不再是 Conan 主线入口。

## 4. Internal owner hierarchy

### `global_timbre_anchor`

- speaker identity owner
- 稳住身份
- 不抢 pitch/timing authority

### `M_style`

- expression owner
- 风格强度、accent posture、情绪张力、局部 delivery
- 内部 dual-scale（fast + slow）但对外仍是单一 owner

### `M_timbre`

- bounded material track around the identity anchor
- breathiness / brightness / thickness / resonance / onset-release
- 在材质子空间内局部主导，但不成为第二个全局 owner
- TVT material track：content-synchronous prior + slerp + material router

说明：

- `mainline_owner` metadata 只是层级标签
- 真正 owner 关系以 `global_timbre_anchor / M_style / M_timbre` 为准

## 5. Current implementation policy

当前必须继续保持：

- decoder-side only 主线语义
- no timing writeback
- `global_timbre_to_pitch = false`
- `M_style > M_timbre`
- dynamic timbre = bounded material track, not a second speaker stream
- `style_trace_mode: dual` + `style_router_enabled`
- `style_to_pitch_residual: true`（bounded, voiced-only）
- `style_to_pitch_residual_mode: auto`（style 只输出 pitch intent；若有 rhythm frame plan 则落在 post-rhythm canvas）

## 6. What changed in this round

这次收口进一步落实了：

- canonical inference config 单独固化
- checkpoint config 改成轻量 overlay
- runners 默认口径统一
- Conan 单参考 demo 独立
- Conan Gradio demo 独立
- old NATSpeech/PortaSpeech surface 明确标成 legacy
- factorized report 改成显式开启
- inference metadata 在 mainline batch runner 中落盘
- decoder adapter 对 zero / 无效分支已做 hard no-op
- local style owner 存在时，late-stage 默认跳过 `global_style_summary` 重复注入
- dynamic timbre 已增加 runtime hard budget
- style 内部 dual-scale router 已上线（对外仍是单 owner）
- timbre 已升级为 TVT material track（global memory + content-synchronous prior + slerp + router）
- bounded style-to-pitch residual 已加入（语义为 pitch intent → canvas；不打开 global timbre to pitch）

## 7. Remaining gaps

最重要的剩余问题仍是工程侧：

1. Emformer 已 stateful，但 acoustic decoder 仍是 prefix recompute
2. vocoder 仍是 bounded-left-context re-synthesis，不是真正 stateful vocoder
3. online/offline parity 还需要更系统的验证

## 8. Design intent

> **External single reference. Internal three-role split. Strong decoder-side realization.**

主线目标不是控制面越来越花，
而是让 Conan 的单参考 strong-style 路线更稳、更清楚、更接近可部署形态。
