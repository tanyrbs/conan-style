# Conan 当前阶段、设计约束与路线图

Updated: 2026-04-01

## 1. 当前阶段是什么

当前不是继续扩公开控制面的阶段，而是 **mainline consolidation** 阶段。

核心目标只有一条：

> **单参考输入 → 稳身份 → 强表达 → 强材质 → decoder-side 合成**

因此当前阶段优先级是：

1. 统一 canonical inference config / runner / checkpoint config / README / docs
2. 收紧角色边界：`global_timbre_anchor` / `M_style` / `M_timbre`
3. 保持单参考产品面
4. 继续推进 streaming 工程正确性

## 2. 当前 canonical 口径

### 唯一 canonical inference config

- `egs/conan_mainline_infer.yaml`

### 唯一 canonical checkpoint 入口

- `checkpoints/Conan/config.yaml`
- `--exp_name Conan`

### Conan 主线 demo

- 单参考 batch/demo：`inference/run_voice_conversion.py`
- style profile sweep：`inference/run_style_profile_sweep.py`
- report/eval：`inference/run_style_profile_evaluation.py` / `inference/run_style_profile_report.py`
- Gradio：`inference/conan_gradio/app.py`

## 3. 当前设计冻结项

### 外部产品面

默认只保留：

- `src_wav`
- `ref_wav`
- optional `style_profile`
- optional `style_strength`

### 内部 owner 层级

- `global_timbre_anchor`：identity owner
- `M_style`：expression owner
- `M_timbre`：bounded material enhancer / residual

### 不变约束

- `reference_contract_mode: collapsed_reference`
- `global_timbre_to_pitch: false`
- `timing_writeback_allowed: false`
- `dynamic_timbre` 的角色是 style realization enhancer，不是独立主控制通道

## 4. 本阶段已经落实的事

- canonical inference config 已单独落到 `egs/conan_mainline_infer.yaml`
- `checkpoints/Conan/config.yaml` 已改成指向 canonical config 的轻量 overlay
- runner 默认口径已统一到 canonical config + canonical checkpoint
- Conan 单参考 demo 已单独立起
- Conan Gradio demo 已单独立起
- 旧 NATSpeech/PortaSpeech Gradio surface 已明确降级为 legacy / non-Conan
- factorized report 已改为显式 `--enable_factorized_report` 才生成
- `set_hparams()` 已允许 config 中的 `work_dir` 在无 `exp_name` 时保留，不再被清空
- style / timbre query 已拆开，不再共同吃 `content + condition + anchor`
- dynamic timbre 的 style conditioning 已改成 owner-aware：`LayerNorm + stopgrad`
- control loss 已补成更接近 owner/stage 的版本：
  - local mask-aware dynamic timbre budget
  - boundary penalty
  - late-stage owner / anchor budget
- 输出端已补一层 mel-side identity proxy：
  - `mel_out -> encode_spk_embed -> global_timbre_anchor.detach()`
- test / smoke 已开始直接覆盖 prefix-online 路径：
  - `tasks/Conan/streaming_prefix_online_smoke.py`
  - `tasks/Conan/style_mainline_smoke.py`
  - `inference/streaming_parity_smoke.py`

## 5. 当前还没彻底做完的工程项

### P0

- 真正 stateful decoder cache（当前仍是 prefix-online acoustic recompute）
- 真正 stateful vocoder（当前仍是 bounded left-context re-synthesis）
- 冻结外部 speaker verifier 版本的更强 identity loss / speaker drift 评测闭环

### P1

- 把 inference-side online/offline parity 持续接入回归门槛
- 更细的 voiced / silence / chunk-boundary diagnostics

### P2

- 更完整的 identity / style / material 评测闭环
- 更明确的 streaming diagnostics

## 6. 当前实现细节（owner hierarchy / streaming）

### Query / owner hierarchy

- `style_query = LN(content + condition + small global_style_summary prior)`
- `timbre_query = LN(content + condition)`
- `global_timbre_anchor` 不再作为 style/timbre query shared base 的一部分
- `global_timbre_anchor` 继续只服务于：
  - identity anchor
  - anchor preserve / recenter
  - 可选 pitch prior（mainline 默认关）

### Dynamic timbre owner policy

- `dynamic_timbre_style_context = LN(M_style_fast + 0.5 * M_style_slow + coarse_style_prior)`
- 默认 `stopgrad`
- 语义上是：
  - `M_style` owns
  - `M_timbre` follows and enhances

### Streaming 语义

当前 streaming 需要明确理解为：

- reference cache：一次构建
- content / Emformer：stateful
- acoustic model：prefix recompute
- vocoder：bounded left-context / whole-mel synthesis

这意味着现在已经把 **online path 变成了 first-class validation target**，
但还没有完成 strict incremental decoder / vocoder。

## 7. 未来拓展边界

允许的未来拓展：

- 更强的 streaming stack
- 更强的 identity/style/material 指标
- 更受约束的 emotion/accent 扩展
- research-only factorized ablation

不应回到主线默认面的方向：

- 多参考公开拼装控制
- dynamic timbre 单飞
- style/timbre 写回 timing 或 pitch authority
- 让 research surface 反向定义产品面

## 8. 一句话路线图

> **先把 Conan 单参考 strong-style 主线做稳、做清楚、做可跑；再把扩展做成受约束的旁路。**
