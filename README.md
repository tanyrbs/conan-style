# Conan (Single-Reference Strong Style + Material VC)

本仓库当前主线是：

> **单参考输入 → global timbre anchor 稳身份 → M_style 主表达 → M_timbre 做材质增强 → decoder-side 合成**

## 产品面目标

默认外部接口收敛为：

- `src_wav`
- `ref_wav`
- `style_strength`
- 可选 `style_profile`

`ref_timbre/ref_style/ref_dynamic_timbre` 这类拆分参考仍可用于研究/ablation，  
但**不再是默认产品接口**。

## 内部角色分工

- **global_timbre_anchor / z_id**：身份 owner，负责 speaker identity 稳定
- **global_style_summary + M_style**：表达 owner，负责“怎么说”
- **M_timbre**：受约束的材质残差，负责 breathiness / brightness / thickness / onset-release 等局部声质活性

关键原则：

- **decoder-side only**
- **禁止 style/timbre 写回 timing**
- **`global_timbre_to_pitch` 默认关闭**
- **M_style 的优先级高于 M_timbre**

## 当前实现状态

已经落地：

- 单参考 `collapsed_reference` 主线
- `global_timbre_anchor / global_style_summary / M_style / M_timbre / decoder_style_bundle`
- decoder-side style adapter
- dynamic timbre 默认重新启用
- M_timbre 已开始受 style 上下文约束
- timbre 正则与 warmup 已纳入配置主线

仍待继续优化：

- **严格增量 streaming** 还没完全做完  
  当前只有 **Emformer** 是严格 stateful；主模型仍是**前缀重算式 online**，vocoder 已改成**有限左上下文重合成**，但还不是真 stateful vocoder。
- pre-decoder 查询路径里仍保留少量 anchor/query 注入，尚未完全收敛成“纯 decoder-only 语义面”

## 现阶段任务

当前阶段不是继续加公开控制面，而是**收口主线**：

- **P0：主线收口**
  - 默认外部接口固定为单参考
  - `M_style > M_timbre > anchor` 的 owner 层级稳定
  - 保持 `decoder-side only` 与 `no timing writeback`
- **P1：工程补强**
  - 把 online 路径从前缀重算推进到真正增量
  - 补齐 smoke / inference / profile sweep 的主线路径验证
  - 继续清理 legacy / factorized 用户表面
- **P2：评测闭环**
  - 统一看 speaker 保真、风格强度、材质活性、边界稳定性
  - 建立 online/offline parity 指标

一句话：**现在先把系统做“对、稳、可持续扩展”，而不是把控制面做“多”。**

## 当前设计冻结项

当前建议视为冻结的设计约束：

- 外部默认 **单参考**
- 内部默认 **三角色**
  - `global_timbre_anchor`：身份 owner
  - `M_style`：表达 owner
  - `M_timbre`：材质 residual
- `global_timbre_to_pitch: false`
- `timing_writeback_allowed: false`
- dynamic timbre 的定位是 **style realization enhancer**

以下方向暂不作为主线：

- 公开多参考拼装控制
- 把 dynamic timbre 做成独立产品旋钮体系
- 把 style/timbre 回写到 timing / pitch authority
- 重新扩展成“多头标签控制平台”

## 未来拓展

未来扩展分三层：

### 1. 近期扩展

- true incremental decoder cache
- stateful vocoder
- 更强的 timbre budget / gate curriculum
- 更明确的 streaming 诊断输出

### 2. 中期扩展

- 更强的 identity / style / material 评测闭环
- 更稳的 boundary-aware material control
- 更明确的 online/offline consistency loss
- 推理侧主线与 research-only 工具彻底分层

### 3. 远期扩展

- 在**不破坏单参考产品面**的前提下，保留内部 factorized ablation 能力
- 更强的 deployment-oriented streaming stack
- 受约束的情绪 / accent 扩展，但不改变主线 owner 关系

## Preset

- `strong_style`（默认）
- `extreme`

## Gate Warmup

- `decoder_style_adapter_gate_bias_start: -1.0`
- `decoder_style_adapter_gate_bias_end: 0.0`
- `decoder_style_adapter_gate_bias_warmup: 80000`

## 文档

- 核心说明：`docs/strong_style_mainline_upgrade_20260331.md`
- 当前阶段 / 设计 / 路线图：`docs/current_phase_design_and_roadmap_20260401.md`
- inference 入口说明：`inference/README.md`
