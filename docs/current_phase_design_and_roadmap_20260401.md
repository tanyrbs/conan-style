# Conan 当前阶段、设计约束与未来路线

Updated: 2026-04-01

## 1. 当前阶段是什么

当前不是“继续加控制能力”的阶段，而是**主线收口阶段**。

核心目标只有一句话：

> **把系统稳定收成：单参考输入 → 稳身份 → 强表达 → 强材质 → decoder-side 合成**

因此当前阶段最重要的不是：

- 做更多公开控制维度
- 做更多多参考产品表面
- 做更多标签化玩法

而是：

- 统一主线
- 收紧角色边界
- 稳定训练与推理
- 给未来扩展留下干净接口

## 2. 当前阶段任务

### P0：系统收口

必须优先完成：

- 默认产品面固定为单参考
- `global_timbre_anchor / M_style / M_timbre` 三角色关系固定
- `decoder-side only` 继续保持
- `timing_writeback_allowed = false`
- `global_timbre_to_pitch = false`

### P1：训练与推理稳定化

当前重点：

- dynamic timbre 继续作为受约束增强器，而不是独立 owner
- 保持 `M_style > M_timbre`
- 持续监控 speaker drift、清晰度、边界脏化
- 补齐 smoke / mainline regression / inference sanity

### P2：工程闭环

当前还没彻底完成：

- 真正增量 decoder state cache
- 真正 stateful vocoder
- online/offline parity 指标
- 更明确的 streaming 诊断输出

## 3. 当前设计

## 3.1 外部设计

默认主线输入应理解为：

- `src_wav`
- `ref_wav`
- `style_strength`
- optional `style_profile`

研究态 split reference 可以保留，但：

- 不应作为默认产品入口
- 不应主导 README / example / runner 的叙事
- 不应反向影响主线架构判断

## 3.2 内部设计

### `global_timbre_anchor`

职责：

- 稳住说话人 identity
- 约束材质分支围绕 anchor 摆动

不负责：

- 主表达
- timing authority

### `M_style`

职责：

- 主表达
- 情绪张力
- accent posture / phrase delivery
- 风格强度

它是**expression owner**。

### `M_timbre`

职责：

- 材质增强
- 声质活性
- breathiness / brightness / thickness / onset-release

它是**material residual**，不是第二个 style owner。

## 3.3 decoder-side 设计原则

当前主线必须坚持：

- style/timbre 只影响 realization
- 不回写 timing
- 不夺取 pitch authority
- late stage 中 `M_style` 优先级高于 `M_timbre`

## 4. 当前阶段不做什么

以下内容可以研究，但不应进入默认主线：

- 多参考组合控制产品面
- dynamic timbre 单飞
- style / timbre 回写 timing
- 把主线重新做成多标签控制面板
- 让 factorized ablation 反向定义产品接口

## 5. 当前完成定义

一个“当前阶段完成”的版本，至少要满足：

### 训练层

- 单参考训练主线稳定
- `M_style` 对风格强度确实有效
- `M_timbre` 提升材质活性但不明显拖走 identity

### 推理层

- 默认 runner / example 以单参考为中心
- 配置与 docs 讲的是同一套故事
- legacy / research-only 工具不再混入主线路径

### 工程层

- smoke 能跑
- mainline regression 能覆盖关键模式
- streaming 实现状态有明确说明

## 6. 未来拓展怎么做

## 6.1 近期拓展

优先顺序：

1. incremental decoder cache
2. stateful vocoder
3. streaming consistency
4. 更完整的 mainline validation

## 6.2 中期拓展

- 更强的 identity/style/material 三指标评测
- 更稳的 boundary-aware material control
- 更清晰的 research-only 目录与脚本分层
- 更明确的 online deployment profiling

## 6.3 远期拓展

可以保留但必须受约束：

- emotion/accent 扩展
- factorized internal ablation
- 更强的 style profile 家族

前提是：

> **不破坏单参考产品面，不破坏 owner 层级，不破坏 decoder-side only 原则。**

## 7. 一句话路线图

当前路线不是“把系统做得更花”，而是：

> **先把主线做窄、做稳、做强，再把扩展做成受约束的旁路。**
