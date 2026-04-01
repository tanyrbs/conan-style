# Conan (Single-Reference Strong Style VC)

本仓库已收敛为**单参考强迁移**版本：  
**外部只输入一条参考音频 + 一个 style_strength 旋钮**，内部做最小必要分工：

- **z_spk**：全局音色锚点（speaker identity）
- **g_style**：句级风格摘要
- **M_style**：局部风格记忆（decoder-side 强注入）

目标是“听感优先”的强风格 + 强音色，而不是多因子可组合控制。  
情绪/口音等条件通道默认关闭（如需再开启）。

`style_id` 监督默认关闭（只保留单旋钮）。

## 文档

- **核心说明**：`docs/strong_style_mainline_upgrade_20260331.md`

> 旧文档已清理，仅保留上述新文档。

## Preset

默认只保留强风格 preset：

- `strong_style`（默认）
- `extreme`

## Gate Warmup

默认启用 style gate warmup（前稳后强）：

- `decoder_style_adapter_gate_bias_start: -1.0`
- `decoder_style_adapter_gate_bias_end: 0.0`
- `decoder_style_adapter_gate_bias_warmup: 80000`

## 配置清理

单参考版已移除：
- dynamic timbre 训练相关项
- emotion/accent 控制头
- style_id 监督
