# Conan (Single-Reference Strong-Style Mainline)

当前仓库主线已经统一为：

> **单参考输入 → global_timbre_anchor 稳身份 → M_style 主表达 → M_timbre 做材质增强 → decoder-side 合成**

## Canonical Conan mainline

唯一 canonical 推理配置：

- `egs/conan_mainline_infer.yaml`

唯一 canonical checkpoint 入口：

- `checkpoints/Conan/config.yaml`
- `--exp_name Conan`

当前默认产品面：

- `src_wav`
- `ref_wav`
- optional `style_profile`
- optional `style_strength`

split reference / factorized swap 仍保留给研究与 ablation，
但不再代表 Conan 主线产品接口。

## 直接可跑的 Conan mainline 入口

### 1) 单参考 strong-style demo

先编辑：

- `inference/conan_single_reference_demo.example.json`

再运行：

```bash
python inference/run_voice_conversion.py
```

### 2) Style profile sweep demo

先编辑：

- `inference/conan_style_profile_sweep.example.json`

再运行：

```bash
python inference/run_style_profile_sweep.py
python inference/run_style_profile_evaluation.py --sweep_dir infer_out_profiles/conan_mainline_demo
python inference/run_style_profile_report.py --sweep_dir infer_out_profiles/conan_mainline_demo
```

### 3) Conan Gradio demo

```bash
python inference/run_conan_gradio_demo.py
```

对应设置文件：

- `inference/conan_gradio/gradio_settings.yaml`

默认 runner 已经绑定：

- `--config egs/conan_mainline_infer.yaml`
- `--exp_name Conan`

## 当前冻结设计

- 外部默认单参考
- `reference_contract_mode: collapsed_reference`
- `global_timbre_to_pitch: false`
- `timing_writeback_allowed: false`
- `M_style` 是 expression owner
- `M_timbre` 是 bounded material enhancer，而不是第二个主风格 owner

## 当前阶段重点

1. 统一 Conan mainline config / runner / README / docs / checkpoint config
2. 把 Conan 单参考 strong-style demo 单独立起来
3. 把旧 NATSpeech/PortaSpeech surface 明确剥离为 legacy / non-Conan
4. 继续推进真正增量的 decoder / vocoder streaming

## 文档

- 主线升级说明：`docs/strong_style_mainline_upgrade_20260331.md`
- 当前阶段 / 设计 / 路线图：`docs/current_phase_design_and_roadmap_20260401.md`
- 推理入口说明：`inference/README.md`

## Legacy / non-Conan surface

以下内容不再代表 Conan 主线：

- `inference/tts/gradio/*`（旧 NATSpeech/PortaSpeech TTS demo）
- `inference/run_voice_conversion_nvae.py`
- `inference/factorized_swap_builder.py`
- `inference/run_style_profile_evaluation.py --enable_factorized_report`
