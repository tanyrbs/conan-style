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

补充说明：

- Conan mainline 只使用 `reference_contract_mode: collapsed_reference`
- 当前内部的 style/timbre 三角色分工是 **single-reference weak internal factorization**
- `reference_contract.factorization_guaranteed = false` 是主线设计事实，不是 bug
- canonical mainline 训练默认只保留 **4 个控制正则**：
  - `lambda_output_identity_cosine`
  - `lambda_dynamic_timbre_budget`
  - `lambda_dynamic_timbre_boundary`
  - `lambda_decoder_late_owner`
- 这 4 个只约束 control regularization；总训练 loss 仍包含 mel / pitch / VQ 等主干项

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

如果本地没装 Gradio：

```bash
pip install gradio
```

默认 runner 已经绑定：

- `--config egs/conan_mainline_infer.yaml`
- `--exp_name Conan`

## Canonical training / dry run / eval

唯一 canonical 训练配置：

- `egs/conan_emformer.yaml`

训练前检查：

```bash
python tasks/Conan/mainline_train_prep.py --config egs/conan_emformer.yaml
```

本地 CPU 最小 dry run：

```bash
python tasks/Conan/mainline_cpu_dry_run.py ^
  --config egs/conan_emformer.yaml ^
  --binary_data_dir data/binary/libritts_single_smoke ^
  --num_steps 2
```

正式训练命令模板：

```bash
python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineTrain
```

训练后固定回归入口：

```bash
python tasks/Conan/style_mainline_smoke.py --config egs/conan_mainline_infer.yaml
python tasks/Conan/streaming_prefix_online_smoke.py --config egs/conan_mainline_infer.yaml
python tasks/Conan/decoder_style_adapter_contract_smoke.py
```

补充说明：

- `Conan` 保留给仓库内 canonical inference checkpoint 入口
- 真正训练请使用新的 `--exp_name`
- 当前 online 口径仍是 **prefix-online parity target**，不是 fully stateful decoder/vocoder

## 当前冻结设计

- 外部默认单参考
- `reference_contract_mode: collapsed_reference`
- `global_timbre_to_pitch: false`
- `timing_writeback_allowed: false`
- `M_style` 是 expression owner
- `M_timbre` 是 bounded material enhancer，而不是第二个主风格 owner
- `mainline_owner` metadata 只是主线层级标记；真正的 owner hierarchy 以
  `global_timbre_anchor / M_style / M_timbre` 为准
- style / timbre query 已拆分：不再共享 `content + condition + anchor`
- `M_style -> M_timbre` 已改成 owner-aware conditioning：`LayerNorm + stopgrad`
- decoder adapter 对 zero tensor / 无效分支已做 hard no-op，避免“禁用分支仍偷偷注入”
- decoder style bundle 会先按 `effective_signal_epsilon` 过滤近零分支，再进入 adapter owner 判定
- late-stage 默认会在 local style owner 已存在时跳过 `global_style_summary` 重复注入
- dynamic timbre 已增加 runtime hard budget，确保 `M_timbre` 受 `M_style` 预算约束
- canonical `mainline_minimal` 已收缩成 4-loss control pack：
  - identity cosine
  - dynamic timbre budget
  - dynamic timbre boundary
  - decoder late-owner
- 输出端已补 mel-side identity proxy loss：`mel_out -> global_timbre_anchor`
- validation / test 已直接覆盖 prefix-online path，并回传 offline/online mel parity
- mainline style profile 若收到 research 风格 override，默认会告警并收回 canonical mainline；研究态请显式使用 `research_*` 或 opt-in

## 当前阶段重点

1. 继续推进真正增量的 decoder / vocoder streaming
2. 把 online/offline parity 变成固定 smoke / regression 入口
3. 继续观察强风格下 speaker drift / intelligibility / boundary stability
4. 后续再升级到冻结外部 speaker verifier 的更强 identity loss

## 新增 smoke / regression

```bash
python tasks/Conan/style_mainline_smoke.py
python tasks/Conan/streaming_prefix_online_smoke.py
python tasks/Conan/decoder_style_adapter_contract_smoke.py
```

其中：

- `style_mainline_smoke.py`：检查单参考 mainline contract / query split / decoder-side bundle，以及 disabled branches 不得泄漏到 decoder bundle
- `streaming_prefix_online_smoke.py`：检查 Conan 任务侧 offline mel vs prefix-online chunked mel
- `decoder_style_adapter_contract_smoke.py`：检查近零 style/timbre 分支 hard no-op 与 late-stage owner/fallback 语义

用于防止“只在离线路径稳定、在线路径没被回归”的问题。

如果你有真实 `src_wav / ref_wav`，还可以直接检查推理前端的 online/offline parity：

```bash
python inference/streaming_parity_smoke.py \
  --exp_name Conan \
  --src_wav <src.wav> \
  --ref_wav <ref.wav>
```

## 文档

- 主线升级说明：`docs/strong_style_mainline_upgrade_20260331.md`
- 当前阶段 / 设计 / 路线图：`docs/current_phase_design_and_roadmap_20260401.md`
- canonical training / dry run / eval：`docs/canonical_training_mainline_20260401.md`
- 推理入口说明：`inference/README.md`

## Legacy / non-Conan surface

以下内容不再代表 Conan 主线：

- `inference/tts/gradio/*`（旧 NATSpeech/PortaSpeech TTS demo）
- `inference/run_voice_conversion_nvae.py`
- `inference/research/*`
- `inference/run_style_profile_evaluation.py --enable_factorized_report`
