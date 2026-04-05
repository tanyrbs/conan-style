# Conan mainline AutoDL 云端训练交接文档

更新日期：2026-04-05  
编写目的：把当前 Conan mainline 迁移到 AutoDL 云端做正式训练前，给接手同学一份**可直接执行**、**不要误判**的交接手册。  
核查基线：`3e766fa`（在本交接文档编写前的训练修复基线）

---

## 0. 一句话结论

当前主线代码已经满足“**可以进入正式训练前的 AutoDL 预检阶段**”这个条件，但要分清两件事：

1. **代码/数据契约层面**：当前仓库主线是健康的。  
   - `pytest`：`95 passed, 1 skipped`
   - `compileall`：通过
   - 本地 1-step 训练 smoke：通过
   - full binary scan：`code_contract_ready=true`、`data_ready=true`

2. **当前本地 shell 环境本身**：**不满足正式训练环境要求**。  
   当前本地 full prep 结果是：
   - `code_contract_ready=true`
   - `data_ready=true`
   - `data_dependent_preview_ready=true`
   - `environment_ready=false`
   - `ready=false`

原因不是主模型代码坏了，而是**本地运行环境版本不对**：
- Python `3.13`
- `torch/torchaudio 2.7.0+cpu`
- `nltk 3.9.1`
- `g2p_en` / `nltk` 资源在这个环境里还出现了递归导入异常

**因此：上 AutoDL 后第一优先级不是“直接长训”，而是先把环境钉到 canonical 版本，再跑 full prep。**

---

## 1. 当前 shipped 主线到底是什么

这条主线更准确的描述是：

- **owner-first expressive VC**
- **strong prosodic variation**
- **strong style-driven contour control**

不要对外讲成：

- “完整显式 timing planner rewrite”
- “已经做成完全的后节奏重规划系统”

原因：

- `style_to_pitch_residual` 主线已接通
- post-rhythm pitch canvas 相关 runtime key 虽已预留，但 shipped mainline 大多数时候仍是 **source-aligned residual + owner-style 强驱动**
- `factorization_guaranteed` 仍然是 `false`

这不影响训练，但会影响你对实验结果的解释方式。

---

## 2. 本次训练前我已经核对过的关键点

### 2.1 训练正确性下限相关

已经确认未回退：

- 数据集入口的防御式逐帧对齐仍在  
  `tasks/Conan/dataset.py`
- VQ 初始化不会再被 eval 路径提前“吃掉”  
  `modules/Conan/prosody_util.py`
- DDP 下 VQ EMA 统计已做全局 reduce，而不是 rank-local 漂移
- reference mel padding 判定已改为“整帧全 padding 才算 padding”
- internal identity encoder 在辅助 identity loss 时会临时 `eval()`，避免 dropout / BN 抖动
- `global_style_summary_fallback_to_timbre` canonical 默认已是 fail-closed：`false`
- 控制头 final layer 已从 hard-zero 改成 tiny non-zero (`1e-3`)，避免上游长期断梯度

### 2.2 训练前数据验收相关

已经确认：

- `mainline_train_prep.py` 支持：
  - `--binary_scan_items`
  - `--full_binary_scan`
- JSON summary 会输出：
  - `binary_frame_alignment_scan_mode`
  - `binary_frame_alignment_scan_limit`
- fallback valid/test holdout 分组已优先使用解析后的真实 speaker id，而不是脆弱的 path segment heuristic

### 2.3 推理/验证依赖相关

虽然这份文档主要是训练交接，但训练中的 valid/infer 仍会依赖这些组件，所以也确认过：

- built-in `HifiGAN` / `HifiGAN_NSF` 已基本打通 local-hparams-safe 路径
- `legacy_global_hparams_bridge` 对 built-in vocoder 默认已偏向关闭
- `_vocoder_warm_zero()` 不再把 mel bin 写死为 `80`

这意味着：**云端训练时只要 `checkpoints/hifigan_vc` / `checkpoints/Emformer` 等必需资产存在，训练期验证链路不应该再因为全局 hparams 污染而踩老坑。**

---

## 3. 本地核查结果（2026-04-05）

### 3.1 自动化验证

执行结果：

```bash
python -m pytest -q
python -m compileall -q modules tasks inference utils data_gen tests
```

结果：

- `pytest`：`95 passed, 1 skipped`
- `compileall`：通过

### 3.2 full binary prep 结果

执行：

```bash
python tasks/Conan/mainline_train_prep.py \
  --config egs/conan_emformer.yaml \
  --full_binary_scan \
  --output_path smoke_runs/autodl_train_prep_20260405.json
```

当前本地结果摘要：

- `code_contract_ready=true`
- `data_ready=true`
- `data_dependent_preview_ready=true`
- `binary_frame_alignment_scan_mode=full`
- `environment_ready=false`
- `ready=false`

失败项全部属于**环境依赖**，不是主模型代码契约：

- `torch` 版本不匹配：本地 `2.7.0+cpu`，期望 `2.5.1`
- `torchaudio` 版本不匹配：本地 `2.7.0+cpu`，期望 `2.5.1`
- `nltk` 版本不匹配：本地 `3.9.1`，期望 `3.8.1`
- Python 版本不匹配：本地 `3.13`，期望与 pinned `torchaudio 2.5.1` 兼容的 `3.10/3.11` 类环境
- `g2p_en` / `nltk` 资源在本地 base env 下有递归导入异常

### 3.3 1-step 训练 smoke

执行：

```bash
python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanAutoDLHandoverSmoke1 \
  --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=100000,num_sanity_val_steps=0,max_updates=1,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[],dataloader_persistent_workers=False,dataloader_pin_memory=False,tb_log_interval=1"
```

结果：

- 1-step smoke 通过
- 早期诊断值正常打印
- 没有立即出现 NaN / shape contract crash / vocoder 路径硬错误

注意：这个 smoke 是**链路正确性确认**，不是吞吐能力结论。

---

## 4. 上 AutoDL 前必须准备的资产

至少保证以下目录在云端可用：

### 4.1 训练数据

- `data/binary/libritts_single`
- `data/processed/libritts_single`

当前 canonical training config：

- `binary_data_dir: data/binary/libritts_single`
- `processed_data_dir: data/processed/libritts_single`

### 4.2 训练/验证依赖 checkpoint

- `checkpoints/Emformer`
- `checkpoints/hifigan_vc`

### 4.3 代码仓库

- 当前仓库完整工作树
- 推荐直接使用当前主分支最新提交

如果你要**继续训已有 Conan 主模型**，还需要同步已有实验目录，例如：

- `checkpoints/Conan`
- 或你实际的历史 `checkpoints/<exp_name>`

---

## 5. AutoDL 环境建议

## 5.1 版本要求

正式训练环境建议对齐到：

- Python：`3.10.x` 或 `3.11.x`
- `torch==2.5.1`
- `torchaudio==2.5.1`
- `nltk==3.8.1`

不要拿当前本地这个：

- Python `3.13`
- `torch/torchaudio 2.7.0+cpu`

去直接类比 AutoDL 可训练性。

## 5.2 安装原则

最稳的做法：

1. 先选一个 **CUDA 可用**、并且能装/已带 `torch==2.5.1` + `torchaudio==2.5.1` 的 AutoDL 镜像
2. 再补齐其余 Python 依赖
3. 再下载 `nltk` 必需资源
4. 再跑 `mainline_train_prep.py --full_binary_scan`

示例流程（bash）：

```bash
conda create -n conan python=3.10 -y
conda activate conan

# 这里优先保证 torch / torchaudio 是 CUDA 可用且版本对齐 2.5.1
# 具体 CUDA wheel 安装方式按你选用的 AutoDL 镜像来，不要盲目接受错误版本。

pip install -r requirements.txt

python - <<'PY'
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
PY

python - <<'PY'
import torch, torchaudio, nltk
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("nltk:", nltk.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
```

### 5.3 一个重要提醒：不要用 `torchrun`

这个项目当前 trainer 的多卡逻辑是：

- 通过 `CUDA_VISIBLE_DEVICES` 识别可见 GPU
- 当可见 GPU 数量 `> 1` 时，内部 `mp.spawn(...)`
- DDP backend 由 `ddp_backend: auto` 自适应

**因此：不要再额外套一层 `torchrun`。**

正确方式是：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/run.py --config egs/conan_emformer.yaml --exp_name ConanMainlineAutoDL
```

而不是：

```bash
# 不推荐
torchrun ...
```

---

## 6. AutoDL 上线前的标准预检流程

按顺序做，不要跳。

### Step 1：确认目录

```bash
ls data/binary/libritts_single
ls data/processed/libritts_single
ls checkpoints/Emformer
ls checkpoints/hifigan_vc
```

### Step 2：full prep

```bash
python tasks/Conan/mainline_train_prep.py \
  --config egs/conan_emformer.yaml \
  --full_binary_scan \
  --output_path smoke_runs/autodl_mainline_train_prep.json
```

**必须同时满足：**

- `code_contract_ready=true`
- `environment_ready=true`
- `data_ready=true`
- `train_ready_now=true`

只要 `train_ready_now` 不是 `true`，就**不要开始长训**。

### Step 3：1-step smoke

先单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanAutoDLSmoke1 \
  --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=100000,num_sanity_val_steps=0,max_updates=1,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[],dataloader_persistent_workers=False,dataloader_pin_memory=False,tb_log_interval=1"
```

检查：

- 能正常起模型、起 dataloader、跑前向、回传、优化一步
- 没有 NaN / CUDA OOM / checkpoint 路径错误 / vocoder 路径错误

### Step 4：8-step smoke（可选但推荐）

如果 1-step 正常，建议再做一个短一点的多步 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanAutoDLSmoke8 \
  --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=1000000,num_sanity_val_steps=0,max_updates=8,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[],dataloader_persistent_workers=False,dataloader_pin_memory=False,tb_log_interval=1"
```

如果要测试多卡初始化，再单独做一个小步数 DDP smoke。

---

## 7. 正式训练命令

## 7.1 单卡

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanMainlineAutoDL
```

## 7.2 单机多卡

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanMainlineAutoDL \
  --hparams "ddp_master_port=29541"
```

说明：

- `ddp_backend` 现在是 `auto`
- 在 Linux + CUDA 的 AutoDL 环境里，会走 `nccl`
- 如果同机上还有别的分布式任务，记得改 `ddp_master_port`

## 7.3 从已有 checkpoint warm start

如果要用已有 Conan checkpoint 继续开新实验：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanMainlineAutoDLWarm \
  --hparams "load_ckpt=checkpoints/Conan/model_ckpt_steps_200000.ckpt,ddp_master_port=29541"
```

## 7.4 中断后继续训练

如果是**同一个 `exp_name`** 的实验目录，重新执行同一条命令即可。  
trainer 会自动从该 `work_dir` 的最新 checkpoint 恢复，不需要额外包一层 `torchrun`。

---

## 8. 训练中最需要关注的事项

下面这些是**最关键的 watch list**。

## 8.1 第一优先级：不是 loss 漂亮，而是“训练对象没脏”

### A. prep 是否真绿

最关键的是：

- `train_ready_now=true`

不是只看：

- `code_contract_ready=true`

如果只是 code contract 绿，而 environment/data 不绿，不要开始长训。

### B. 有没有出现 frame trim 告警

`ConanDataset` 现在会在运行时防御式 trim，但这不是让你放心忽略数据问题。

如果你怀疑云端同步了旧 sidecar，可以在 smoke 阶段临时加：

```bash
--hparams "dataset_warn_on_frame_trim=true,dataset_warn_on_frame_trim_max_events=16"
```

一旦出现 trim warning：

- 先停下来
- 检查 binary sidecar 是否混旧缓存
- 必要时重新 full prep / 重生对应二进制

### C. `global_style_summary_fallback` 不应该乱跳

canonical contract 是 fail-closed：

- `global_style_summary_fallback_to_timbre: false`

所以训练日志里如果大量出现：

- `diag_global_style_summary_fallback > 0`

说明 reference contract / cache /上游 summary 提供链路有问题，需要排查。

## 8.2 早期很多控制诊断是“关着的”，这通常是正常现象

不要把以下现象误判成 bug：

- `diag_active_mainline_control_loss_count = 0`
- `diag_lambda_output_identity_cosine = 0`
- `diag_lambda_pitch_residual_safe = 0`
- `diag_lambda_style_success_rank = 0`

因为 canonical schedule 本来就是分阶段开的：

- `lambda_dynamic_timbre_budget`：从 step `0` 开始 warmup
- `lambda_output_identity_cosine`：step `2000` 开
- `lambda_decoder_late_owner`：step `2000` 开
- `lambda_pitch_residual_safe`：step `4000` 开
- `lambda_style_success_rank`：step `20000` 开

所以**前几千步控制分支比较“保守”是预期行为**。

## 8.3 重点看哪些监控

### 基础收敛面

- `l1`
- `ssim`
- `fdiff`
- `uv`
- `val/total_loss`

### 身份/音色稳定面

- `diag_output_identity_ref_cos`
- `diag_output_identity_target_distance`
- `diag_identity_encoder_frozen_for_loss`

### 动态音色/预算约束

- `diag_dynamic_timbre_gate_mean`
- `diag_dynamic_timbre_gate_std`
- `diag_runtime_dynamic_timbre_style_budget_clip_frac`
- `diag_runtime_dynamic_timbre_style_budget_overflow_mean`
- `diag_decoder_stage_dynamic_timbre_budget_overflow_mean`

### 风格成功监督是否真正起效

- `diag_style_success_rank_term_active`
- `diag_style_success_negative_source_label`
- `diag_style_success_negative_source_proxy`
- `diag_style_success_proxy_informative_feature_count`

注意：

- 如果当前 staged artifacts 没有有效弱标签桶，主线语义更接近 `paired_only`
- 小 batch 下出现 proxy fallback 不奇怪
- 真正要担心的是：support 一直不够，rank term 永久起不来

## 8.4 哪些值“看起来奇怪”但不算 bug

- `diag_factorization_guaranteed = 0`  
  这是设计如此，不是 bug。

- early stage `style_success_rank_term_active = 0`  
  只要还没到 schedule 开启点，或者 batch 支持不够，这都可能是正常的。

- `post-rhythm` 相关能力没有在训练日志里表现成“完整 timing rewrite”  
  这和 shipped contract 一致，不是回退。

---

## 9. 如果 AutoDL 上出问题，优先这样排查

## 9.1 prep 不通过

先看失败项属于哪类：

- `runtime_dependencies`
- `data_staging`
- `mainline_contract`

通常优先级是：

1. 版本不对
2. `g2p_en` / `nltk` 资源没装好
3. 数据目录/ckpt 目录没同步
4. 少量 stale binary sidecar

## 9.2 一上来就 OOM

先按这个顺序缩：

1. `max_tokens`
2. `max_sentences`
3. `ds_workers`

尽量不要第一反应就改主模型结构或控制面配置。

## 9.3 多卡起不来

先检查：

- `CUDA_VISIBLE_DEVICES`
- `ddp_master_port`
- 是否误用了 `torchrun`

这套 trainer 的正确使用方式是：

- 只设置 `CUDA_VISIBLE_DEVICES`
- 直接 `python tasks/run.py ...`

## 9.4 训练不炸，但 loss 语义看起来发虚

先查：

- 有没有 frame trim warning
- `binary_frame_alignment_scan_mode` 是否真的是 `full`
- 当前二进制是不是老缓存 + 新 sidecar 混合
- same-speaker reference sidecar 是否被污染

不要只看“程序没报错”。

---

## 10. 非 blocker，但要知道的剩余边界

这些不影响现在开始训练，但要心里有数：

1. **流式推理吞吐上限仍受 prefix-recompute 限制**  
   这是推理架构边界，不是训练 blocker。

2. **`lambda_style_timbre_runtime_overlap` 默认仍是 0.0**  
   canonical shipping 路线偏稳，不是冲论文上限配置。

3. **built-in vocoder local-hparams-safe 已基本打通，但这不是训练主 blocker**  
   它主要影响验证/推理路径隔离性，不是主训练对象正确性。

4. **主线依然不是“严格可解释解耦证明”**  
   `factorization_guaranteed=false` 是诚实的。

---

## 11. 推荐的 AutoDL 实际执行顺序

建议按下面顺序来，不要省步骤：

1. 同步仓库、数据、Emformer、HifiGAN checkpoint
2. 创建 canonical env（Python 3.10/3.11 + torch/torchaudio 2.5.1 + nltk 3.8.1）
3. 下载 `nltk` 必需资源
4. 跑：
   - `mainline_train_prep.py --full_binary_scan`
5. 结果必须 `train_ready_now=true`
6. 跑 1-step smoke
7. 跑 8-step smoke（推荐）
8. 再开正式训练
9. 正式训练头几千步密切盯：
   - OOM
   - trim warning
   - NaN
   - style-success support 是否长期起不来
10. 保留 `terminal_logs/` + `tb_logs/` + prep JSON，作为第一天训练审计材料

---

## 12. 建议保留的命令备忘

### full prep

```bash
python tasks/Conan/mainline_train_prep.py \
  --config egs/conan_emformer.yaml \
  --full_binary_scan \
  --output_path smoke_runs/autodl_mainline_train_prep.json
```

### 1-step smoke

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanAutoDLSmoke1 \
  --hparams "ds_workers=0,max_sentences=1,max_tokens=3000,val_check_interval=100000,num_sanity_val_steps=0,max_updates=1,eval_max_batches=1,num_ckpt_keep=1,save_best=False,save_codes=[],dataloader_persistent_workers=False,dataloader_pin_memory=False,tb_log_interval=1"
```

### 正式单卡

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanMainlineAutoDL
```

### 正式多卡

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/run.py \
  --config egs/conan_emformer.yaml \
  --exp_name ConanMainlineAutoDL \
  --hparams "ddp_master_port=29541"
```

### TensorBoard

```bash
tensorboard --logdir checkpoints/ConanMainlineAutoDL/tb_logs --host 0.0.0.0 --port 6006
```

---

## 13. 最后一句提醒

这次进入 AutoDL 长训前，**最该坚持的原则不是“尽快开跑”，而是“先证明训练对象没脏，再让它跑很久”**。

也就是说：

- 先 full prep
- 再 smoke
- 再长训

不要把“程序能启动”误判成“已经可以放心训 20 万步”。
