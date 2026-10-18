<div align="center">

# FlashLite

**从零手写的小型 LLM 训练框架：Tokenizer → Pretrain → SFT → DPO → GRPO 全链路**

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Transformers](https://img.shields.io/badge/Transformers-5.x-yellow)

</div>

---

# 📌 项目介绍

FlashLite 是一个个人开发的轻量级 LLM 训练框架，目标是在**不依赖高层训练封装**（HF `Trainer` / `trl` / `peft` 等）的前提下，从零实现现代小模型训练的完整闭环：

- **模型结构从零手写**：GQA 分组注意力（含 Q/K RMSNorm）、Sparse MoE（Top-K 路由 + 负载均衡辅助损失）、RMSNorm、SwiGLU、RoPE（支持 `default / linear / llama3` 三种缩放初始化），继承 HF `PreTrainedModel` / `GenerationMixin`，可直接用 `DynamicCache` 做 KV-Cache 推理；
- **训练算法从零手写**：预训练 / SFT / DPO / GRPO（RLAIF）全部为原生 PyTorch 训练循环，DPO 损失、GRPO 的组内相对优势、PPO-Clip、token 级 KL（k3 估计）均为手写实现；
- **工程化训练栈**：DDP 多卡（`mp.spawn`，Windows 下自动切换 gloo 后端）、AMP `bf16` + `GradScaler`、梯度累积、warmup + cosine 退火、断点续训与检查点滚动清理、wandb 指标上报。

> 本项目的整体流程与数据组织参考了开源项目 [minimind](https://github.com/jingyaogong/minimind)（在此致谢 👏），但在模型实现、训练器、数据管线与 GRPO 训练监控等方向上均为独立重写与扩展。训练数据集直接引用自 minimind 开源数据集，详见下文 [数据集说明与引用](#-数据集说明与引用) 与 [致谢](#-致谢)。

#### 🎉 主要特性

- ✅ **Tokenizer 训练**：独立训练的 Byte-level BPE（词表 12288：3 个对话控制符 + 256 字节级基础字母表 + BPE merges），经 `add_special_tokens` 扩展 `<tool_call>` / `<tool_response>` / `<think>` 等 8 个 token（补齐至 8 的倍数）后总词表 **12296**；chat template 为 Qwen3 风格（支持 tools、多轮 tool response、`open_thinking` 思考开关）；
- ✅ **两种预训练数据管线**：流式逐条读取（rank × worker 双维分片、buffer shuffle）与 Doc Packing（HF `datasets` 多进程 map + int32 Arrow 存储，拼接切块至定长）；
- ✅ **SFT**：基于 `<|im_start|>assistant\n` 锚点自动构造 assistant 段监督标签；训练时按比例随机保留/移除空 `<think>` 块，实现"该想时想、该直答时直答"的自适应思考数据混入；
- ✅ **DPO**：chosen/rejected 成对编码，冻结参考模型，手写 DPO 损失；
- ✅ **GRPO（RLAIF）**：组内多次 rollout、规则奖励（思考格式 / 长度 / 重复惩罚 / 工具调用 schema 校验）+ 奖励模型打分、组归一化优势、低信号组自动跳过，并内置 **mode collapse 训练健康度监控**；
- ✅ **多注意力后端**：`eager / sdpa / flash_attention_2` 三套实现，FA2 下手写 varlen unpad/pad，可与 eager 交叉验证数值一致性；
- ✅ **实验性探索**（`model/modeling_exp.py`）：MLA 多头潜在注意力、GatedDeltaNet 线性注意力（含线性注意力递推状态缓存）。

---

# 📐 模型结构

主线模型 `FlashLiteForCausalLM`（`model/modeling.py`）为 Decoder-Only Transformer，默认配置如下（`configuration/model_config.py`）：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `vocab_size` | 12296 | Byte-level BPE 词表 12288 + 8 个扩展 token（tool / think / buffer） |
| `hidden_size` | 512 | |
| `intermediate_size` | 1536 | SwiGLU FFN |
| `num_hidden_layers` | 8 | |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA，8:4） |
| `head_dim` | 64 | Q/K 各挂 RMSNorm（Qwen3 风格） |
| `num_experts` / `num_experts_per_tok` | 4 / 2 | Sparse MoE，`norm_topk_prob` 归一化路由 |
| `router_aux_loss_coef` | 0.01 | 专家负载均衡辅助损失 |
| `max_position_embeddings` | 32768 | |
| `rope_theta` | 1e6 | RoPE base，支持 `default / linear / llama3` 缩放。**注意**：若训练小体量、短上下文（如 ≤2048）的小模型，且后续需要外推，建议将 `rope_theta` 设为 **1e4 ~ 1e5**；过高的 base 会使低频维度占比偏高，短上下文训练下外推效果可能不达预期 |
| `tie_word_embeddings` | False | |

按默认配置估算，总参数量 ≈ **94M**，每 token 激活 ≈ **57M**（2/4 专家激活）；不同配置下的实测参数量与训练指标见下文「训练配置与 W&B 指标」。

结构要点：

- **Pre-Norm + RMSNorm**，残差连接；
- **GQA 注意力**：KV 头复用（`repeat_kv`），三种后端共用同一前向接口（`modeling_utils.py` 的 `ALL_ATTENTION_FUNCTIONS`）；
- **手写 causal mask**（`masking_utils.py`）：按 `eager / sdpa / flash_attention_2` 分别构造布尔掩码、加性掩码与 padding 掩码；
- **Sparse MoE**（`SparseMoeBlock`）：softmax top-k 路由 + 权重归一化，逐专家前向后 `scatter_add` 汇聚；`load_balancing_loss_func` 计算负载均衡损失并叠加到总 loss。

实验性配置 `FlashLiteMLAConfig`（`model_type=flashlitemla`）提供 DeepSeek 风格 MLA 参数（`q_lora_rank` / `kv_lora_rank` / nope+rope 分解头维等），配套实现在 `model/modeling_exp.py`，属进行中的探索，未纳入主线训练。

---

# 🗂️ 项目结构

```text
flashlite/
├── configuration/            # 全局配置（dataclass / PretrainedConfig）
│   ├── model_config.py       #   FlashLiteConfig / FlashLiteMLAConfig
│   ├── training_config.py    #   PretrainingArgs / SFTArgs / DPOArgs / GRPOArgs
│   ├── data_config.py        #   DataArgs
│   └── tokenizer_config.py   #   TokenizerArgs
├── model/
│   ├── modeling.py           # 主模型：GQA + Sparse MoE（含推理冒烟测试）
│   ├── modeling_utils.py     # eager / sdpa / FA2 注意力实现，RoPE apply
│   ├── masking_utils.py      # 三后端 causal mask 构造
│   ├── rope_utils.py         # RoPE 初始化（default / linear / llama3）
│   ├── loss_utils.py         # DPO 损失、GRPO 优势 / 损失、token 级 logp
│   └── modeling_exp.py       # 实验性：MLA、GatedDeltaNet、流式 Cache 等
├── trainer/
│   ├── train_tokenizer.py    # BPE 训练 + 扩展 token 补齐
│   ├── train_pretrain.py     # 预训练（流式数据集）
│   ├── train_pretrain_packed.py  # 预训练（Doc Packing 数据集）
│   ├── train_sft.py          # 监督微调
│   ├── train_dpo.py          # 直接偏好优化
│   ├── train_grpo.py         # GRPO（RLAIF）
│   └── trainer_utils.py      # TrainerUtils / DPOTrainerUtils / RLAIFTrainerUtils
├── data_utils/
│   ├── data.py               # 各阶段 Dataset / IterableDataset 与 collate
│   ├── doc_packing.py        # 预训练数据 Doc Packing 离线脚本
│   └── utils.py              # chat 前后处理、标签构造、品牌词替换
├── tokenizer/                # 已训练的 BPE 分词器（可直接使用）
├── base_utils/               # dataclass 参数解析器、检查点管理、HF 权重加载
└── logger/                   # 按日滚动的日志 handler
```

---

# 🚀 快速开始

## 第 0 步：环境安装

需要 CUDA GPU（训练脚本入口处 `assert torch.cuda.is_available()`）。仓库暂未提供 `requirements.txt`，建议依赖清单：

```bash
pip install torch transformers datasets pandas tokenizers safetensors tqdm wandb
# flash-attn 为硬依赖（modeling_utils 顶层 import），仅用 sdpa/eager 后端也需安装
pip install flash-attn --no-build-isolation
```

> 训练脚本在 Windows 下自动使用 gloo 后端、Linux 下使用 nccl；`wandb` 未配置账号时可 `export WANDB_MODE=disabled` 跳过面板上报。

## 第 1 步：数据准备

训练数据直接使用 [minimind](https://github.com/jingyaogong/minimind) 开源数据集（下载方式见 [数据集说明与引用](#-数据集说明与引用)），下载后按阶段放入本地目录，例如：

```text
./datasets/
├── pretrained/       # 预训练语料  {"text": "..."} 逐行 jsonl
├── sft/              # SFT 数据     {"conversations": [...]}（可含 tools / tool_calls / reasoning_content）
├── dpo/              # DPO 数据     {"chosen": [...], "rejected": [...]}
└── rl/               # GRPO 数据    {"conversations": [...], "ground_truths": [...]}
```

各阶段数据格式示例：

```jsonl
# pretrain
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}

# sft（普通对话）
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}

# sft（工具调用；tools 挂在 system，tool_calls 为 JSON 字符串）
{"conversations": [{"role": "system", "content": "# Tools ...", "tools": "[{...}]"},
                   {"role": "user", "content": "现在几点了"},
                   {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"get_time\",\"arguments\":{}}]"},
                   {"role": "tool", "content": "{\"time\": \"12:00\"}"},
                   {"role": "assistant", "content": "现在是 12 点。"}]}

# dpo
{"chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good answer"}],
 "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad answer"}]}

# grpo（最后一个 assistant 由策略模型 rollout 生成；ground_truths 为工具调用真值，可为空）
{"conversations": [{"role": "user", "content": "帮我算 1+1"}], "ground_truths": []}
```

> 数据管线会自动清洗上游语料中的品牌词（如将语料中出现的 `minimind` 替换为 `flashlite`），并校验多轮 agent 对话合法性。

## 第 2 步：训练 Tokenizer（可选）

仓库自带已训练的 `./tokenizer`，可直接跳过。如需从头训练：

```bash
# ① 训练 Byte-level BPE（取消 train_tokenizer.py __main__ 中 train(args) 的注释后执行）
python trainer/train_tokenizer.py --data_dir ./datasets/pretrained --tokenizer_dir ./tokenizer

# ② 补齐 tool_call / think 等扩展 token（pad 到 8 的倍数，当前 __main__ 默认分支）
python trainer/train_tokenizer.py --tokenizer_dir ./tokenizer
```

## 第 3 步：预训练

```bash
# 方式 A：流式数据集（逐条读取，动态 padding）
python trainer/train_pretrain.py \
    --data_dir ./datasets/pretrained \
    --tokenizer_dir ./tokenizer \
    --ckpt_dir ./checkpoint/pretrain

# 方式 B：Doc Packing（先离线打包，再按 DistributedSampler 消费定长样本）
python data_utils/doc_packing.py \
    --data_dir ./datasets/pretrained --tokenizer_dir ./tokenizer \
    --packed_data_dir ./datasets/packed --cache_dir ./cache
python trainer/train_pretrain_packed.py \
    --packed_data_dir ./datasets/packed \
    --tokenizer_dir ./tokenizer \
    --ckpt_dir ./checkpoint/pretrain_packed
```

> Doc Packing 将全部语料 tokenize 后拼接切块为 `max_length`（默认 2048）定长序列，`int32` 存储；collate 时以 `<|endoftext|>` 后一位为界将文档边界标签置 `-100`，避免跨文档的虚假监督。

## 第 4 步：SFT

```bash
python trainer/train_sft.py \
    --data_dir ./datasets/sft \
    --tokenizer_dir ./tokenizer \
    --ckpt_dir ./checkpoint/sft \
    --resume_train false      # 冷启动时关闭断点续训；已有检查点则可省略
```

> `SFTArgs` 默认开启 `resume_train`，首次训练若 `ckpt_dir` 为空会因找不到检查点而报错，此时显式传入 `--resume_train false`。

## 第 5 步：DPO（可选）

先从 SFT 检查点导出 HF 格式目录（可参考 `model/modeling.py` 的 `__main__`：`save_pretrained` 导出 + 往返一致性校验），并将 tokenizer 文件一并放入导出目录（DPO 从 `--pretrained_dir` 加载模型与 tokenizer）：

```bash
python trainer/train_dpo.py \
    --data_dir ./datasets/dpo \
    --pretrained_dir ./sft \
    --tokenizer_dir ./tokenizer \
    --ckpt_dir ./checkpoint/dpo \
    --resume_train false
```

## 第 6 步：GRPO / RLAIF（可选）

需准备一个 `AutoModelForSequenceClassification` 兼容的奖励模型目录（config + 权重 + tokenizer）：

```bash
python trainer/train_grpo.py \
    --data_dir ./datasets/rl \
    --pretrained_dir ./sft \
    --reward_model_dir ./reward_model \
    --tokenizer_dir ./tokenizer \
    --ckpt_dir ./checkpoint/grpo \
    --resume_train false
```

奖励 = **规则奖励**（`<think>` 闭合与长度、回答长度区间、n-gram 重复惩罚、`<tool_call>` JSON 解析 / 名称与 schema 校验 / 重复调用惩罚，截断到 `[-3, 3]`）+ **奖励模型打分**；优势为组内归一化 `(r - μ) / (σ + ε)`，组内 `σ` 低于阈值（默认 0.15）的弱信号组优势置零跳过；策略损失带 PPO-Clip（`clip_eps=0.2`）与 token 级 KL 正则（`beta=0.1`，k3 估计）。

## 第 7 步：推理验证

`model/modeling.py` 自带冒烟测试：加载 `./checkpoint` 最新检查点，跑一次带 system prompt 的对话生成，并验证 `save_pretrained → 重新加载` 的权重往返一致性（同时输出逐 tensor 差异定位）：

```bash
python model/modeling.py
```

`data_utils/data.py` 的 `__main__` 可用于数据管线自查（解码打印 batch）：

```bash
python data_utils/data.py --data_dir ./datasets/sft
```

---

# 🧪 训练工程细节

- **DDP 多卡**：`mp.spawn` 拉起进程组（Windows `gloo` / Linux `nccl`），未到累积步时用 `model.no_sync()` 挂起梯度同步；
- **AMP**：`autocast(bf16)` + `GradScaler`，梯度裁剪在 `unscale_` 之后进行；
- **学习率**：`LinearLR` warmup + `CosineAnnealingLR` 退火，`SequentialLR` 串联；
- **检查点管理**：按 `log_steps` 保存 `trainer_{step}.pth.tar`（模型权重 + step），滚动保留最近 `keep_ckpts`（默认 5）个；`resume_train` 自动定位最新检查点续训；
- **流式数据分片**：`IterableDataset` 按 `rank × num_workers` 双维切分行，文件级随机 + buffer（1 万条）随机两级 shuffle，多卡多 worker 数据不重叠；
- **显存治理**（GRPO）：rollout 分 chunk 生成、`output_router_logits` 按需开关、累积步完成后 `empty_cache`，并打印 sample 前后的 allocated / reserved 显存对比，用于诊断 KV-Cache 泄漏与碎片；
- **训练中抽样**：每 `gen_steps` 步从当前 batch 抽样生成并打印，便于在线观察模型演化；
- **HF 兼容加载**：`base_utils.utils.load_pretrained` 绕过 transformers 5.x 对自定义 `PreTrainedModel` 的 materialization bug（issue #46620 / #43646），直接 `config.from_pretrained + load_state_dict`，支持单文件 / 分片 safetensors / bin 四种权重形式。

# 📊 GRPO 训练把控

小模型 + RLAIF 极易发生 mode collapse 与奖励信号退化。框架在 GRPO 训练中内置了组级行为监控（wandb 可视化），作为常规训练把控手段：

| 指标 | 含义 |
|---|---|
| `train/group_similarity` | 组内各回复两两字符 3-gram Jaccard 相似度均值，骤升 → 输出趋同（collapse 前兆） |
| `train/reward_std` | 组内奖励标准差，持续走低 → 奖励信号退化（退化组问题） |
| `train/low_signal_ratio` | 组内 `σ < adv_std_threshold` 的弱信号组占比，此类组优势置零、不产生策略梯度 |

配合规则 + 奖励模型的混合稠密奖励，避免二元稀疏奖励下"全错组"梯度消失的问题。

---

# 📈 训练配置与 W&B 指标

> 本节展示个人实测训练时的模型配置与 W&B 训练指标截图。
> 当前已补充 **GRPO（RLAIF）** 阶段指标，Pretrain / SFT 阶段待训练完成后补充；参数量以 `sum(p.numel())` 实测为准，默认配置下的 ≈94M 为公式推算值，仅作参考。

## Ⅰ. 测试配置

<!-- 补充说明：默认配置见上文「模型结构」一节表格；
     与默认配置不同的项（如 vocab_size / hidden_size / num_hidden_layers / num_experts /
     num_experts_per_tok / max_position_embeddings / 序列长度 max_length / 数据集规模 / 卡数等）
     请在下表逐项填写实际使用值，方便读者对齐复现条件 -->

| 配置项 | 测试值 | 默认值（对照） |
|---|---|---|
| hidden_size | 待补充 | 512 |
| num_hidden_layers | 待补充 | 8 |
| num_experts / num_experts_per_tok | 待补充 | 4 / 2 |
| max_length（序列长度） | 待补充 | 2048 |
| 参数量（实测） | 待补充 | ≈94M（推算） |
| 训练数据规模 | 待补充 | — |
| GPU 型号 × 数量 | 待补充 | — |

## Ⅱ. GRPO（RLAIF）

W&B project：`FlashLite-GRPO`。将对应面板截图按下述文件名放入 `assets/` 目录（不存在时新建）即可自动渲染：

| 截图内容 | 文件名 |
|---|---|
| 主训练指标：`train/policy_loss`、`train/aux_loss`、`train/reward`、`train/advantage`、`train/lr`、`train/step` | `assets/wandb_grpo_train.png` |
| 训练把控指标：`train/group_similarity`、`train/reward_std`、`train/low_signal_ratio` | `assets/wandb_grpo_health.png` |

![GRPO 主训练指标](assets/wandb_grpo_train.png)

![GRPO 训练把控指标](assets/wandb_grpo_health.png)

## Ⅲ. Pretrain（待补充）

> train_pretrain.py / train_pretrain_packed.py 当前未接入 wandb，仅 logging 输出；如需 W&B 指标，可参照 train_sft.py 的 log_metrics 模式补接（rank0 `wandb.init` + `log_steps` 处 all_reduce 汇总后 `wandb.log`），或在下方贴训练日志截图。

<!-- TODO：预训练指标截图（train/loss、train/aux_loss、train/lr 曲线），建议文件名 assets/wandb_pretrain.png -->

**训练指标**：待补充

## Ⅳ. SFT（待补充）

<!-- TODO：SFT W&B 面板截图（project: FlashLite-SFT）。
      指标：train/loss、train/aux_loss、train/lr、train/step；建议文件名 assets/wandb_sft.png -->

**训练指标**：待补充

---

# 📁 数据集说明与引用

> [!IMPORTANT]
> 本项目的训练数据集引用自 **minimind** 开源数据集（作者 [@jingyaogong](https://github.com/jingyaogong)）：
>
> - **ModelScope**：<https://www.modelscope.cn/datasets/gongjy/minimind_dataset>
> - **HuggingFace**：<https://huggingface.co/datasets/jingyaogong/minimind_dataset>
>
> 各阶段对应关系：预训练语料对应 `pretrain_*.jsonl`（`{"text": ...}`），SFT 对应 `sft_*.jsonl`（`{"conversations": ...}`，含 tool call / reasoning 样本），DPO 对应 `dpo.jsonl`（`{"chosen": ..., "rejected": ...}`），GRPO / RLAIF 对应 `rlaif.jsonl`（`{"conversations": ...}`，末轮 assistant 留空供 rollout 续写；若数据自带 ground truth 工具调用列可直接使用）。数据的下载、清洗与格式细节请遵循 minimind 仓库及其数据集页面的说明与协议要求。

---

# 🙏 致谢

- 特别感谢 **[minimind](https://github.com/jingyaogong/minimind)** 及其作者 [@jingyaogong](https://github.com/jingyaogong)：本项目的训练流程组织与**训练数据集**均来自 / 参考 minimind。正是这类高质量的从零复现开源项目，让个人开发者能够以极低的门槛完整体验 LLM 训练全流程，强烈推荐前往原项目 star 与学习；
- 感谢 HuggingFace `transformers` / PyTorch / `flash-attn` 等基础设施项目；
- 模型结构设计中参考了 Qwen3（Q/K Norm、chat template）、DeepSeek（MoE 负载均衡、实验性 MLA）等公开实现的设计思想。

---

# 📜 License

本项目代码采用 [Apache-2.0](./LICENSE) 协议开源。

**数据集许可说明**：本项目的训练数据集引用自 minimind 数据集，其数据来源包含多种宽松开源协议的公开数据集；使用本项目的训练数据时，请同时遵循 minimind 数据集页面所列来源数据集的协议要求（详见[数据集说明与引用](#-数据集说明与引用)一节）。
