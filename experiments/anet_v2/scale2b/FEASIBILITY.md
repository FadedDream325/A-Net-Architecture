# 2B 参数规模架构对比：可行性与执行方案

- 结论：**当前硬件与当前代码无法执行可信的 2B 训练/能力对比实验。**
- 本文件不包含任何 2B 训练结果；没有用小模型冒充，也没有伪造 benchmark 分数。
- 可行性计算脚本：experiments/anet_v2/scale2b/feasibility.py
- 机器可读结果：experiments/anet_v2/scale2b/feasibility.json

## 1. 硬件检查（实测）

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 专用显存 | 7.933 GB total / 6.877 GB free |
| Compute Capability | 12.0，26 SM |
| 主机内存 | 15.18 GB total / 3.66 GB available |
| CPU | 32 logical / 16 physical |
| 磁盘可用 | 722 GB |
| PyTorch / CUDA | 2.14.1+cu130 / CUDA 13.0 |
| 多卡 | 无 |

## 2. 2B 训练的内存下界

2,000,000,000 个唯一可训练参数：

| 项目 | FP32 | 混合精度（FP16 权重/梯度 + FP32 master/Adam） |
|---|---:|---:|
| 权重 | 8.0 GB | 4.0 GB（FP16）+ 8.0 GB（FP32 master） |
| 梯度 | 8.0 GB | 4.0 GB |
| AdamW 一阶/二阶 | 16.0 GB | 16.0 GB |
| 合计（不含激活/workspace） | **32.0 GB** | **32.0 GB** |

- GPU 只有 8 GB，差至少 4 倍；不可能装载全参数训练状态。
- 主机只有 15.18 GB total、3.66 GB available；连 8 GB FP32 权重都放不下，更不可能做 CPU/offload 全参数 AdamW 训练。
- 即使使用 8-bit Adam、ZeRO-3/FSDP、activation checkpointing，训练也需要多张 40–80 GB GPU；单张 8 GB 不可行。
- 2B FP16 推理权重约 4 GB，单卡 8 GB 理论上可做小 batch 推理；但这只能测延迟/显存，不能评估“模型能力”，因为没有任何 2B 训练好的权重，也不能用随机权重跑 MMLU/GSM8K/HumanEval。

## 3. 计算量下界

训练 FLOPs 近似为 6 × N × D（N=2B）：

| token 预算 | FLOPs |
|---:|---:|
| 1B | 1.2 × 10^19 |
| 20B | 2.4 × 10^20 |
| 100B | 1.2 × 10^21 |

- 20B tokens 只是 2B 模型的 10 tokens/param，远低于现代语言模型常用预算；对 MMLU/GSM8K/代码能力仍可能严重不足。
- 单张 RTX 5060 Laptop 即使按乐观的有效 10–20 TFLOPS 估算，20B tokens 需要数十天到数百天，且多随机种子不可能。
- 要在合理时间完成 3 seeds，至少需要 8×H100 80GB 级别资源；100B tokens、3 seeds 是数周级集群任务。

## 4. 当前代码缺口

当前仓库的 A-Net v2 是**2D 合成任务上的 MLP workflow**，不是语言模型：

- 没有 decoder-only Transformer / causal attention / RoPE / RMSNorm / SwiGLU；
- 没有 tokenizer、语言语料、数据 pipeline；
- 没有 MMLU、GSM8K、HumanEval/MBPP、HellaSwag/ARC 等评测 harness；
- A-Net 的 workflow 目前是 MLP 块，不是 Transformer layer/attention 块；
- ResultCache 处理的是 block 输出张量，不处理序列/KV cache；
- 没有 FSDP/DeepSpeed/ZeRO、混合精度训练、activation checkpointing、分布式 checkpoint；
- 没有 2B 参数配置求解器，也没有 2B 级的单/多任务训练脚本。

因此“在当前仓库实现 2B Transformer vs A-Net”不是改几个宽度就能完成，需要新增语言模型栈；在没有训练资源的情况下也无法验证。

## 5. 参数匹配本身是可解的（但不是已执行）

一个 2B 量级的 decoder-only Transformer 示例配置：

- vocab = 32,000
- d_model = 2,240
- layers = 32
- FFN = 8,960
- tied embeddings
- 解析参数量 = **1,998,584,000**
- 相对 2B 偏差 = **-0.071%**，满足 1% 要求。

参数公式（tied embedding）：

    P = vocab*d + L*(4*d^2 + 2*d*ffn + 2*d) + d

这说明“参数匹配到 2B 的 1% 以内”是可以用解析/元设备完成的；但**参数匹配不等于能力实验**。A-Net 的语言版本、Shared Backbone 的匹配、unique/reference/active/storage 统计都需要先实现。

## 6. 可执行方案（需要外部算力）

### 阶段 A：参数匹配与架构实现（本地可做）

1. 实现 decoder-only Transformer baseline（RoPE、RMSNorm、SwiGLU、causal attention）。
2. 定义语言版 A-Net：
   - pre-L1：token embedding + 前若干层；
   - workflows：若干 Transformer layer 组；
   - 跨 workflow 调用的 attention/FFN 参数块；
   - terminal L1：最终 norm + 输出头；
   - ResultCache 扩展到 KV cache/序列块。
3. 写参数求解器：调整 d_model/layers/ffn/workflow 数，使四种架构唯一参数均在 2B ±1%。
4. 用 meta device 做参数审计：
   - unique trainable（去重）；
   - reference count；
   - active unique per forward；
   - sharing ratio；
   - checkpoint/storage bytes。
5. 本地只做 CPU/meta 参数验证与小规模单元测试，不冒充 2B 能力。

### 阶段 B：小规模管线验证（本地或 1×GPU）

- 100M–300M 代理模型，验证 tokenizer、数据、训练、评测、checkpoint、恢复；
- 明确标注为“管线验证”，不能作为 2B 结论。

### 阶段 C：2B 训练与能力评测（需 8×H100 80GB 或同级）

- 数据：统一语料、统一 tokenizer、统一 token 数、统一去污染；
- 预算：至少 20B tokens 做管线/趋势，能力结论建议 100B tokens 级别；
- 3 seeds；每 seed 独立初始化、固定数据顺序；
- FSDP/ZeRO-3 + activation checkpointing + BF16；
- 统一 optimizer（AdamW）、lr schedule、batch tokens、精度；
- 评测：MMLU、GSM8K、HumanEval/MBPP、HellaSwag/ARC、held-out perplexity；
- 统计：mean、std、95% bootstrap CI、配对比较、多重比较校正；
- 计算/工程：训练 FLOPs、推理 FLOPs、P50/P95 延迟、吞吐、显存峰值、checkpoint 大小。

### 阶段 D：结论门槛

- 只有唯一参数偏差 ≤1%、相同数据/token/优化器/精度、3 seeds、CI 重叠可解释时，才允许比较能力；
- 若 A-Net 相对 Transformer 的提升没有统计显著性，必须结论为“不显著”；
- 参数共享若降低唯一参数但损害表达能力，必须如实报告负迁移/容量损失；
- 结果复用只影响推理执行，不应被写成训练能力提升。

## 7. 成本量级（粗略）

以 8×H100 80GB、有效 1.5–2.5 PFLOPS 估算：

- 20B tokens：2.4×10^20 FLOPs，约 27–44 小时/seed；
- 3 seeds：约 4–6 天（不含评测/失败重跑）；
- 100B tokens：约 5.6–9 天/seed，3 seeds 约 17–27 天；
- 加上数据准备、评测、调参与失败重跑，实际成本更高。

具体云价格取决于供应商；这是一个数万到数十万元人民币量级的项目，不是单卡实验。

## 8. 结论

**当前机器无法执行 2B 训练，也没有语言模型代码栈，因此不能给出可信的 2B 能力对比数据。**  
我不会用 100M/1B 模型冒充 2B，也不会用随机权重跑 benchmark 冒充能力。

可执行的前置工作是：

1. 在本地完成 2B 参数匹配器和 meta-device 参数审计；
2. 实现语言版 Transformer/A-Net 与评测管线；
3. 在 8×H100 级资源上按上面协议训练 3 seeds；
4. 再回答“哪种架构最强、A-Net 是否显著超过 Transformer、共享是否损害表达、缓存是否有收益”。

在此之前，正确的结论是：**2B 能力对比尚未执行，现有 A-Net 证据只覆盖 2D 合成任务，不能外推到 2B 语言能力。**
