# 参数规模推荐：1B 作为最终验证规模

- 结论：**最终验证推荐 1B 唯一可训练参数**，而不是 2B。
- 本地（8 GB GPU / 15 GB RAM）只能做 50M–100M 的管线验证；1B 训练必须上云。
- 2B 不是不能做，而是相对 1B 成本约翻倍，对“A-Net 是否优于 Transformer”这个架构问题没有数量级上的额外信息量。

## 1. 推荐配置

1B decoder-only Transformer 示例（tied embeddings）：

| 项目 | 值 |
|---|---:|
| vocab | 32,000 |
| d_model | 1,808 |
| layers | 24 |
| FFN | 7,232 |
| tied embeddings | 是 |
| 参数量 | **999,377,424** |
| 相对 1B 偏差 | **-0.062%** |

参数公式：

    P = vocab*d + L*(4*d^2 + 2*d*ffn + 2*d) + d

其他架构用同一预算求解器调整层数/宽度/workflow 数，使 unique trainable 参数都在 1B ±1%。

## 2. 为什么是 1B

- 2B 的训练状态约 32 GB（权重/梯度/Adam，不含激活）；1B 约 16 GB，单张 80 GB GPU 可承载，40 GB GPU 配合 activation checkpointing 也可做小 batch。
- 1B 已经能观察架构在语言建模、推理、代码上的相对差异；2B 主要增加成本，不改变“A-Net vs Transformer”的定性问题。
- 100M 以下只能验证管线，无法支撑 MMLU/GSM8K/HumanEval 这类能力结论。
- 500M 处于尴尬区间：能力仍弱，且本地 8 GB 也难舒适训练；不如直接 1B 上云。

## 3. 资源与预算

### 最低可执行

- 1×H100 80GB（或 1×A100 80GB）；
- 1B 混合精度（BF16/FP16 + FP32 master/Adam）；
- 20B tokens 先跑通并做趋势；
- 3 seeds；
- 时间量级：20B tokens = 1.2×10^20 FLOPs；单卡 H100 约数天/seed，实际取决于有效算力。

### 能力结论建议

- 8×H100 80GB；
- 100B tokens（6×10^20 FLOPs）；
- 3 seeds；
- 约 1–2 周级训练（不含数据、SFT、评测和失败重跑）；
- 统一数据、tokenizer、token 数、AdamW、lr schedule、精度；
- 评测 MMLU、GSM8K、HumanEval/MBPP、HellaSwag/ARC、held-out perplexity；
- 统计 mean/std/95% CI、配对比较、多重比较校正。

## 4. 本地能做什么

| 规模 | 本地可行性 | 用途 |
|---|---|---|
| 50M | 可以 | 管线、tokenizer、评测、checkpoint 验证 |
| 100M | 可以（8 GB 需 activation checkpointing/小 batch） | 趋势/消融，不作为最终能力结论 |
| 300M | 勉强/易 OOM | 不建议作为本地主实验 |
| 1B | 不可训练 | 只能做 FP16 推理/参数审计 |
| 2B | 不可训练 | 只能做 meta 参数审计 |

## 5. 最终建议

**把最终验证规模定为 1B 唯一参数，采用 20B tokens 做 pilot、100B tokens 做能力结论，3 seeds，8×H100 级资源。**

如果只能用当前机器：**把目标降为 50M–100M 的管线与趋势实验，并明确它不是最终能力验证**；不要用这个规模冒充 1B/2B 结论。

## 6. 下一步

1. 本地实现 1B 参数求解器 + meta 参数审计（unique/reference/active/sharing/storage）。
2. 实现 1B Transformer/A-Net 语言栈与评测接口。
3. 云端先跑 100M/1B 小 token 管线，再开 20B/100B tokens 的 3-seed 正式实验。
