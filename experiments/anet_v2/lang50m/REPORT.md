# 50M 三架构语言模型 pilot 报告

- 目的：在 50M 唯一参数预算下，比较 Transformer、Shared Backbone、A-Net。
- 性质：**pilot / 代理实验**，不是 MMLU/GSM8K/HumanEval 能力结论。
- 数据：Tiny Shakespeare（GitHub）、CPython 源码（GitHub）、本地生成四则运算/括号表达式。
- 规模：50M 唯一参数，3 seeds（2026–2028），每 seed 1500 步，batch 16，seq 128。
- 原始数据：experiments/anet_v2/lang50m/results/run_50m/raw_runs.jsonl
- 汇总：summary.csv / summary.json

## 1. 参数匹配

| 架构 | 唯一可训练参数 | 相对 50M 偏差 | 存储 FP32 |
|---|---:|---:|---:|
| Transformer | 49,915,904 | -0.168% | 199.66 MB |
| Shared Backbone | 50,178,560 | +0.357% | 200.71 MB |
| A-Net | 50,177,033 | +0.354% | 200.71 MB |

三种架构都在 50M ±1% 内。

A-Net 的统计：

| 口径 | 数值 |
|---|---:|
| 唯一参数 | 50,177,033 |
| 参数引用总量 | 56,383,401 |
| 单次前向激活唯一参数 | 37,501,632 |
| 共享参数元素 | 6,206,368 |
| 共享比例（shared / unique） | 12.37% |

- A-Net 的唯一参数是 50M，但 pair 模式只激活 2/3 个 workflow，单次前向只用到约 37.5M 参数；**这不是等激活计算比较**。
- Transformer/Shared Backbone 单次前向激活全部约 50M 参数。

## 2. 结果：bits-per-byte（越低越好）

3 seeds mean ± 95% CI：

| 架构 | text | code | math | 平均 |
|---|---:|---:|---:|---:|
| Transformer | 2.638 | 1.815 | 1.568 | **2.007 ± 0.055** |
| Shared Backbone | 2.608 | 1.775 | 1.542 | **1.975 ± 0.028** |
| A-Net | 2.727 | 2.017 | 1.677 | **2.141 ± 0.107** |

- Shared Backbone 在本 pilot 中最好；比 Transformer 低约 0.032 bpc，但两者 CI 重叠，不能称为显著。
- A-Net 比 Transformer 高约 **0.133 bpc**（更差），CI 与 Transformer 基本不重叠；这不是“A-Net 超过 Transformer”的证据。
- 三个任务上 A-Net 都更差，代码任务差距最大（2.017 vs 1.815）。
- 3 seeds、每个任务约 1M 训练字符，预算很小；结论只能作为趋势。

## 3. 工程性能

| 架构 | 训练时间/seed | batch1 P50 | batch16 P50 | 峰值显存 |
|---|---:|---:|---:|---:|
| Transformer | 121.7 s | 4.33 ms | 18.14 ms | 1.98 GB |
| Shared Backbone | 120.8 s | 4.79 ms | 18.20 ms | 2.11 GB |
| A-Net | 94.6 s | 5.67 ms | 14.53 ms | 1.69 GB |

- A-Net 训练更快、显存更低，主要因为单次只激活 37.5M 参数（2 个 workflow + pre + shared）。
- batch=1 延迟 A-Net 更差（5.67 ms vs 4.33/4.79）；batch=16 时 A-Net 更低（14.53 ms），与激活参数更少一致。
- 这是不同激活计算量下的比较，不能直接得出“A-Net 工程更优”。

## 4. 能回答与不能回答

能回答（仅限该 pilot）：

1. 50M 参数匹配和训练管线可跑通；3 seeds 可完成。
2. 在这个小预算、char-level 三域任务上，**A-Net 没有超过 Transformer**，反而更差。
3. Shared Backbone 在该设置下略好于 Transformer，但差异不显著。
4. A-Net 的唯一参数是 50M，但激活参数只有约 37.5M；参数共享没有带来表达能力优势，至少在这个 pilot 中没有。

不能回答：

- MMLU、GSM8K、HumanEval、知识问答、代码能力；
- 2B/1B 规模下的架构结论；
- 结果复用的收益（本 pilot 没有测 cache）；
- 统计显著的“架构优劣”最终结论（3 seeds、3M tokens、小数据）。

## 5. 结论

在这个 50M pilot 中：

- **能力最强：Shared Backbone（1.975 bpc）**；
- **A-Net 没有显著超过 Transformer；实际更差（2.141 vs 2.007 bpc）**；
- 参数共享没有表现出表达能力优势；A-Net 的激活容量更低是重要混杂因素；
- 结果复用未测试，不能评价；
- 该结果不能外推到 1B/2B 或真实语言 benchmark。

## 6. 局限

- 每任务约 1M 训练字符，1500 步，严重不足以训练 50M 模型到收敛。
- 数据是 Shakespeare + CPython 源码 + 合成数学，不是代表性语言/知识/代码 benchmark。
- 只有 3 seeds；CI 较宽。
- A-Net 使用 pair 路由，单次只激活 2 个 workflow；与全参数激活的 Transformer 不是等计算比较。
- 没有接 SFT；没有 MMLU/GSM8K/HumanEval。
- 没有测试 A-Net Basic/Share/Full 的 cache 消融；本报告只有三种架构。
