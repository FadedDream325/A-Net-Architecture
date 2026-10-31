# 无末端 L1 / 专业输出 A-Net：3 seeds 纯域评测

- 分支：experiment/anet-v2
- 目的：去掉末端 L1（Lω/跨 workflow 加权整合），让每个任务直接走自己的专业 workflow 和任务头。
- 模型：NoTerminalANetLM vs Transformer-12L
- seeds：2026、2027、2028
- 训练：数学 12M tokens/run；代码 6M tokens/run；batch=16，seq=128，AdamW，FP32
- 运行数：2 架构 × 2 域 × 3 seeds = 12 runs

## 1. 架构

NoTerminalANetLM：

~~~text
embedding -> pre 2 blocks -> primary workflow 4 blocks -> + shared 2 blocks -> task head
~~~

- 每个任务只用自己的 primary workflow 和任务头；
- 没有 terminal 加权平均；
- 没有辅助 workflow 输出；
- shared 2 个 block 作为共享残差特征；
- 这是 sparse A-Net 去掉 terminal L1 的消融，不是 Shared-Depth（Shared-Depth 本来就没有 Lω）。

## 2. 参数与成本

| 模型 | 唯一参数 | 单次激活参数 | 激活 FLOPs/token | 有效深度 | 训练峰值 allocated |
|---|---:|---:|---:|---:|---:|
| Transformer-12L | 37,503,168 | 37,503,168 | 40,591,360 | 12 | 1.60–1.61 GB |
| NoTerminal A-Net | 50,177,024 | 25,088,896 | 27,148,288 | 6（+共享旁路2） | 1.19–1.29 GB |

- A-Net 显存更低，是因为单次只激活 25M 参数、有效路径更短，不是共享本身省显存。
- 两者不是等激活参数比较；A-Net 的激活 FLOPs 约为 Transformer 的 67%。

## 3. Math Exact Match

Math 训练 12M tokens；测试 100 题；严格整数精确匹配。

| 模型 | 2026 | 2027 | 2028 | mean ± 95% CI |
|---|---:|---:|---:|---:|
| Transformer-12L | 0.48 | 0.54 | 0.53 | **0.517 ± 0.080** |
| NoTerminal A-Net | 0.44 | 0.52 | 0.47 | **0.477 ± 0.100** |

- A-Net 均值低约 0.040（4 个百分点）。
- 3 seeds 的 CI 重叠，不能称为稳定显著。
- 与 6M-token 的 Shared-Depth 结果相比，数学加长后 Transformer EM 从 0.34 提升到 0.52；但 A-Net 仍低于 Transformer。

## 4. Code Pass@1（语法解析）

Code 训练 6M tokens；测试 100 个可解析的 6 行代码前缀。

| 模型 | 2026 | 2027 | 2028 | mean ± 95% CI |
|---|---:|---:|---:|---:|
| Transformer-12L | 0.15 | 0.10 | 0.29 | **0.180 ± 0.245** |
| NoTerminal A-Net | 0.15 | 0.12 | 0.16 | **0.143 ± 0.052** |

- A-Net 均值低约 0.037，但 Transformer 的 seed 方差很大（0.10–0.29）。
- 这是 ast.parse 语法通过率，不是执行/单元测试 Pass@1。
- 纯代码 BPC：Transformer 2.033–2.160；A-Net 2.192–2.210，A-Net 略差。

## 5. 结论

1. **去掉末端 L1、改走专业输出后，A-Net 没有超过 Transformer。**
2. 数学 EM：0.477 vs 0.517；代码语法 Pass@1：0.143 vs 0.180；两者 CI 均重叠。
3. A-Net 的训练显存更低（约 1.2 vs 1.6 GB），但这是激活参数只有 25M 的代价，不是无损优势。
4. 这个消融不能单独证明“末端 L1 有害”，因为 A-Net 同时减少了有效深度和激活计算。
5. 当前 A-Net 的数学/代码纯域能力仍不支持“超过 Transformer”。

## 6. 局限

- 50M 模型，数学 12M/code 6M tokens；远未饱和。
- 3 seeds，CI 宽；没有正式显著性。
- 数学只有 100 题；代码只测语法解析。
- NoTerminal A-Net 与 Transformer 不是等激活参数/等 FLOPs 比较。
- 没有 Lα 特征分解；没有 Lω 可选开关（本实验直接移除）。
- 结果只适用于 byte-level 代理任务，不是 MMLU/GSM8K/HumanEval。

## 7. 产物

- 模型：experiments/anet_v2/lang50m/model.py（NoTerminalANetLM）
- 训练：experiments/anet_v2/lang50m/phase1_train.py
- 评测：experiments/anet_v2/lang50m/phase1_eval.py
- 原始训练：experiments/anet_v2/lang50m/results/phase1_no_terminal_math12m/raw_runs.jsonl
- 评测：experiments/anet_v2/lang50m/results/phase1_no_terminal_math12m/eval.json
- checkpoint：未提交（约 2.1 GB）
