# Phase 1 纯域准确率报告：3 seeds

- 分支：experiment/anet-v2
- 模型：Transformer-12L vs Shared-Depth A-Net
- seeds：2026、2027、2028
- 训练：每 run 6M tokens，batch=16，seq=128，AdamW，FP32
- 纯域：100% math、100% code
- 运行数：2 架构 × 2 域 × 3 seeds = **12 runs**
- 评测：math Exact Match；code syntax Pass@1

## 1. 评测定义

### Math Exact Match

- 从 math_test 取 100 个 Q/A 样本；
- prompt 截到 “A: ”；
- 贪婪生成最多 12 bytes，遇到换行停止；
- 解析生成整数；
- 与目标整数完全相等记 1，否则 0。

这不是“答案包含正确数字”，而是严格整数精确匹配。

### Code Pass@1（语法解析）

- 从 code_test 取 100 个 6 行、可被 ast.parse 解析的完整前缀；
- prompt 为该前缀；
- 贪婪生成最多 24 bytes，遇到换行停止；
- 要求 continuation 非空；
- 对 prefix + continuation 做 ast.parse；
- 无 SyntaxError 记 1。

这是**语法解析 Pass@1**，不是执行测试/单元测试 Pass@1。

## 2. 参数与成本

| 模型 | 唯一参数 | 激活参数 | 激活 FLOPs/token | 训练 allocated 峰值 | 训练 reserved 峰值 |
|---|---:|---:|---:|---:|---:|
| Transformer-12L | 37,503,168 | 37,503,168 | 40,591,360 | 1.556 GB 均值 / 1.607 GB 最大 | 1.665 GB 均值 / 1.727 GB 最大 |
| Shared-Depth A-Net | 50,177,024 | 37,501,632 | 40,591,360 | 1.650 GB 均值 / 1.651 GB 最大 | 1.772 GB 均值 / 1.781 GB 最大 |

测量口径：torch.cuda.max_memory_allocated / max_memory_reserved，batch=16、seq=128、FP32、纯 math 或纯 code 训练。A-Net 比 Transformer 多约 0.094 GB allocated（+6.0%）和 0.107 GB reserved（+6.4%）。

- 12 个 checkpoint 约 2.4 GB；只用于评测，未提交。
- 每 run 训练 165–180 秒。

## 3. Math Exact Match（3 seeds）

| 模型 | seed 2026 | seed 2027 | seed 2028 | mean ± 95% CI |
|---|---:|---:|---:|---:|
| Transformer-12L | 0.25 | 0.34 | 0.43 | **0.340 ± 0.224** |
| Shared-Depth A-Net | 0.10 | 0.28 | 0.31 | **0.230 ± 0.282** |

- A-Net 均值低 0.110（约 11 个百分点），但 3 seeds 的 CI 很宽，不能把差异当作稳定显著。
- 纯数学 BPC 也支持同一方向：Transformer 1.023–1.044；A-Net 1.063–1.172。
- **纯数学下 A-Net 没有超过 Transformer。**

## 4. Code Pass@1（语法解析，3 seeds）

| 模型 | seed 2026 | seed 2027 | seed 2028 | mean ± 95% CI |
|---|---:|---:|---:|---:|
| Transformer-12L | 0.20 | 0.10 | 0.26 | **0.187 ± 0.201** |
| Shared-Depth A-Net | 0.24 | 0.15 | 0.17 | **0.187 ± 0.117** |

- 两者均值完全一样（0.1867），但方差不同；没有证据表明 A-Net 的代码语法通过率更高。
- 纯代码 BPC：A-Net 2.004–2.094，Transformer 2.033–2.168；A-Net 略低，但 Pass@1 没有转化。
- **纯代码下 A-Net 没有语法 Pass@1 优势。**

## 5. 与混合比例 pilot 的关系

Phase 1 混合比例 BPC pilot 中，A-Net 在 75/50/25% 数学比例下两个域都略低；但本纯域 EM/Pass 评测没有复现能力优势：

- 纯数学：A-Net 更差；
- 纯代码：Pass@1 持平。

这说明“BPC 略低”不能直接等同于“任务准确率更高”，需要分别报告 BPC 和任务指标。

## 6. 结论

在当前 50M/6M-token 设置下：

1. **纯数学 Exact Match：Transformer 更好（0.340 vs 0.230），但 3-seed CI 重叠。**
2. **纯代码语法 Pass@1：两者持平（0.187 vs 0.187）。**
3. **没有证据支持 A-Net 在纯域能力上超过 Transformer。**
4. A-Net 的混合比例 BPC 优势没有转化为纯域 EM/Pass 优势。

## 7. 局限

- 50M 模型、6M tokens/run；远未到能力饱和。
- 3 seeds，CI 很宽；没有正式显著性结论。
- Math EM 只有 100 题；生成是贪婪解码，不是采样/多答案。
- Code Pass@1 只测 ast.parse，不执行代码；不能等同于功能正确性。
- 当前 A-Net 是 Shared-Depth 变体，没有 Lα 特征分解和 Lω 整合；不能代表完整设计。
- 结果只适用于 byte-level char/代码语法代理，不是 MMLU/GSM8K/HumanEval。

## 8. 产物

- 评测脚本：experiments/anet_v2/lang50m/phase1_eval.py
- 评测结果：experiments/anet_v2/lang50m/results/phase1_pure/eval.json
- 训练原始结果：experiments/anet_v2/lang50m/results/phase1_pure/raw_runs.jsonl
- checkpoint：experiments/anet_v2/lang50m/results/phase1_pure/checkpoints（未提交）
