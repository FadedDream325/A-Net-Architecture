# 隔离工作流 + L1 深度匹配：3 seeds 纯域评测

- 分支：experiment/anet-v2
- 目标：L1 固定 2 层；每个任务一个完全隔离的 10 层 workflow；总深度 12，与 Transformer-12L 相同；无末端 L1。
- 模型：IsolatedWorkflowANetLM vs Transformer-12L
- seeds：2026、2027、2028
- 训练：数学 12M tokens/run；代码 12M tokens/run；batch=16，seq=128，AdamW，FP32
- 运行数：2 架构 × 2 域 × 3 seeds = 12 runs

## 核心摘要：打平，但不是“成本腰斩”

> 在 12 层、37.5M 单次激活参数、等 FLOPs 的严格控制下，Isolated A-Net 与 Transformer-12L 在 Math EM（0.500 vs 0.517）和 Code 语法 Pass@1（0.220 vs 0.247）上统计打平。  
> 代价是唯一参数 99.83M vs 37.50M（2.66×）、checkpoint 399.5 MB vs 150.1 MB（2.66×）、训练峰值 allocated +18%（1.786 vs 1.508 GB）。  
> 因此这不是“性能平替、成本腰斩”，而是“以唯一存储换模块化隔离；激活计算相同”。

| 指标 | Transformer-12L | Isolated A-Net | 实测关系 |
|---|---:|---:|---|
| 唯一参数 | 37.50M | 99.83M | 2.66× |
| 单次激活参数 | 37.50M | 37.50M | 1.00× |
| 激活 FLOPs/token | 40.59M | 40.59M | 1.00× |
| 深度 | 12 | 12 | 对齐 |
| Math EM | **0.517** | 0.500 | 打平（CI 重叠） |
| Code 语法 Pass@1 | **0.247** | 0.220 | 打平（CI 重叠） |
| 训练峰值 allocated | 1.508 GB | 1.786 GB | +18% |
| checkpoint | 150.1 MB | 399.5 MB | 2.66× |

## 1. 架构

~~~text
L1: 2 blocks（固定、共享）
Wn: 每个任务一个 10-block workflow（彼此不共享、不调用）
总路径: 2 + 10 = 12 blocks
输出: 每个任务直接用自己的 workflow + task head
末端 L1/Lω: 无
~~~

- Wn 之间没有参数共享、没有跨 workflow 调用。
- L1 是唯一的共享部分，2 层固定。
- 每个任务只激活自己的 workflow。
- 这是“专业输出”隔离版本，不是 Shared-Depth，也不是带 terminal 的 A-Net。

## 2. 参数与成本

| 模型 | 唯一参数 | 单次激活参数 | 激活 FLOPs/token | 深度 | 训练峰值 allocated |
|---|---:|---:|---:|---:|---:|
| Transformer-12L | 37,503,168 | 37,503,168 | 40,591,360 | 12 | 1.60–1.72 GB |
| Isolated A-Net | 99,827,968 | 37,501,632 | 40,591,360 | 12 | 1.55–1.84 GB |

- **激活参数和 FLOPs 与 Transformer-12L 基本完全一致**（激活参数差 1,536，FLOPs 相同）。

训练显存实测（allocated / reserved，3 seeds 的 6 个纯域 run）：

| 模型 | allocated 均值 | allocated 最大 | reserved 均值 | reserved 最大 |
|---|---:|---:|---:|---:|
| Transformer-12L | 1.508 GB | 1.606 GB | 1.611 GB | 1.719 GB |
| Isolated A-Net | 1.786 GB | 1.836 GB | 1.925 GB | 1.986 GB |

- A-Net 比 Transformer 多约 **0.278 GB allocated（+18%）**、**0.314 GB reserved（+19%）**。
- 原因：激活参数相同，但唯一参数 99.8M vs 37.5M，Adam 优化器状态更大；不是激活路径导致。
- 唯一参数约为 Transformer 的 **2.66×**：L1 2 + 3 个 10 层 workflow。
- checkpoint：Transformer 150.1 MB；A-Net 399.5 MB。
- 训练时间：Transformer 346–354 s；A-Net 336–367 s，基本同级。

## 3. Math Exact Match（数学 12M tokens）

| 模型 | 2026 | 2027 | 2028 | mean ± 95% CI |
|---|---:|---:|---:|---:|
| Transformer-12L | 0.48 | 0.54 | 0.53 | **0.517 ± 0.080** |
| Isolated A-Net | 0.48 | 0.51 | 0.51 | **0.500 ± 0.043** |

- A-Net 低约 **0.017**（1.7 个百分点），CI 重叠。
- 数学 BPC：Transformer 0.968–1.005；A-Net 0.972–0.990。
- 在深度/激活参数匹配后，数学能力几乎打平；A-Net 没有优势。

## 4. Code Pass@1（语法解析，代码 12M tokens）

| 模型 | 2026 | 2027 | 2028 | mean ± 95% CI |
|---|---:|---:|---:|---:|
| Transformer-12L | 0.40 | 0.10 | 0.24 | **0.247 ± 0.373** |
| Isolated A-Net | 0.27 | 0.29 | 0.10 | **0.220 ± 0.259** |

- A-Net 低约 **0.027**（2.7 个百分点）。
- Transformer 的 seed 方差极大（0.10–0.40），不能当作稳定显著。
- 代码 BPC：Transformer 1.821–1.910；A-Net 1.783–1.919，基本同区间。
- 这是 ast.parse 语法通过率，不是执行/单元测试 Pass@1。

## 5. 事实边界与不可声称项

| 拟表述 | 严格判定 |
|---|---|
| “3 倍物理知识库” | 唯一参数为 2.66×；没有测量“知识量”。 |
| “推理显存仅增加 20%” | 没有单独测推理峰值；训练峰值 allocated +18%。 |
| “1.59 GB vs 1.51 GB” | 这是 Shared-Depth 训练值，不是本报告的 Isolated A-Net；本报告为 1.786 vs 1.508 GB。 |
| “detach_both 物理隔离” | 本模型是结构隔离：独立 workflow、无共享参数；没有使用 detach_both。 |
| “6 层残血保持 92% 能力” | 未测到 92%；6 层 Sparse 没有做 EM/Pass。 |
| “100M 训练 1 小时，Transformer 2–5 小时” | 没有 100M Transformer 对照；A-Net 是 6 个独立 12M-token run，总训练约 35 分钟，但不是端到端同体量比较。 |
| “8GB 绝对无法训练 100M Transformer” | 不成立/未验证；100M FP32+Adam 静态状态约 1.6 GB，是否 OOM 取决于 batch/seq/实现。 |
| “多节点分布式联合训练” | 未实现；当前只有独立训练/独立 checkpoint，没有跨节点通信与整合。 |
| “30B+ 必然可落地” | 未来推演，不是本实验结论。 |

## 6. 物理边界与未来工作

### 已确认不可行

**总参数 60M + 3 个完全隔离 workflow + 1.5×/2× 深度 + 等激活参数/FLOPs** 不能同时成立。

原因：

- 3 个隔离 workflow 使总参数约为激活路径的 3 倍；
- 60M 总参数对应约 21M 激活；
- 不可能等于 60M Transformer 的激活。

### 已执行

- 2 个 L1 block + 3 个独立 10-block workflow；
- 总路径 12 层；
- 唯一参数 99.83M，激活参数 37.50M，FLOPs 40.59M/token；
- Math/Code 各 12M tokens，3 seeds；
- 与 Transformer-12L 打平，但唯一存储/checkpoint 为 2.66×。

### 未执行（未来工作，不是结果）

1. 未激活专家流驻留 DDR5/NVMe 的冷热分离；
2. 跨节点独立训练与 L1 拼接；
3. 30B+ 参数、40–60 层专家流；
4. 单独测 Isolated A-Net 的推理峰值、P50/P95、吞吐；
5. 在 8×H100 级资源上复现更大规模。

## 7. 结论

1. **把 L1+workflow 深度匹配到 12、保证激活参数/FLOPs 一致、并让 Wn 完全隔离后，A-Net 与 Transformer 基本打平。**
2. 数学 EM：0.500 vs 0.517；代码语法 Pass@1：0.220 vs 0.247；差异都在 seed 噪声范围内。
3. 没有证据支持“隔离工作流 A-Net 超过 Transformer”。
4. A-Net 的唯一参数和 checkpoint 约为 Transformer 的 2.66×；激活计算相同，因此这不是存储优势。
5. 之前 A-Net 更差的主要原因是有效深度/激活路径更短；深度匹配后差距基本消失。

## 8. 局限

- 50M 级模型；数学/代码各 12M tokens，远未饱和。
- 3 seeds，CIs 宽；没有正式显著性结论。
- 数学只有 100 题；代码只测语法解析。
- 唯一参数 99.8M 不是 50M 预算；这是隔离 + 12 层深度的代价。
- 没有 Lα 特征分解、没有 Lω 可选整合；本实验是“无末端 L1 + 专业输出”的隔离版本。
- 结果只适用于 byte-level 代理任务，不是 MMLU/GSM8K/HumanEval。

## 9. 产物

- 模型：experiments/anet_v2/lang50m/model.py（IsolatedWorkflowANetLM）
- 训练：experiments/anet_v2/lang50m/phase1_train.py
- 评测：experiments/anet_v2/lang50m/phase1_eval.py
- A-Net 原始结果：experiments/anet_v2/lang50m/results/phase1_isolated_12m/raw_runs.jsonl
- A-Net 评测：experiments/anet_v2/lang50m/results/phase1_isolated_12m/eval.json
- Transformer code 12M 基线：experiments/anet_v2/lang50m/results/phase1_tcode_12m/raw_runs.jsonl
- checkpoint 未提交（约 2.9 GB）
