# A-Net v2 阶段 2 实验报告：跨工作流参数共享

- 分支：experiment/anet-v2
- 代码提交：ff14545（Stage 2 runner / gradient audit）；实验记录含 code_hash
- 数据：experiments/anet_v2/results/stage2/raw_runs.jsonl
- 汇总：summary.csv / summary.json / summary.md
- 规模：315 个成功 run，0 失败
- 环境：RTX 5060 Laptop，8 GB 专用显存；显存守卫开启；峰值 allocated 0.218 GB

## 1. 本阶段回答什么

1. 跨工作流调用同一参数块时，唯一参数存储是否真的下降？
2. 多个调用方的梯度是否能正确累积到同一参数对象？
3. 共享参数更新后，所有调用方是否看到同一版本？
4. 不同任务目标在共享参数上是否产生梯度冲突/负迁移？
5. 直接共享与“共享 + 适配器”相比如何？
6. pair（2 个工作流）与 dense（全部 3 个工作流）两种并发模式有什么差异？

## 2. 设计与对照

- 宽度：128、512、2048。
- 相关性：low、medium、high。
- seeds：2026–2030（5 个）。
- epochs：30；train/val/test = 6000/1500/1500 每任务。
- 并发模式：
  - pair：每样本激活 2 个工作流；
  - dense：每样本激活全部 3 个工作流。
- 模型：
  - Independent：每个任务独立 MLP（只用于 pair 参考；dense 不重复跑）；
  - A-Net Basic：每个 workflow 拥有自己的 cross block，无共享；
  - A-Net Share：所有 workflow 调用 w0 拥有的同一个 cross block；
  - A-Net Share+Adapter：共享 cross block + 每个 workflow 一个低秩适配器，适配器参数单独统计。
- 缓存：全部关闭；本阶段只隔离“参数共享”，不混入结果复用。
- 每个 A-Net run 训练后执行一次共享块梯度审计：同一输入 batch、逐任务 loss/backward，不更新参数，记录每任务梯度范数、两两 cosine、负 cosine 比例。

## 3. 对象身份与存储

运行期身份审计：

- A-Net Share：90/90 runs 的 shared_cross_blocks 恰为 1 个；
- A-Net Share+Adapter：90/90 runs 恰为 1 个共享 cross block；
- A-Net Basic：90/90 runs 没有跨 workflow 共享块；
- 共享块在 registry 中只注册一次；其他 workflow 持有 grant，不持有参数副本；
- 适配器参数单独记录，范围 3,288–52,248，取决于宽度；
- 所有 run 的 input_device=cuda:0，峰值显存 <= 0.218 GB，无 shared GPU memory 回退。

唯一参数相对 Independent（跨全部 pair 宽度/相关性）：

| 模型 | 唯一参数比 | 激活参数比 | 适配器参数 |
|---|---:|---:|---:|
| A-Net Basic | 0.977 | 1.96 | 0 |
| A-Net Share | 0.656 | 1.48 | 0 |
| A-Net Share+Adapter | 0.683 | 1.53 | 单独计 |

解释：

- 共享确实把唯一参数从 Basic 的 0.977× 降到 Share 的 0.656×；在 h512 上为 531,985 vs 794,641，约 33% 存储下降。
- 但 pair 模式本身激活 2 个工作流，所以 Share 的激活参数仍是 Independent 的 1.48×，Basic 是 1.96×；共享降低了相对 Basic 的激活计算，但没有把 pair 模式变成“比 Independent 更省计算”。
- Adapter 只增加约 2.6% 唯一参数（相对 Share），但能提供每 workflow 的特化；代价是额外计算。

## 4. 精度与负迁移

### 4.1 pair，宽度 512，medium 相关性（5 seeds）

| 模型 | task0 binary acc | task1 regression MSE | task2 5-class acc | 平均 acc | 唯一参数 | 激活参数 |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 97.88 | 0.0008 | 94.23 | 96.05 ± 0.42 | 805,896 | 268,290 |
| A-Net Basic | 96.79 | 0.0062 | 93.59 | 95.19 ± 1.02 | 794,641 | 530,443 |
| A-Net Share | 96.67 | 0.0198 | 93.27 | 94.97 ± 0.91 | 531,985 | 399,115 |
| A-Net Share+Adapter | 96.48 | 0.0057 | 93.37 | 94.93 ± 0.72 | 545,065 | 407,835 |

### 4.2 pair 相对 Independent 的整体差（跨 3 宽度 × 3 相关性）

| 模型 | 平均 acc 差（pp） | task0 acc 差（pp） | task1 MSE 差 | task2 acc 差（pp） |
|---|---:|---:|---:|---:|
| A-Net Basic | -0.953 | -1.775 | +0.0058 | -0.130 |
| A-Net Share | -1.197 | -2.138 | +0.0106 | -0.256 |
| A-Net Share+Adapter | -1.627 | -3.007 | +0.0130 | -0.246 |

- 所有 A-Net 变体在 pair 模式下都低于 Independent；主要损失在二分类任务和回归任务。
- 共享相对 Basic 平均再低约 0.24 pp；回归 MSE 从 0.0058 增到 0.0106（约 1.8×）。
- 适配器没有消除负迁移：平均再低 0.67 pp，task0 损失扩大到 -3.0 pp，回归 MSE 也更差（0.0130 vs Share 的 0.0106）。
- 重要：pair 模式激活 2 个工作流，而 Independent 只跑 1 个任务 MLP；A-Net 的激活参数是 1.48–1.96× Independent，因此这不是等计算比较，精度差不能单独归因于“共享”。

### 4.3 dense（3 个工作流）概要

dense 模式下，A-Net Share 与 Basic 的差异较小且方向不一致：

| width | Basic acc | Share acc | Adapter acc |
|---:|---:|---:|---:|
| 128 | 94.59 | 94.96 | 94.52 |
| 512 | 94.44 | 94.32 | 93.58 |
| 2048 | 93.49 | 92.92 | 93.92 |

（三个相关性的平均。）dense 的激活参数为 Independent 的约 3×；它验证的是“全 workflow 并发”的结构，不是效率优势。

## 5. 梯度冲突

共享块（w0.cross）的逐任务梯度审计，pair 模式：

| 模型 | 平均 pairwise cosine | 最小 cosine（典型） | 负 cosine 比例 |
|---|---:|---:|---:|
| A-Net Share | +0.025 | 约 -0.04 | 0.43 |
| A-Net Share+Adapter | +0.051 | 约 -0.01 | 0.37 |

- 平均 cosine 略为正，但**负 cosine 比例仍有 37–43%**：大约每 2–3 个任务对中就有 1 个方向冲突。平均值会掩盖局部冲突。
- 在 h512 / medium / pair 上，Share 的负 cosine 比例为 0.533，task1 MSE 为 0.0198，而 Basic 为 0.0062；冲突与回归性能下降同时出现，但这里只能说明相关性，不能单独证明因果。
- 适配器把平均 cosine 从 +0.025 提高到 +0.051，并降低负冲突比例，但没有转化为稳定精度收益；某些配置甚至更差。
- 有少数 run 明显崩溃（如 pair h512 low Share 90.57%，pair h128 medium Adapter 89.33%），说明共享参数在高冲突任务上存在不稳定性；A-Net Basic 也有个别低分 run（h2048 low 90.63%），所以不能把所有方差都归因于共享。

## 6. 工程性能

本阶段只报告与参数共享相关的延迟/吞吐；缓存全部关闭，数据全部在 cuda:0。

- pair h512 medium：A-Net Basic 612 µs，Share 643 µs，Adapter 650 µs（batch=1 P50，5 seeds 均值；Independent 的 single 约 80–100 µs，但 Independent 不运行 pair，不可直接当作同计算比较）。
- 共享减少了唯一参数和激活参数，但没有降低 eager batch=1 延迟；多工作流串行调用仍是 launch/Python 开销主导。
- dense 的延迟更高（h512 medium：Basic 622 µs，Share 621 µs，Adapter 865 µs），与激活 3 个工作流一致。
- 峰值显存 <= 0.218 GB；没有 shared GPU memory 回退。

## 7. 结论

1. **参数共享的存储收益成立。** Share 的唯一参数约为 Independent 的 0.656×，比 Basic 的 0.977× 低约 33%；共享块确实只有一个对象、一个存储位置。
2. **对象身份/版本/授权机制成立。** 90/90 Share、90/90 Adapter run 都只有一个 shared cross block；其他 workflow 通过 grant 调用同一对象；适配器参数单独统计。
3. **共享参数的计算收益只在相对 Basic 的意义上成立。** pair 模式下 Share 激活参数 1.48× Independent，Basic 1.96×；共享把 A-Net 自身的激活计算降低约 25%，但仍高于 Independent，因为 pair 本身跑 2 个工作流。
4. **共享没有带来精度收益。** Share 比 Independent 低 1.20 pp，比 Basic 低约 0.24 pp；回归 MSE 明显变差。Adapter 没有修复负迁移。
5. **梯度冲突真实存在但幅度中等。** 平均 cosine 接近 0（+0.025/+0.051），负 cosine 比例 37–43%；个别配置崩溃，说明需要按任务/配置选择共享，而不是无条件共享。
6. **高相关性没有消除负迁移。** 在该任务集合上，相关性不是共享冲突的充分解药；共享块容量、任务目标和适配器设计更关键。
7. **当前最优的诚实表述：** 跨工作流参数块调用是“可正确实现、可降低唯一存储”的机制；在本阶段的 pair/dense 设定下，它没有带来稳定的任务性能或延迟收益，且存在可测的梯度冲突和负迁移。

## 8. 局限

- pair/dense 激活多个 workflow，与 Independent 的单任务 MLP 不是等计算比较；不能把精度差直接归因于共享机制。
- 只有 3 个合成任务；没有真实任务上的共享结构先验。
- 30 epochs、固定超参；没有针对共享块的学习率/正则化调优。
- 梯度审计使用同一个输入 batch 和逐任务 loss；它衡量的是任务目标冲突，不等同于完整训练动力学。
- Adapter 只测试了一种低秩结构（rank=8）和一个位置；没有做 adapter 容量/位置消融。
- 峰值显存 0.218 GB，远低于专用显存；本阶段没有触发 OOM 或共享显存回退。

## 9. 产物

| 文件 | 内容 |
|---|---|
| experiments/anet_v2/results/stage2/raw_runs.jsonl | 315 条原始记录（含 gradient_audit、identity、latency raw） |
| experiments/anet_v2/results/stage2/summary.csv | 63 个 (mode,width,corr,model) 组汇总 |
| experiments/anet_v2/results/stage2/summary.json | 汇总与失败列表 |
| experiments/anet_v2/results/stage2/summary.md | 可读表格 |
| experiments/anet_v2/results/stage2/figures/ | 精度与梯度 cosine 图 |
| experiments/anet_v2/core/gradient_audit.py | 梯度冲突审计 |
| experiments/anet_v2/runners/run_stage2.py | Stage 2 runner |
