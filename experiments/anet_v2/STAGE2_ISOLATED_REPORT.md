# Stage 2 隔离版报告：关闭共享块/辅助工作流梯度互联

- 分支：experiment/anet-v2
- 代码提交：76fdf21（隔离模式）；实验记录含 code_hash
- 数据：experiments/anet_v2/results/stage2_isolated_both/
- 规模：315 个成功 run，0 失败
- 模式：gradient_isolation = detach_both
- 对照：同一矩阵的耦合版 experiments/anet_v2/results/stage2/

## 1. 隔离语义

detach_both 同时做两件事：

1. detach_shared：所有调用方拿到的共享 cross block 输出都 detach，梯度不进入共享块；共享块只做固定前向，不再被任何任务更新。
2. detach_aux：在 pair/dense 中，只有当前 task_id 对应的主 workflow 参与反向传播；辅助 workflow 的最终输出 detach，不接收该任务 loss 的梯度。

因此：

- 共享块的存储仍然只有一份，唯一参数/存储收益保留；
- 不同任务的工作流在反向图上互相隔离；
- 共享块从“可训练共享参数”变成“固定共享特征 + 每任务自己的 block/adapter”；
- 这是严格的隔离，不是“共享块仍然由所有任务共同训练”的另一种正则化。

## 2. 精度：隔离 vs 耦合

同一 315-run 矩阵（pair+dense，3 宽度，3 相关性，5 seeds），隔离版相对耦合版的平均精度变化：

| 模式 | 模型 | 隔离 − 耦合（pp） | 范围（pp） |
|---|---|---:|---:|
| pair | A-Net Basic | +0.856 | +0.11 … +2.34 |
| pair | A-Net Share | +1.126 | +0.54 … +1.78 |
| pair | A-Net Share+Adapter | +1.493 | +0.31 … +2.99 |
| dense | A-Net Basic | +1.599 | +0.58 … +2.91 |
| dense | A-Net Share | +1.681 | +0.18 … +4.08 |
| dense | A-Net Share+Adapter | +1.750 | +0.52 … +3.21 |

- 所有 A-Net 变体在隔离后都变好；pair 提升约 0.9–1.5 pp，dense 提升约 1.6–1.8 pp。
- A-Net Basic 也提升，说明提升主要来自“辅助 workflow 不再接收其他任务梯度”（workflow 隔离），而不是只来自共享块冻结。
- 耦合版中 Share 在 h512 low seed2026 曾降到 90.57%；隔离版同配置为 95.17% 左右，collapse 消失。

## 3. 隔离后的绝对结果

pair，宽度 512，medium 相关性：

| 模型 | task0 binary acc | task1 regression MSE | task2 5-class acc | 平均 acc | 唯一参数 | 激活参数 |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 97.88 | 0.0008 | 94.23 | 96.05 | 805,896 | 268,290 |
| A-Net Basic | 97.43 | 0.0022 | 94.28 | 95.85 | 794,641 | 530,443 |
| A-Net Share | 97.45 | 0.0032 | 94.28 | 95.87 | 531,985 | 399,115 |
| A-Net Share+Adapter | 97.71 | 0.0026 | 94.31 | 96.01 | 545,065 | 407,835 |

跨全部 pair 组合相对 Independent：

| 模型 | 平均 acc 差（pp） | task0 acc 差（pp） | task1 MSE 差 | task2 acc 差（pp） |
|---|---:|---:|---:|---:|
| A-Net Basic | -0.10 | -0.25 | +0.0022 | +0.058 |
| A-Net Share | -0.07 | -0.22 | +0.0022 | +0.080 |
| A-Net Share+Adapter | -0.14 | -0.29 | +0.0020 | +0.019 |

- 隔离后 A-Net Share 与 Independent 的平均精度差只有 -0.07 pp，已经接近持平；task0 仍低约 0.22 pp，回归 MSE 仍高约 0.0022，但远小于耦合版的 +0.0106。
- Share 与 Basic 几乎同精度（95.87 vs 95.85，medium 512），唯一参数 0.656×，激活参数 1.48× Independent（pair 运行两个 workflow）。
- Adapter 在 medium 512 到 96.01%，但没有稳定超过 Share；适配器仍不是必需的。

dense，medium 512：

| 模型 | 平均 acc | 唯一参数 | 激活参数 |
|---|---:|---:|---:|
| A-Net Basic | 95.74 | 794,641 | 793,099 |
| A-Net Share | 95.81 | 531,985 | 530,443 |
| A-Net Share+Adapter | 95.83 | 545,065 | 543,523 |

## 4. 梯度审计的含义

隔离版中共享块的逐任务梯度全部为 0，因此：

- gradient_audit 的 mean cosine = 0，conflict fraction = 0；
- 这不是“冲突被解决”的测量结果，而是“共享块不再接收梯度”的构造结果；
- 共享块参数保持初始化值，不参与优化；优化器只更新 pre-L1、各 workflow 的 own block、adapter、terminal。

如果目标是“共享块仍由所有任务共同训练，但抑制冲突”，detach_both 不是那个方案；它选择了冻结共享表示来换取稳定性。下一阶段若需要可训练共享块，应使用 detach_aux 或梯度手术/PCGrad，而不是 detach_both。

## 5. 存储与显存

- 唯一参数比与耦合版相同：Share 0.656× Independent，Adapter 0.683×；隔离不改变存储结构。
- pair 激活参数：Share 1.479×、Basic 1.962× Independent；隔离不改变前向计算。
- 峰值显存：0.241 GB；所有输入 cuda:0；无 shared GPU memory 回退。

## 6. 结论

1. **隔离版数据更好。** 所有 A-Net 变体在 pair/dense 下都优于耦合版，pair 提升 0.86–1.49 pp，dense 提升 1.60–1.75 pp；Share 在 pair 下与 Independent 平均只差 -0.07 pp。
2. **提升主要来自 workflow 隔离，而非共享块本身。** A-Net Basic 也提升约 0.86/1.60 pp，说明辅助 workflow 不接收其他任务梯度是关键。
3. **共享存储收益保留。** Share 唯一参数仍为 0.656× Independent，Share 与 Basic 同精度；代价是共享块不再被训练。
4. **这不是“更好的共享学习”，而是“冻结共享特征 + 任务隔离”。** 若要求共享块可训练，必须单独验证 detach_aux/梯度手术，不能把 detach_both 的结论外推。
5. **默认策略：** 按你的要求，后续新实验默认使用 gradient_isolation=detach_both；如需可训练共享块，显式改为 none 或 detach_aux，并单独报告。

## 7. 产物

- 隔离版原始记录：experiments/anet_v2/results/stage2_isolated_both/raw_runs.jsonl
- 汇总：summary.csv / summary.json / summary.md
- 图：figures/
- 耦合版对照：experiments/anet_v2/results/stage2/
- 代码：core/models.py 的 gradient_isolation 参数；runners/run_stage2.py
