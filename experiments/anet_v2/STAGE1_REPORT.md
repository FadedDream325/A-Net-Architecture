# A-Net v2 阶段 1 实验报告

- 分支：experiment/anet-v2
- 代码提交：99fca77（阶段 1 结果记录的 code_hash 对应该提交的 v2 源码与测试）
- 运行日期：2026-10-30
- 数据：experiments/anet_v2/results/stage1/raw_runs.jsonl
- 汇总：experiments/anet_v2/results/stage1/summary.csv / summary.json / summary.md
- 原始延迟：同一 raw 记录中的 raw_us 数组，另有 latency_summary.csv
- 结论口径：本报告只讨论阶段 1（sparse 模式、单工作流）。Stage 2/3 尚未运行。

## 1. 本阶段回答什么

阶段 1 要回答的是：在新的 A-Net v2 定义下，基础工作流结构是否可训练、是否随宽度扩展、与 Independent / Shared Backbone 相比精度、存储、计算和工程性能如何。

本阶段不启用多工作流同时激活，也不测试跨工作流参数共享与结果复用；这两项分别在阶段 2/3 中验证。

## 2. 实现范围

- ParameterBlock / ParameterRegistry：唯一所有者、唯一存储、显式授权、参数版本。
- Workflow / WorkflowGraph：固定调用图、拓扑排序、环检测、前置 L1、末端 L1。
- TerminalL1：只接收各 workflow 的最终输出；单工作流时有快速路径，避免无意义的 stack/softmax。
- ResultCache：按 block 身份、版本、输入键、运算配置缓存；训练每个 optimizer step 清空；缓存张量保留 autograd 图。
- 五口径 accounting：reference / unique / active / theoretical FLOPs / actual executions & FLOPs。
- 六类模型：Independent、Shared、A-Net Basic、A-Net Share、A-Net Reuse、A-Net Full，另有 adapter 变体。
- 数据：二分类、回归、五分类；低/中/高三档相关性；固定 train/val/test 划分与 SHA256。
- 测试：58 passed（14 个旧测试 + 44 个 v2 测试）。
- 阶段 1 只使用 sparse 模式，四个模型：Independent、Shared、A-Net Basic、A-Net Full。

## 3. 实验配置

| 项目 | 值 |
|---|---|
| 宽度 | 64, 128, 256, 512, 1024, 2048 |
| 模型 | Independent, Shared, A-Net Basic, A-Net Full |
| 相关性 | low (rho=0.20), medium (0.70), high (0.95) |
| seeds | 2026, 2027, 2028, 2029, 2030 |
| epochs | 30 |
| train / val / test | 6000 / 1500 / 1500 每任务 |
| batch | 512 |
| optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| 精度 | FP32 |
| active mode | sparse（每样本 1 个 workflow） |
| 推理 batch | 1, 64, 256, 1024 |
| 重复次数 | 100 |
| runs | 360 |
| 失败 | 0 |

数据在计时前已移动到 CUDA；所有 Stage 1 记录的 input_device 为 cuda:0，host_to_device_bytes 为 0。吞吐使用真实墙钟时间，不使用事件时长之和。

## 4. 精度结果

### 4.1 中等相关性，mean ± std（5 seeds）

| width | Independent | Shared | A-Net Basic | A-Net Full |
|---:|---|---|---|---|
| 64 | 94.83 ± 1.07 | 93.86 ± 1.06 | 94.63 ± 1.46 | 94.59 ± 1.46 |
| 128 | 95.49 ± 1.02 | 93.93 ± 1.52 | 95.49 ± 0.89 | 94.79 ± 1.08 |
| 256 | 96.09 ± 0.81 | 94.01 ± 2.14 | 95.81 ± 0.28 | 95.68 ± 0.55 |
| 512 | 96.05 ± 0.42 | 94.15 ± 2.21 | 95.74 ± 0.43 | 95.65 ± 0.56 |
| 1024 | 95.73 ± 0.74 | 94.67 ± 1.01 | 95.83 ± 0.47 | 95.85 ± 0.69 |
| 2048 | 95.70 ± 0.33 | 94.85 ± 0.75 | 95.47 ± 0.19 | 95.46 ± 0.24 |

跨全部 18 个 (width, correlation) 组合，相对 Independent 的平均差：

| 模型 | 平均差 (pp) | 范围 (pp) |
|---|---:|---:|
| Shared | -1.367 | -2.24 … -0.45 |
| A-Net Basic | -0.067 | -0.45 … +0.24 |
| A-Net Full | -0.216 | -0.81 … +0.16 |

观察：

- A-Net Basic 基本与 Independent 持平（-0.07 pp），在部分宽度/相关性子集上略高。
- A-Net Full 平均低 0.22 pp，主要来自小宽度；在 1024/2048 附近接近 Independent。
- Shared Backbone 在所有宽度和相关性下都明显更低（平均 -1.37 pp），且方差更大；共享主干 + 线性头在本任务集合上容量不足。
- 5 seeds 下这些差异没有做正式显著性检验；不应把 0.1–0.3 pp 的差异当成确定结论。

### 4.2 相关性影响

相对 Independent 的平均差（pp，跨 6 个宽度）：

| 相关性 | Shared | A-Net Basic | A-Net Full |
|---|---:|---:|---:|
| low | -1.349 | -0.132 | -0.323 |
| medium | -1.404 | -0.154 | -0.312 |
| high | -1.347 | +0.084 | -0.012 |

- 高相关性对 A-Net 有轻微帮助（Basic 从 -0.13 到 +0.08，Full 从 -0.32 到 -0.01）。
- Shared Backbone 并没有因为任务更相关而追上；其瓶颈更像是任务头容量，而不是共享程度。
- 相关性由共享 latent 的实测平均绝对值给出：low ≈ 0.11，medium ≈ 0.52，high ≈ 0.91。

## 5. 参数、存储与计算

### 5.1 唯一参数与激活参数

跨全部组合相对 Independent：

| 模型 | 唯一参数比 | 激活参数比 |
|---|---:|---:|
| Shared | 0.341 | 1.028 |
| A-Net Basic | 0.967 | 0.996 |
| A-Net Full | 0.651 | 0.996 |

- A-Net Basic 的唯一参数几乎与 Independent 相同（0.967），因为每个 workflow 的 own + cross 两个 block 合计约等于一个同宽度 MLP；它没有真正的参数存储优势。
- A-Net Full 的唯一参数约为 Independent 的 65%，这是共享 cross block 替换重复块的结果；在 2048 宽度下为 8,419,345 vs 12,660,744（0.665）。
- 三个模型的激活参数几乎相同（A-Net 0.996），说明 sparse 模式下共享存储没有降低单次前向的激活参数或理论计算量。这是本阶段最重要的区分：**唯一参数下降不等于实际计算下降**。
- A-Net 的 adapter 变体在阶段 1 未运行；阶段 2 单独统计。

### 5.2 存储与计算口径

- checkpoint 未保存；模型存储可按唯一参数 × 4 bytes 估算：2048 宽度下 Independent ≈ 50.6 MB，A-Net Full ≈ 33.7 MB，Shared ≈ 16.9 MB。
- 理论 FLOPs 与 Independent 基本相同（激活参数比 0.996），不是旧 A-Net 那种“约一半计算”的结构。
- 实际执行次数在 sparse 模式下等于理论调用数；A-Net Full 的 cache 没有跨调用方可复用，因此没有实际计算节省。

## 6. 工程性能

### 6.1 batch=1 延迟（P50，微秒，5 seeds 均值）

| width | Independent | Shared | A-Net Basic | A-Net Full |
|---:|---:|---:|---:|---:|
| 64 | 86.3 | 86.4 | 215.7 | 557.4 |
| 128 | 82.3 | 86.7 | 316.6 | 489.6 |
| 256 | 186.1 | 148.3 | 290.9 | 525.9 |
| 512 | 117.9 | 113.3 | 427.1 | 496.7 |
| 1024 | 108.1 | 95.8 | 229.5 | 549.0 |
| 2048 | 90.4 | 92.7 | 239.9 | 555.3 |

相对 Independent：

| 模型 | P50 比值均值 | 范围 | P95 比值均值 |
|---|---:|---:|---:|
| Shared | 1.11 | 0.78 … 2.17 | 1.20 |
| A-Net Basic | 2.65 | 1.56 … 3.85 | 2.92 |
| A-Net Full | 4.75 | 2.83 … 6.46 | 5.19 |

### 6.2 batch=1024 吞吐（samples/s，5 seeds 均值）

| width | Independent | Shared | A-Net Basic | A-Net Full |
|---:|---:|---:|---:|---:|
| 64 | 8.72M | 9.30M | 4.21M | 1.99M |
| 256 | 7.45M | 10.46M | 3.52M | 2.02M |
| 1024 | 2.64M | 2.54M | 2.25M | 2.08M |
| 2048 | 0.742M | 0.742M | 0.749M | 0.745M |

相对 Independent：A-Net Basic 平均 0.61×（0.33–1.01），A-Net Full 平均 0.47×（0.21–1.00）。在 2048 宽度，三者吞吐几乎相同；小宽度下 A-Net 的多次小 kernel 与 Python 路径占主导。

### 6.3 为什么 A-Net 慢：数据已在显存

审查中专门修正了输入设备判断：此前的设备比较把 cuda 与 cuda:0 视为不同，导致基准报告了 32 bytes 的一次性传输；计时循环本身从未做主机传输。修正后：

- 所有 Stage 1 记录 input_device = cuda:0；
- host_to_device_bytes = 0；
- 新增测试 test_benchmark_gpu.py，断言计时输入在 CUDA 上且无 H2D。

在显存常驻、零 H2D 的直接测量中（h=64/2048，batch=1，100 repeats）：

| width | Independent | A-Net Basic | A-Net Full |
|---:|---:|---:|---:|
| 64 | 75–80 us | 188–311 us | 363–519 us |
| 2048 | 86 us | 191 us | 373 us |

因此延迟差不是“数据在内存里”。原因是：batch=1 时 3–7 个很小的 CUDA kernel 的 launch/Python 调度占主导；A-Net Full 还额外承担缓存键、查找、参数版本和缓存管理的开销。要真正降低这部分，需要 CUDA Graph、torch.compile、融合算子或静态图部署；这不是阶段 1 的数据搬运问题。

### 6.4 训练时间与内存

跨全部组合相对 Independent：A-Net Basic 训练时间 1.21×，A-Net Full 1.33×；Shared 0.98×。

中等相关性、2048 宽度：

| 模型 | 训练 GPU 峰值 (GB) | 训练 CPU RSS (GB) |
|---|---:|---:|
| Independent | 0.239 | 1.66 |
| Shared | 0.156 | 1.63 |
| A-Net Basic | 0.238 | 1.66 |
| A-Net Full | 0.199 | 1.64 |

A-Net Full 的 GPU 峰值低于 Independent（共享块只存一份），但 CPU RSS 基本相同（数据与框架开销占主导）。

## 7. 阶段 1 结论

1. **基础工作流结构可训练、可评估、可扩展。** 360 个 run 全部成功，6 个宽度、3 档相关性、5 seeds 均完成；没有 NaN/Inf/OOM 失败。
2. **A-Net Basic 的精度与 Independent 基本持平（-0.07 pp 平均），但存储几乎不省（0.967×），单次激活计算相同（0.996×）。** 它不是一个“更省参数/更省计算”的结构，而是一个精度可比的模块化替代结构。
3. **A-Net Full 通过共享 cross block 把唯一参数降到约 0.65× Independent，但激活计算仍为 0.996×。** 在 sparse 模式下，共享没有转化成计算节省，因为每个样本只调用一个 workflow，共享块没有重复调用可复用。
4. **工程性能目前是 A-Net 的弱项。** batch=1 延迟为 Independent 的 2.65×（Basic）/4.75×（Full），batch=1024 吞吐为 0.61×/0.47×；只有 2048 宽度大 batch 时吞吐追平。数据已在显存，差异来自 kernel launch、Python 调度和缓存开销。
5. **Shared Backbone 精度明显更低（-1.37 pp），但延迟/吞吐与 Independent 接近，存储最省。** 在本任务集合上，它没有因为任务相关性提高而追平。
6. **高相关性对 A-Net 略有帮助，对 Shared 没有明显帮助。** 相关性不是当前 Shared 的主要瓶颈；任务头容量/共享主干表达能力更可能。
7. **结果支持“机制正确性”和“存储收益有条件成立”，不支持“计算或延迟收益”。** 结果复用与跨工作流共享的计算收益必须在阶段 3 的多工作流/重复输入场景中单独验证。

## 8. 局限

- 只有合成数据（二分类/回归/五分类），没有真实图像、文本或表格任务。
- 30 epochs、固定超参；没有 early stopping 或超参搜索。
- 5 seeds，未做正式显著性检验；0.1–0.3 pp 的差异应视为噪声范围内。
- 只在 RTX 5060 Laptop 单卡、FP32、eager PyTorch 下测量；GPU 与桌面应用共享。
- Stage 1 只跑 sparse 模式；cache 没有可命中的重复共享调用，因此 Full 的缓存开销只体现为成本，没有体现为收益。
- A-Net 的 own+cross 双 block 设计使 Basic 的激活计算与 Independent 接近，这是设计选择；阶段 2 会测试不同共享/适配器结构。
- 延迟的 batch=1 结果受 CPU 调度与 GPU 负载影响，P95 波动较大；报告给出均值和范围，不把单点当结论。
- 未保存 checkpoint；冷启动/重载不在本阶段范围内。

## 9. 下一步

- 阶段 2：pair/dense 模式，no-share / direct-share / share+adapter 三组；验证对象身份、唯一存储、梯度累积、梯度冲突和负迁移。
- 阶段 3：0/25/50/75% 输入重复率；缓存命中、失效、梯度、查找开销、内存与延迟；区分 identity 键与 content 键。
- 阶段 4：2×2（无共享/仅共享/仅复用/共享+复用）× 相关性 × 宽度 × workflow 数；统计与报告。

## 10. 产物索引

| 文件 | 内容 |
|---|---|
| experiments/anet_v2/results/stage1/raw_runs.jsonl | 360 条原始记录（含逐次延迟样本、loss curve、config、代码哈希） |
| experiments/anet_v2/results/stage1/summary.csv | 72 个 (width, correlation, model) 组汇总 |
| experiments/anet_v2/results/stage1/summary.json | 汇总与失败列表 |
| experiments/anet_v2/results/stage1/summary.md | 可读表格 |
| experiments/anet_v2/results/stage1/latency_summary.csv | 每个 run/mode/batch 的 P50/P95/吞吐/缓存 |
| experiments/anet_v2/results/stage1/figures/ | accuracy/active params/latency/throughput 图 |
| experiments/anet_v2/results/stage1/aggregate.log | 聚合日志 |
| tests/test_*.py | 58 个测试 |
