# Phase 0：50M 语言栈烟雾与审计报告

- 分支：experiment/anet-v2
- 记录时提交：607354a
- 设备：RTX 5060 Laptop，7.93 GB 专用显存；15.18 GB RAM
- 脚本：experiments/anet_v2/lang50m/phase0.py
- 结果：experiments/anet_v2/lang50m/results/phase0/phase0_results.json
- 日志：experiments/anet_v2/lang50m/phase0.log

## 1. 门禁结果

| 检查 | 结果 |
|---|---|
| 前向输出形状 | **通过**：所有模型 [2, 64, 256] |
| 反向梯度 | **通过**：所有可训练参数梯度有限 |
| checkpoint 保存/加载 | **通过**：最大绝对差 0.0 |
| 固定 train/val 划分 | **通过**：256-byte 重叠 0 |
| 50M 模型显存 | **通过**：峰值 0.76–1.19 GB |
| 失败/异常 | 0 |

说明：code 语料存在 33 个 32-byte 公共惯用片段（如常见 Python 语法），但 256-byte 精确重叠为 0；这属于代码惯用片段，不是训练/验证样本泄漏。

## 2. 参数、激活参数与 FLOPs

| 架构 | 唯一参数 | 单次激活参数 | 激活 FLOPs/token | 激活 block | 唯一 block |
|---|---:|---:|---:|---:|---:|
| Transformer-12L | 37,503,168 | 37,503,168 | 39,018,496 | 12 | 12 |
| Shared-Depth A-Net | 50,177,024 | 37,501,632 | 39,018,496 | 12 | 16 |
| Sparse A-Net | 50,177,033 | 37,501,632 | 39,018,496 | 12 | 16 |
| Shared Backbone | 50,178,560 | 50,178,560 | 51,937,280 | 16 | 16 |

- Shared-Depth 与 Transformer-12L 的**激活参数、深度、激活 FLOPs 完全对齐**（1,536 参数差）。
- Sparse A-Net 唯一参数 50M，但有效路径只有 6 层；不能与 12 层 Transformer 直接比较。
- FLOPs 是按 CausalBlock 结构解析估算的 per-token 数量，包含 attention score/value 项；不是 profiler 实测，也不是整次训练总 FLOPs。

## 3. checkpoint

| 架构 | checkpoint 大小 | 加载后输出差 |
|---|---:|---:|
| Transformer-12L | 150.1 MB | 0.0 |
| Shared-Depth A-Net | 200.8 MB | 0.0 |
| Sparse A-Net | 200.8 MB | 0.0 |
| Shared Backbone | 200.8 MB | 0.0 |

checkpoint 仅用于验证，测试后已删除；没有把 800 MB checkpoint 提交到仓库。

## 4. 数据划分

- 每域 200,000 bytes；train 180,000 / val 20,000。
- 三域均通过 train_last_index < val_first_index，256-byte 重叠为 0。
- 这是 Phase 0 的临时数据规模，不是 Phase 1 的 30M + 30M tokens。

## 5. 观察到的问题

1. Transformer-12L 首次计时约 15.5 秒，其他模型约 0.05 秒；这是首次 CUDA kernel/allocator 预热，不是训练速度差异。Phase 1 必须给每个模型加 warmup 后再计时。
2. 只有 1 个 optimizer step 的 BPC 没有能力解释力；本阶段只验证管线，不能用于架构结论。
3. 当前模型没有 Lα 特征分解、没有可选 Lω、没有数学/编程任务准确率；Phase 0 只验证现有 Shared-Depth 语言栈。
4. FLOPs 是解析估算；需要与 torch profiler/公式交叉验证后再写入正式结果。
5. 当前没有 30M + 30M tokens 数据管线；Phase 1 需要新建。

## 6. 结论

Phase 0 门禁通过：

- 前向/反向/保存/加载/评估均可运行；
- 50M 模型在 8 GB GPU 上训练状态峰值约 1.19 GB，资源足够；
- Shared-Depth A-Net 与 Transformer-12L 的激活参数/深度/激活 FLOPs 可比。

可以进入 Phase 1，但必须先完成：

1. 30M 数学 + 30M 编程 tokens 数据池；
2. 固定 train/val/test 与防泄漏；
3. 5 种领域比例、batch 内混合/边界切换；
4. 数学/编程准确率与 BPC 评测；
5. 训练计时 warmup 与 checkpoint 周期保存。

## 7. 推荐的 Phase 1 首步

先做**数据管线 smoke**，不直接开 60M 全矩阵：

- 生成 30M math + 30M code 池；
- 验证 5 个比例的实际 token 计数；
- 验证 batch 内混合与 batch 间切换的 batch 形状；
- 验证 train/val/test 文件级隔离；
- 再跑 6M-token/模型的轻量 pilot。
