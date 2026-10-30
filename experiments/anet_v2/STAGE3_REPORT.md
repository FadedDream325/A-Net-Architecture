# A-Net v2 阶段 3 实验报告：结果缓存

- 分支：experiment/anet-v2
- 代码提交：7b92279（Stage 3 runner/benchmark）
- 数据：experiments/anet_v2/results/stage3/raw_bench.jsonl
- 汇总：summary.csv / summary.json / summary.md
- 规模：288 个成功 stream，0 失败
- 环境：RTX 5060 Laptop，8 GB 专用显存；显存守卫开启
- 默认策略：gradient_isolation=detach_both；缓存键默认 identity

## 1. 设计

- 宽度：128、512、2048。
- seeds：2026、2027、2028。
- 模型：
  - A-Net Reuse：无跨 workflow 共享，只有 request-level 缓存；
  - A-Net Full：共享 cross block + 缓存。
- 并发模式：pair（每样本 2 个 workflow）。
- 输入重复率：0%、25%、50%、75%。
- batch：1、256；每个 stream 200 次请求。
- 缓存开/关使用同一模型、同一权重；cache off 是纯矩阵乘法控制。
- 本阶段使用固定随机初始化权重；目标是缓存正确性和工程性能，不是任务精度。
- 所有输入在计时前放到 cuda:0；计时循环内 H2D = 0。

重复模式：

- 0%：每次使用不同的 pool 索引；
- 25%：每 4 次有 1 次复用上一次输入；
- 50%：每 2 次有 1 次复用；
- 75%：每 4 次有 3 次复用。

## 2. 正确性

288/288 个 stream：

- cache off vs cache on 的输出最大绝对差：0.0；
- 参数更新后 cache 失效：全部 True；
- H2D：0；
- 所有 stream 的 input_device = cuda:0。

缓存的正确性测试（单元测试）另外覆盖：

- 同 block/同输入/同配置命中；
- 不同输入/不同 dtype/不同 train-eval/Dropout 状态不命中；
- 参数版本更新后旧 entry 失效；
- 两个调用方共享缓存结果时梯度正确；
- 训练每个 optimizer step 清空缓存；
- 缓存键 identity/content 两种模式。

## 3. 命中率与执行次数

batch=1（3 个宽度平均）：

| 模型 | 重复率 | 命中率 | 实际/理论执行 | P50 比（cache on/off） |
|---|---:|---:|---:|---:|
| Reuse | 0% | 0.00 | 1.00 | 1.68 |
| Reuse | 25% | 0.25 | 0.75 | 1.70 |
| Reuse | 50% | 0.50 | 0.50 | 1.64 |
| Reuse | 75% | 0.75 | 0.25 | 1.21 |
| Full | 0% | 0.20 | 0.80 | 1.72 |
| Full | 25% | 0.40 | 0.60 | 1.64 |
| Full | 50% | 0.60 | 0.40 | 1.54 |
| Full | 75% | 0.80 | 0.20 | 1.20 |

batch=256（3 个宽度平均）：

| 模型 | 重复率 | 命中率 | 实际/理论执行 | P50 比 |
|---|---:|---:|---:|---:|
| Reuse | 0% | 0.00 | 1.00 | 1.67 |
| Reuse | 25% | 0.25 | 0.75 | 1.52 |
| Reuse | 50% | 0.50 | 0.50 | 1.47 |
| Reuse | 75% | 0.75 | 0.25 | 0.98 |
| Full | 0% | 0.20 | 0.80 | 1.53 |
| Full | 25% | 0.40 | 0.60 | 1.39 |
| Full | 50% | 0.60 | 0.40 | 1.34 |
| Full | 75% | 0.80 | 0.20 | 1.02 |

- 命中率精确等于输入重复率（Reuse），Full 额外多 20%：pair 内两个 workflow 调用同一 shared block，第一次 miss、第二次 hit。
- 执行次数确实下降：75% 重复率下 Reuse 只剩 25%，Full 只剩 20%。
- 但执行次数下降没有自动转化为延迟下降。

## 4. 延迟与收益边界

### 4.1 batch=1

所有配置的 cache-on 都比 cache-off 慢：

- 0–50% 重复率：P50 比 1.5–1.8；
- 75% 重复率：仍约 1.2。

原因：batch=1 时每个 block 的一次 miss 也要执行 cache key 构造、dict 查找、store 和统计；这些 Python/CPU 开销大于一次小 kernel 的节省。

### 4.2 batch=256

只有高宽度 + 高重复率出现净收益：

| 宽度 | 模型 | 重复率 | 命中率 | P50 比 |
|---:|---|---:|---:|---:|
| 2048 | Reuse | 75% | 0.75 | 0.67 |
| 2048 | Full | 75% | 0.80 | 0.76 |
| 2048 | Full | 50% | 0.60 | 0.96 |
| 2048 | Reuse | 50% | 0.50 | 1.11 |
| 512 | Full | 75% | 0.80 | 1.14 |
| 128 | Full | 75% | 0.80 | 1.17 |

结论：缓存的收益边界是**大宽度、大 batch、高重复率**；在 128/512 宽度或低重复率下，查找/存储开销抵消甚至超过计算节省。

### 4.3 缓存内存与查找成本

- 查找平均约 1.33 µs/次。
- batch=1 时缓存内存最大约 4.9 MB。
- batch=256、2048 宽度时达到配置上限 256 MB，发生 eviction：
  - Reuse 最多 786 次 eviction；
  - Full 最多 595 次 eviction。
- 因此高重复率下的收益不是无成本的：缓存本身占用最高 256 MB，并且存在淘汰。
- identity 键不需要张量内容哈希；content 键未用于本阶段性能主表（它会把张量搬到 CPU 做哈希，成本更高，只保留在正确性测试中）。

## 5. 结论

1. **缓存正确性成立。** 288/288 stream 的 on/off 输出完全相同，参数更新后全部失效，H2D=0。
2. **缓存确实减少重复执行。** 75% 重复率下执行次数降到 20–25%；Full 在 0% 重复率也有 20% 的 pair 内复用。
3. **缓存不是延迟收益的充分条件。** batch=1 全部更慢；batch=256 只有在 2048 宽度、50–75% 重复率时才有净收益（P50 比 0.67–0.96）。
4. **缓存开销可测且不可忽略。** 查找约 1.33 µs/次，batch=256 下缓存内存可达 256 MB 上限并发生淘汰。
5. **与 Stage 2 的组合含义：** 参数共享降低唯一存储，缓存降低重复执行；两者都需要在“重复输入/多调用方”场景才可能转化为延迟收益。低重复率/小 batch 下应默认关闭缓存（本项目的 inference_cache 默认关闭）。

## 6. 局限

- 缓存实验使用随机初始化权重；不评估缓存对任务精度的影响（on/off 输出一致性已单独验证）。
- 只测了 pair 模式；dense 会增大单次 forward 内的重复调用，但也会放大缓存键/内存开销，需单独测。
- 只测了 identity 键；content 键的内容哈希成本更高，未作为性能默认。
- 缓存内存上限 256 MB 是配置值，未做更大缓存容量/淘汰策略的消融。
- 单 GPU、FP32、eager PyTorch；CUDA Graph/静态图可能改变收益边界。

## 7. 产物

| 文件 | 内容 |
|---|---|
| experiments/anet_v2/results/stage3/raw_bench.jsonl | 288 条 stream 原始记录（含 raw_us、cache stats） |
| experiments/anet_v2/results/stage3/summary.csv | 96 个组汇总 |
| experiments/anet_v2/results/stage3/summary.md | 可读表格 |
| experiments/anet_v2/results/stage3/figures/ | 命中率、延迟比图 |
| experiments/anet_v2/core/cache_benchmark.py | 请求流缓存基准 |
| experiments/anet_v2/runners/run_stage3.py | Stage 3 runner |
| tests/test_cache_stream.py | 重复索引与缓存流测试 |
