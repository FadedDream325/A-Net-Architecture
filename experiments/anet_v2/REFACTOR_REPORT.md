# Router / Cache 重构报告

- 分支：experiment/anet-v2
- 代码提交：70ab69d（重构）；MANIFEST 记录 code_hash
- 基准数据：experiments/anet_v2/results/refactor_benchmark/
- 环境：RTX 5060 Laptop，8 GB 专用显存，PyTorch 2.14.1+cu130
- 原则：旧 Stage 1 结果保留，不覆盖；本报告记录重构后的新测量。

## 1. 改了什么

### 1.1 Router：不再在 forward 里做任务预测/分支

- 构造时预计算 _active_table：sparse 为 (t,)，pair 为 (t, (t+1)%T)，dense 为全部 workflow。
- 预计算并直接持有 _prel1_block、_own_blocks[w]、_cross_blocks[w]、_adapter_blocks[w]；热路径不再按字符串 ID 查 ModuleDict，也不再每样本判断 active_mode。
- TerminalL1 的 heads 改为 nn.ModuleList，直接 heads[task_id]；K=1 时跳过 stack/softmax/pooling。
- 废除了每样本 if/else 路由；Task ID 已知时就是列表/张量索引。
- forward 仅保留一个顶层缓存分派判断；训练/评估状态在外部切换。

### 1.2 Dense / 混合 batch：argsort 分组，反排还原

forward_mixed 现在：

1. order = torch.argsort(task_ids, stable=True)；
2. 用 index_select 把同一 Task 的样本排成连续段；
3. 用 torch.unique_consecutive(..., return_counts=True) 得到组边界；
4. 逐组调用 forward（组内无 per-sample 分支）；
5. torch.cat 后按 inverse = torch.argsort(order) 反排回原顺序。

没有每样本 Python mask，也没有同一 Warp 内不同任务的 if 分支。

### 1.3 缓存：单次推理默认关闭

- 新增 ANetModel.set_inference_cache(enabled)。
- 默认 inference_cache=False：eval 时走 _forward_uncached，不构造 cache key、不做哈希、不做查找，只跑矩阵乘法和末端整合。
- inference_cache=True 时才启用 request-level 复用。
- benchmark/runner 新增 --inference-cache/--no-inference-cache，默认关闭。
- 训练阶段仍保留缓存语义：每个 optimizer step 清空；缓存张量保留 autograd 图。
- 缓存键仍支持 identity（默认）与 content（正确性测试/重复内容）两种模式。

### 1.4 专用显存守卫

新增 core/device_guard.py：

- 启动前要求 free dedicated VRAM >= 2.0 GiB，否则直接拒绝运行；
- 用 torch.cuda.set_per_process_memory_fraction(0.80) 把 PyTorch allocator 限制在专用显存总量的 80%，在 Windows 共享显存回退前先 OOM，而不是悄悄跑进 shared GPU memory；
- 每条 benchmark 记录 input_device、host_to_device_bytes、vram_before/after、gpu_peak_allocated_bytes；
- 所有计时输入在预热前一次性放到 cuda:0，计时循环内没有 H2D。

## 2. 显存证据

本次重构基准：

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，总专用显存 7.933 GiB；
- 运行前空闲 6.877 GiB，used 1.056 GiB（桌面应用）；
- 每个配置峰值 allocated <= 0.079 GiB；
- 所有记录 input_device = cuda:0，host_to_device_bytes = 0；
- allocator 上限 80% 专用显存；没有 shared GPU memory 回退。

## 3. 重构后基准（low 相关性，3 seeds，100 repeats）

### batch=1 P50（微秒，mean ± std）

| width | Independent | A-Net Basic | A-Net Full（纯路径） | A-Net Full（缓存开） |
|---:|---:|---:|---:|---:|
| 64 | 76.7 ± 5.6 | 229.7 ± 43.4 | 198.1 ± 5.3 | 241.4 ± 2.8 |
| 256 | 85.5 ± 2.6 | 188.5 ± 1.3 | 192.0 ± 2.1 | 237.3 ± 1.5 |
| 1024 | 82.8 ± 3.7 | 219.8 ± 7.9 | 325.5 ± 114.6 | 238.5 ± 1.9 |
| 2048 | 90.9 ± 17.5 | 228.0 ± 52.5 | 196.9 ± 1.3 | 239.4 ± 2.0 |

### batch=1024 吞吐（samples/s，mean）

| width | Independent | A-Net Basic | A-Net Full（纯路径） | A-Net Full（缓存开） |
|---:|---:|---:|---:|---:|
| 64 | 8.63M | 3.66M | 4.14M | 3.85M |
| 256 | 9.58M | 4.30M | 4.25M | 3.86M |
| 1024 | 8.15M | 4.34M | 3.30M | 3.85M |
| 2048 | 5.67M | 4.16M | 4.09M | 3.55M |

## 4. 与重构前的对比

重构前 Stage 1（medium，缓存默认开、旧 terminal、旧 router）的 batch=1 P50：

| width | Independent | A-Net Basic | A-Net Full |
|---:|---:|---:|---:|
| 64 | 86.3 | 215.7 | 557.4 |
| 2048 | 90.4 | 239.9 | 555.3 |

重构后（low，3 seeds，纯路径）：

| width | Independent | A-Net Basic | A-Net Full（纯） | A-Net Full（缓存开） |
|---:|---:|---:|---:|---:|
| 64 | 76.7 | 229.7 | 198.1 | 241.4 |
| 2048 | 90.9 | 228.0 | 196.9 | 239.4 |

关键变化：

- A-Net Full 从约 555 微秒降到约 198 微秒（batch=1，h64/h2048），约 2.8 倍改善；主要来自单次推理关闭缓存、terminal 快速路径和直接 block 引用。
- 缓存开仍有约 +20% 开销（198 到 241 微秒），但不再是重构前的数倍开销；缓存应在重复输入/多调用方场景使用，而不是单次请求默认开启。
- A-Net Basic 与 Full 纯路径基本同速；Full 的共享存储不再以单次推理的缓存开销为代价。
- A-Net 仍比 Independent 慢约 2.3 到 2.6 倍（batch=1）。原因不是 H2D，也不是 shared GPU memory，而是多次小 kernel launch、Python 调度和末端整合；batch=1024 时 A-Net 吞吐约为 Independent 的 0.4 到 0.7 倍，在 2048 宽度接近。
- Dense/mixed 路径使用 argsort 分组，消除了每样本 mask 与任务分支；正确性测试覆盖了 pair/dense 与逐样本循环的一致性。

## 5. 限制

- 重构基准的 low 相关性与旧 Stage 1 medium 不完全同分布；上表是工程延迟对比，不是精度对比。
- h1024 A-Net Full 纯路径的 325 ± 115 微秒是异常高方差；GPU 与桌面应用共享，batch=1 仍受 CPU/调度影响。
- 仍未使用 CUDA Graph / torch.compile / 融合算子；若要把 A-Net 的 batch=1 延迟压到 Independent 附近，需要这些静态图手段。
- 共享显存守卫是拒绝加 allocator 上限，不是操作系统级隔离；它确保实验不会故意超过专用显存预算。

## 6. 复现

    python -m experiments.anet_v2.runners.benchmark_router_cache \
      --widths 64 256 1024 2048 \
      --seeds 2026 2027 2028 \
      --out-dir experiments/anet_v2/results/refactor_benchmark

开关：

    --no-inference-cache   # 单次推理纯矩阵乘法（默认）
    --inference-cache      # 允许 request-level 缓存复用
