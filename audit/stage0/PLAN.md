# A-Net v2 分阶段实施计划

- 依据：audit/stage0/REPO_AUDIT.md
- 目标：验证提示词中的 A–E，而不是给旧“稀疏专家路由”补结论
- 原则：旧代码/旧结果/旧报告不覆盖；v2 新目录、新结果根、新分支；先机制正确性，再规模与性能

## 0. 边界与分支

- 新建分支：experiment/anet-v2
- 新代码根：experiments/anet_v2/
- 新结果根：experiments/anet_v2/results/
- 旧文件只读：multitask_anet_demo.py、experiments/scaling/、旧 results、PDF、ZIP、旧 tests
- 需要改旧 README 时只追加 v2 链接，不删旧内容
- 历史 17% 口径已由 ERRATA.md 显式勘误，PDF 不修改

## 1. v2 架构规范

### 1.1 核心对象

1. ParameterBlock
   - 唯一 block_id；owner_workflow_id；输入/输出接口；参数量；操作配置；
   - 单调递增 version；params_version（可由 Tensor._version 校验）；
   - forward(x, config)；
   - cacheable / stateful 标记（Dropout、BatchNorm 等）。
2. ParameterRegistry
   - 只注册一次同一对象；禁止 deepcopy 注册；
   - grants: block_id -> set(workflow_id)，区分 owner 与 user；
   - 提供 owner_of、can_call、iter_blocks、unique_parameter_count、reference_count。
3. Workflow
   - 拥有若干 ParameterBlock；
   - 固定 call plan（v1 不学习）；
   - 可通过 registry 调用其他 workflow 的 block；
   - 输出只暴露最终 tensor，不暴露中间 hidden state。
4. WorkflowGraph
   - 节点 = block call site；边 = tensor 依赖；
   - 构建时做拓扑排序与环检测；
   - 记录每个 call site 的 block_id、caller、输入来源、config。
5. PreL1
   - 输入编码 + 固定分配（task_id -> 激活的 workflow 集合）；
   - v1 支持 sparse(K=1) 与 multi(K=T) 两种模式。
6. TerminalL1
   - 只接收各 workflow 的最终输出（list/tuple 或拼接张量）；
   - 不读取 workflow 内部 hidden state；
   - 输出逐任务 logits/回归值；
   - 测试用一个 sentinel wrapper 证明它拿不到中间状态。
7. ResultCache
   - key = (block_id, block_version, input_fingerprint, config_fingerprint)；
   - input_fingerprint：默认对 tensor 内容做 blake2b（记录哈希耗时）；可选 input_id 预计算键；
   - config_fingerprint：dtype、device、training/eval、autocast、deterministic、dropout p、BN state token、op version；
   - training：每个 optimizer step 前清空；只在同一次 forward 内复用；禁止跨 step；
   - eval：可在 request 内/跨 request 复用，按配置；
   - stats：hits、misses、lookups、hash_us、lookup_us、evictions、bytes、executions。
8. Accounting
   - references：所有 call site 引用参数之和（重复计）；
   - unique：不同参数元素；
   - active：一次 forward 中实际参与的不同参数元素（缓存命中仍算“被使用”，但不重复执行）；
   - theoretical_flops：未缓存、按调用图的解析量；
   - actual_executions：每个 block 实际 forward 次数；
   - actual_flops：按实际执行次数累计；
   - adapter_unique / adapter_flops / adapter_executions 单独字段；
   - 所有字段写入 run 记录，不只写汇总。

### 1.2 关键语义

- 共享必须是“替换重复块”，不是额外加块：
  - no-share：每个 workflow 拥有自己的 S_t；
  - direct-share：所有 workflow 调用 S_0，S_0 只注册一次；
  - adapter：S_0 + 每个 workflow 一个小适配器 A_t；A_t 参数单独统计。
- 共享调用点必须让“同一 block + 同一输入 + 同一配置”真实出现，否则 cache 没有命中机会：
  - 所有激活 workflow 在 PreL1 的同一 h0 上调用 S；
  - 各 workflow 再对自己的输出做特化。
- 2×2 因子：
  - C = Basic：无跨 workflow 调用、无 cache；
  - D = Share：有跨 workflow 调用、无 cache；
  - E = Reuse：无跨 workflow 调用、有 cache（重复输入驱动命中）；
  - F = Full：有跨 workflow 调用、有 cache。
- 对照：A Independent；B Shared Backbone；另加 Share+Adapter。
- v1 固定调用图；不实现学习路由/自动发现共享。

## 2. 任务与数据

### 2.1 三类任务

- 非线性二分类：2D/8D 非线性边界（sine/checker/spiral 变体），accuracy + F1；
- 非线性回归：连续目标（sin/cos/径向函数），MSE/MAE/R²；
- 多分类：5 类非线性分区，accuracy + macro-F1。

每个 suite 含 3 个任务（每类一个），任务之间共享底层 latent 的程度由 rho 控制。

### 2.2 相关性三档

- high：rho = 0.9；
- medium：rho = 0.5；
- low：rho = 0.1。

生成方式（建议）：

```text
z_shared = f_shared(x)                  # 共享非线性底层特征
z_t      = f_t(x)                       # 任务特化特征
z_t'     = rho * z_shared + sqrt(1-rho^2) * z_t
y_t      = h_t(z_t')                    # 分类阈值 / 回归函数 / 多类 softmax
```

- 固定数据种子、模型种子、优化器种子，分开记录；
- train/val/test = 70/15/15；测试集只在最终评估使用；
- 每个任务/每个 suite 保存数组 SHA256 与统计量；
- 量化相关性：报告 pairwise CKA / 标签一致率 / 目标相关系数，不只写“高/中/低”。

### 2.3 规模与预算

- 宽度：64、128、256、512、1024、2048；
- 每个宽度定义各架构的实际参数量，并给出参数匹配/FLOPs 匹配的控制组；
- 主实验 5 seeds：2026–2030；
- 统一 AdamW、lr、weight decay、epoch、batch、FP32；不允许按架构调参；
- 数据预加载，训练时间排除 I/O；GPU/CPU 峰值单独记录；
- 失败规模（OOM/NaN）必须记录并报告，不隐瞒。

## 3. 实验矩阵

### 阶段 1：基础工作流与规模

目的：验证 A + 建立 scaling 曲线。

- 宽度：6 档；
- 架构：Independent、Shared、ANetBasic、ANetFull；
- 相关性：high/medium/low；
- seeds：5；
- runs：6 × 4 × 3 × 5 = 360；
- 额外：ANetShare / ANetReuse 在 128/512/2048 三档 × 3 相关性 × 5 seeds = 90；
- 合计约 450 个训练 run。

测量：

- 前向、反向、参数更新；
- 单 workflow 推理（K=1）、多 workflow 推理（K=T）、混合任务 batch；
- GPU 显存峰值、CPU RSS 峰值；
- 训练耗时、推理 P50/P95、吞吐；
- 唯一参数、references、active、理论 FLOPs、实际 executions、实际 FLOPs；
- 逐 run 原始延迟样本。

### 阶段 2：跨 workflow 参数块调用

目的：验证 B，尤其是“唯一参数存储是否下降”。

- 对照：no-share / direct-share / share+adapter；
- 宽度：128/512/2048；
- 相关性：3 档；
- seeds：5；
- 新增 runs：3 变体 × 3 宽度 × 3 相关性 × 5 = 135（与阶段 1 重叠的 no-share 可复用）。

测量：

- id(block)、params data_ptr、state_dict key 数；
- unique / references / active；
- 多调用方梯度 = 各自梯度之和；
- 参数更新后所有调用方看到同一版本；
- 共享块梯度余弦相似度、冲突率、任务间负迁移；
- 相对 Independent 的逐任务性能差；
- adapter 参数量/FLOPs 单独列出。

### 阶段 3：结果复用

目的：验证 C，正确性优先于性能。

- 输入重复率：0%、25%、50%、75%；
- cache on/off；
- 模式：eval request 内、eval 跨 request；training 只在同一次 forward 内；
- 宽度：128/512/2048；
- 相关性：3 档；
- seeds：5；
- 推理 run 可复用阶段 1/2 的 checkpoint，成本低。

测量：

- hit rate、miss rate；
- 每个 block 实际执行次数；
- 理论 FLOPs、实际 FLOPs、节省比例；
- 缓存查找耗时、哈希耗时、缓存内存、eviction；
- 总延迟、吞吐、P50/P95；
- 断点：当 hash+lookup 开销 > 节省的计算时，明确报告“无收益”。

正确性测试必须先于性能测试：

- 同 block/同输入/同配置 -> 命中且数值一致；
- 不同输入 -> 不命中；
- 不同 dtype/精度/device/op config -> 不命中；
- train/eval、Dropout、BN 状态 -> 不命中；
- 参数更新 -> 旧 entry 失效；
- 两个调用方共享 cache 结果 -> 梯度都正确到达；
- 训练跨 optimizer step 不得复用。

### 阶段 4：综合实验

目的：验证 D/E。

- 2×2：无共享/仅共享/仅复用/共享+复用；
- 相关性：3 档；
- 宽度：3 档（128/512/2048）；
- workflow 数：2/3/4；
- 输入重复率：0/25/50/75%（推理侧）；
- seeds：5。

核心组合先跑 4 模式 × 3 相关性 × 3 宽度 × 3 workflow 数 × 5 seeds = 540 个训练/评估 run；重复率实验在最优/最差配置上做 4 档推理，避免无意义的全交叉。

### 阶段 5：报告

- 原始数据、summary、图、失败日志；
- 逐项回答提示词 10 个问题；
- 按“机制正确性/存储收益/计算收益/工程性能/任务性能/扩展性/局限”组织；
- 明确哪些结论是实测、哪些是理论、哪些没有证据。

## 4. 测试计划（先于大规模实验）

必须文件：

- tests/test_parameter_identity.py
- tests/test_gradient_flow.py
- tests/test_parameter_version.py
- tests/test_cache_correctness.py
- tests/test_cache_invalidation.py
- tests/test_cache_gradient.py
- tests/test_workflow_dag.py
- tests/test_output_integration.py

建议追加：

- tests/test_accounting.py：五口径 vs 手工/FlopCounter；
- tests/test_data_contract.py：划分、哈希、无泄漏；
- tests/test_cache_overhead.py：hash/lookup 成本；
- tests/test_reproducibility.py：同 seed 逐位一致；
- tests/test_failure_paths.py：非法 task_id、环、未授权调用、NaN。

验收：

- 8 个必需文件全绿，才允许跑阶段 2/3；
- 缓存失效/梯度任一失败，停止性能实验；
- 对象身份、唯一存储、版本一致性必须有断言，不靠日志；
- 末端 L1 的输入白名单测试必须通过。

## 5. 测量协议

- 统一硬件、精度、batch、线程数；GPU 空闲时测量；
- 预热后再计时；每个 batch 至少 200 次，保存逐次样本；
- 模型顺序交错，降低频率漂移偏差；
- 报告 P50、P95、median、mean、std、bootstrap 95% CI；
- 训练时间与推理时间分开；冷启动与 warm 分开；
- GPU：CUDA Event；CPU：perf_counter_ns；都显式 synchronize；
- CPU RSS 峰值用采样/Windows API，不用 tracemalloc 代替；
- 不以参数量或理论 FLOPs 代替实测延迟；
- 延迟 CV > 0.2 时增加重复/交错，不能直接下结论。

## 6. 结果管理

每次 run 保存：

- run_id、命令、config、config_hash；
- git commit、dirty diff hash、Python/torch/CUDA/cuDNN/驱动/CPU/GPU/RAM；
- 数据种子与数组 SHA256；
- 原始逐 epoch/逐 step 指标；
- 原始逐次延迟样本（JSONL/CSV）；
- 训练/推理峰值内存；
- checkpoint（可选，但 cache/冷启动实验必须保存）；
- failures.jsonl（含 traceback）；
- summary（从 raw 派生，不覆盖 raw）。

目录：

```text
experiments/anet_v2/results/<experiment>/<run_id>/
  config.json
  environment.json
  manifest.json
  raw.jsonl
  latency_raw.jsonl
  summary.json
  stdout.log
  stderr.log
  checkpoint.pt
```

- append-only；resume 必须校验 config_hash + code_hash + environment_hash；
- 新实验不得写入旧 results 目录；
- 旧结果只读。

## 7. 资源与失败处理

- 预检：small width、1 seed、max-steps=30，先验证端到端；
- 正式：按阶段 1→2→3→4，任何阶段失败先修，不跳阶段；
- OOM：记录 width/batch/架构，降一档重跑，报告失败规模；
- NaN/Inf：保存 traceback 与最后一个 finite step，停止该 run；
- 时间不足：保留已完成 raw 与失败日志，报告覆盖缺口，不伪造；
- 不在实验中途改架构定义；改定义必须新版本号并重跑。

## 8. 阶段交付物

| 阶段 | 交付物 | 通过条件 |
|---|---|---|
| 0 | 本审查 + 证据 + ERRATA | 已完成 |
| 1 | core + 8 测试 + preflight + scaling raw/summary | 测试全绿；preflight 0 失败 |
| 2 | 共享三组实验 + 梯度冲突/负迁移报告 | 对象身份/版本/梯度测试通过 |
| 3 | cache 正确性/失效/梯度 + 重复率性能报告 | 缓存测试全绿；收益/开销如实 |
| 4 | 2×2 综合矩阵 + 统计报告 | 覆盖相关性/规模/workflow/重复率 |
| 5 | 最终研究报告 + 原始数据 + 图 | 回答 10 个问题；区分实测/理论/无证据 |

## 9. 立即可执行的第一步

1. 从当前 HEAD 建分支 experiment/anet-v2。
2. 新增 experiments/anet_v2/core/parameter_block.py 与 tests/test_parameter_identity.py、test_parameter_version.py。
3. 跑 CPU 测试；通过后加 workflow/cache 与其余测试。
4. 全部 8 个测试通过后，跑 preflight（small width、1 seed、30 steps）。
5. preflight 通过后再开始阶段 1 的 450-run 矩阵；中途用 --resume，不覆盖旧 raw。

在第一步完成前，不开始任何大规模训练，也不修改旧结果。
