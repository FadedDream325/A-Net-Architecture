# A-Net 阶段 0 仓库审查报告

- 仓库：https://github.com/FadedDream325/A-Net-Architecture
- 审查分支：`experiment/scaling-and-crosslayer`
- 审查基点：`4752d3bed1df97f36755733e2f7591a1d03aaef8`（工作区干净）
- 审查日期：2026-10-30
- 本报告只做审查与核验，不修改任何历史代码、结果或报告。新增文件全部位于 `audit/stage0/`。

## 0. 结论摘要

1. **当前代码实现的是一个“共享浅层 L1 + Task-ID Top-1 专家路由”模型，不是提示词定义的完整 A-Net。** 它没有末端 L1 整合、没有参数所有权/使用权注册表、没有参数版本、没有 ResultCache，也没有“多个工作流调用同一参数块”的可复用调用图。现有 A/B/C 证据只能支持“稀疏专家路由”这一较弱的架构，不能直接支持提示词中的 A-Net。
2. **旧实验数据本身是自洽的。** 36 个主实验 + 9 个跨层消融全部成功，0 条失败记录；45 条记录的参数量、激活参数量、MACs、FLOPs 全部与解析公式一致；summary 与 raw 一致；报告中的表格数字可复算。
3. **旧报告中的 17% 口径确实是错的（分母混用）。** 历史 PDF 用 A-Net 单任务激活参数 8,770 除以独立基线三任务总参数 51,462，得到 17.0%；独立基线单任务激活参数是 17,154，正确比例是 51.1%。新报告改为约 50% 是正确的，但历史 PDF 仍在仓库中，需要显式勘误。
4. **旧实验的延迟/吞吐结论不够稳健。** batch=1 的 A-Net/Independent P50 比值在三个 seed 上分别为 2.607、0.747、1.266（small）；报告只给了“均值之比”1.49×。大 batch 吞吐的 small 规模逐 seed 比值也在 0.447–1.339 之间。不能把这些均值差当成稳定的硬件结论。
5. **现有 14 个测试全部通过，旧简单任务结果可精确复现。** 但提示词要求的 8 个测试文件（参数版本、缓存正确性/失效/梯度、DAG、末端整合等）一个都不存在；缓存、版本、DAG、末端 L1 也都没有实现。
6. **需要新建 v2 实验体系，而不是在旧代码上改定义。** 旧代码、旧结果、旧报告必须保留；v2 应放在新目录、新结果根、新分支上，先做最小核心与测试，再做规模/共享/缓存/综合实验。

## 1. 审查范围与方法

审查了以下内容：

- Git 分支、提交历史、远端、工作区状态、tag、忽略规则；
- 全部受控文件（60 个，约 4.0 MB）与提交历史中的变更；
- README、PROTOCOL、REPORT、requirements、PDF 历史报告；
- 现有模型、训练器、聚合脚本、冷启动脚本、三个历史脚本；
- 两组原始结果（`raw_runs.jsonl`、`summary.json/csv/md`、`config.json`、`environment.json`、图）；
- 提交的 ZIP 快照、父目录中的 patch/bundle；
- 架构草图 macro/micro；
- 当前环境、GPU、依赖；
- 现有测试套件；
- 复现实验：旧简单任务 demo、旧 scaling 的 small/A-Net/seed=2026、CUDA 兼容性测试。

审计脚本：`audit/stage0/audit_results.py`（只读旧结果，输出到 `audit/stage0/`）。
机器可读证据：`audit/stage0/stage0_evidence.json`、`recomputed_runs.csv`、`recomputed_groups.csv`、`latency_ratios.csv`。
历史文件哈希：`audit/stage0/legacy_manifest.json`。

## 2. 仓库状态

### 2.1 Git 与提交

| 项目 | 值 |
|---|---|
| 当前分支 | `experiment/scaling-and-crosslayer` |
| HEAD | `4752d3b Add downloadable A-Net scaling snapshot ZIP` |
| 主分支 | `main` = `c68e320 Add files via upload` |
| 实验提交 | `0dda2fe Add controlled scaling and cross-layer feasibility study` |
| 远端 | `https://github.com/FadedDream325/A-Net-Architecture.git` |
| 工作区 | 干净（审查前）；`git fsck` 无输出 |
| tag | 无 |

提交链：

```text
4752d3b (HEAD) Add downloadable A-Net scaling snapshot ZIP
0dda2fe        Add controlled scaling and cross-layer feasibility study
c68e320        Add files via upload
d8ed473        Initial commit
```

### 2.2 文件与产物

- 旧仓库（`c68e320`）：README、PDF 报告、2 张架构草图、4 个 Python 脚本、LICENSE、.gitignore；没有 tests、requirements、CI、结构化结果。
- 实验提交（`0dda2fe`）：新增 `experiments/scaling/`、`tests/`、`requirements.txt`，并重写 README；提交了 36 条主实验和 9 条消融的 raw/summary/图，以及冷启动结果。
- `4752d3b`：新增 `A-Net-Architecture-scaling.zip`（1.67 MB），内容是同一套代码与结果的快照。
- 旧结果目录：`experiments/scaling/results/scaling_v1`、`experiments/scaling/results/scaling_cross_ablation`。
- **checkpoint 文件不在仓库中**：`.gitignore` 忽略了 `results/*/checkpoints/` 与 `*.pt`；冷启动 JSON 存在，但没有对应权重，无法直接复跑冷启动，除非重新训练或从别处恢复。

### 2.3 ZIP 快照完整性

- ZIP 共 59 个文件条目，全部能在工作区找到。
- 按字节比较有 17 个文件不同，差异全部是 CRLF/LF 行尾（每个文件大小正好多出“行数”个字节）。
- 将行尾归一化后，**ZIP 与当前工作区 0 个不匹配**。
- 结论：ZIP 是当前代码与结果的可靠快照，可用于历史归档；但不能替代原始 raw 记录中的代码哈希。

### 2.4 历史 PDF

- PDF 共 12 页，保留在仓库根目录。
- 内容与 README 描述一致：简单任务 99.50/99.42/99.37；复杂任务 96.23/94.49/95.84；CUDA 验证通过。
- PDF 中的“17.0%”口径错误（见 4.4）；新 REPORT 已修正，但 PDF 本身未勘误。
- PDF 的复杂任务与当前 scaling 的数据/噪声/宽度不同，不能直接比较；REPORT 已说明这一点。

## 3. 架构一致性审查

### 3.1 提示词定义 vs 当前实现

| 提示词要求 | 当前代码 | 判定 |
|---|---|---|
| 前馈结构，取消工作流间反馈 | `ANetHardRouting` 是 L1→expert 前馈；没有工作流图 | 部分符合：没有反馈，但也没有工作流间边 |
| 前置 L1 负责输入处理与工作流分配 | `l1 = Linear(2,h)+ReLU`；分配由外部 `task_id` 整数索引完成 | 部分符合：有 L1，但分配不是由 L1 产生 |
| 多个相对独立工作流，每个拥有自己的参数层 | 3 个 expert，各自拥有参数；L1 是共享的，不属于任何 workflow | 部分符合 |
| 工作流可调用其他工作流拥有的参数块，且是同一参数对象 | 只有父模型拥有的 `shared_block`；所有 expert 调用它；不属于任何 workflow | **不符合**：没有“所有者工作流”，也没有所有权/使用权分离 |
| 参数所有权与使用权分离 | 没有 Registry/Owner/Grant/版本 | **不符合** |
| 相同参数块、相同输入、相同配置可复用结果 | 没有 ResultCache | **不符合** |
| 末端 L1 整合各工作流最终输出 | 直接返回被选中 expert 的 logits | **不符合** |
| 第一版固定、可控的调用关系 | Task-ID 固定路由；cross 固定插在 fc1 后 | 符合 |
| 参数版本 | 没有 | **不符合** |
| 可选输入适配器，单独统计 | 没有 | **不符合** |

### 3.2 当前三个 A-Net 相关模型

1. `ANetHardRouting`（`multitask_anet_demo.py`）：
   `x -> shared L1 -> expert[task_id] -> logits`。每个样本只执行一个 expert；未选中的 expert 不参与前向，也收不到梯度。这是**稀疏专家路由**，不是完整 A-Net。
2. `ANetCrossLayer`（`experiments/scaling/models.py`）：
   `x -> L1 -> expert[t].fc1 -> shared_block -> expert[t].fc2`。`shared_block` 注册在父模型上，作为参数传入 expert，确实没有复制参数；这验证了“直接调用同一对象”和“多路径梯度累积”的机制，但它是**在原有 expert 之外额外加一个块**，不是“用共享块替换各工作流自己的重复块”，因此不能证明跨工作流共享会减少唯一参数存储。
3. `cross_off`：
   同一个类，`use_cross_layer=False`，参数仍注册但不执行，用于等参数量的“是否执行块”消融。这个设计是合理的，但它只分离了“执行”与“存储”，没有分离“共享存储”与“非共享存储”。

### 3.3 与架构草图的关系

两张草图显示的是：多个工作流 L1→L4→L6 等，工作流之间可以调用其他工作流的参数块，最后由末端 L1 做整合。也就是说，**草图表达的结构比当前代码更接近提示词，而当前代码是草图的简化/退化版**。当前仓库没有实现草图中的末端整合和跨工作流调用图，不能拿当前 A-Net 的结果去证明草图中的完整结构。

### 3.4 架构结论

- 当前代码可以证明：共享浅层 + 稀疏专家 + Task-ID 硬路由在合成任务上可训练、可推理、参数量/激活量可计算。
- 当前代码**不能证明**：完整 A-Net 的末端整合、跨工作流参数块调用、参数所有权/使用权、结果复用、缓存梯度正确性、参数版本。
- 因此，提示词中的 A（基础工作流架构）目前只得到“相关弱化架构”的证据；B（跨工作流参数块调用）只有“同一父对象被多个 expert 调用”的机制证据，没有“替换重复块从而减少存储”的证据；C（结果复用）完全没有实现。

## 4. 旧结果核验

### 4.1 原始记录完整性

| 数据集 | 记录数 | 成功 | 失败 | 唯一 run_id | 模型 | 规模 | seed |
|---|---:|---:|---:|---:|---|---|---|
| `scaling_v1` | 36 | 36 | 0 | 36 | independent/shared/anet/cross | small/medium/large | 2026/2027/2028 |
| `scaling_cross_ablation` | 9 | 9 | 0 | 9 | cross_off | small/medium/large | 2026/2027/2028 |

- 每条记录都有 `cost`、`train`、`metrics`、`inference`、`routing`、`data`、`config`。
- 所有成功记录中，同一 (scale, model) 的 `cost` 完全一致。
- 所有 `accuracy_per_task` 都在 [0,1]，均值与逐任务均值一致。
- 所有 batch 的 `p50_us <= p95_us`，吞吐为正。
- 没有重复成功 run_id，没有失败的 traceback 记录。
- **但 REPORT 提到的 preflight checkpoint 失败没有对应的失败 JSONL**；失败日志没有保留，只有文字描述。结果管理上应保留失败记录。

### 4.2 参数与计算量公式核验

对 45 条成功记录逐条重算：

| 量 | 公式 | 结果 |
|---|---|---|
| Independent 唯一参数 | `3*(h*h+6h+2)` | 全部一致 |
| Shared 唯一参数 | `h*h+10h+6` | 全部一致 |
| A-Net 唯一参数 | `3h+3he+9e+6` | 全部一致 |
| Cross/Cross-off 唯一参数 | A-Net + `e*e+e` | 全部一致 |
| Independent/Shared MACs | `h*h+4h` | 全部一致 |
| A-Net MACs | `2h+he+2e` | 全部一致 |
| Cross MACs | A-Net + `e*e` | 全部一致 |
| Cross_off MACs | A-Net | 全部一致 |
| FLOPs | `2*MACs` | 全部一致 |
| checkpoint 大小 | 约 `4*unique_params` 字节 | 在 1% + 64 KB 容差内一致 |

结论：旧实验的**解析核算**是自洽的；参数与 FLOPs 公式没有发现错误。

### 4.3 summary/report 数字复算

- `summary.json` 的 12 行与 raw 重算完全一致（accuracy mean/std、batch1 P50、组内 seeds）。
- REPORT 3.1 的 accuracy 表、3.2 的比率表、3.3 的存储/训练/延迟表，与 raw 复算在四舍五入范围内一致。
- 但 `summary` 只保留 batch1 P50 均值，**没有保留 P95、P50 中位数、延迟标准差、吞吐标准差**；raw 中的 P95 没有进入任何聚合表。
- `summary` 对每个组只取均值，没有置信区间；REPORT 已声明“探索性”，这是诚实的，但不能满足提示词对 P50/P95 和多种子统计的要求。

### 4.4 历史 17% 口径核验

历史 PDF（第 10–11 页）写：

- 独立基线：总参数 51,462；A-Net：总参数 25,542，单样本激活 8,770。
- “A-Net 单样本激活参数只有独立基线的 17.0%”。

复算：

```text
8,770 / 51,462 = 17.04%       # 分子是 A-Net 单任务路径，分母是独立基线三个任务的总参数
独立基线单任务激活 = 51,462 / 3 = 17,154
8,770 / 17,154 = 51.12%       # 同口径比较
A-Net 总参数 / 独立总参数 = 25,542 / 51,462 = 49.63%
A-Net MACs = 2*128 + 128*64 + 2*64 = 8,576
独立 MACs = 128*128 + 4*128 = 16,896
8,576 / 16,896 = 50.76%       # 计算量也不是 17%
```

结论：17.0% 同时混用了“单任务激活”与“三任务总参数”两个口径；按同口径，A-Net 的激活参数和理论计算量都约为独立基线的 50%。新 REPORT 的修正（约 50%）是正确的。历史 PDF 应保留，但需要显式勘误；建议新增 `ERRATA.md`，不要修改 PDF 原文件。

### 4.5 复现结果

在恢复的 PyTorch 2.14.1+cu130 / RTX 5060 Laptop 环境下：

1. 现有测试：`14 passed in 7.70s`（`audit/stage0/pytest_stage0.log`）。
2. 旧简单任务 demo（CPU，seed=42）：
   - 独立 99.50%，共享 99.42%，A-Net 99.37%；
   - 与历史 PDF 完全一致；逐任务 99.25/100.00/99.25 等也一致。
   - 日志：`audit/stage0/legacy_demo_cpu.log`。
3. 旧复杂任务（high-freq sine / four rings / checkerboard，CPU 8 线程，3 seeds）：
   - 独立 96.23 ± 0.19，共享 94.49 ± 1.25，A-Net 95.84 ± 0.14；
   - 逐任务均值也与历史 PDF 完全一致（如独立 96.31/97.87/94.51，A-Net 95.58/98.00/93.93）；
   - 说明旧复杂任务结果可精确复现；
   - 日志/原始 JSON：`audit/stage0/complex_tasks_cpu_stage0.log`、`audit/stage0/.complex_results_cpu.json`。
4. 旧 scaling 的 `small / anet / seed=2026` 完整 50 epoch 复现：
   - 历史逐任务 accuracy：`[0.9923332929611206, 0.9816666841506958, 0.968999981880188]`；
   - 复现逐任务 accuracy：完全相同（delta = 0, 0, 0）；
   - 总参数 100,230，激活 33,922，与历史一致；
   - 训练耗时 5.57 s vs 历史 5.00 s（+11%，受系统负载影响）；
   - batch=1 P50 115.4 µs vs 历史 221.0 µs，说明历史那条延迟记录是明显的高离群值。
   - 日志/结果：`audit/stage0/repro_small_anet_s2026/`、`repro_small_anet_s2026.log`。
5. CUDA 兼容性测试：0 失败，FP32/FP16/BF16/cuDNN/三种模型 forward-backward-Adam/autocast 全部通过；GPU 为 RTX 5060 Laptop，sm_120，7.93 GiB。
   - 日志：`audit/stage0/cuda_compat_stage0.log`。

结论：旧代码的**训练结果**可以精确复现；旧代码的**延迟结果**不可精确复现，且历史单条延迟可能受系统负载影响。

### 4.6 溯源问题

- 两份 `environment.json` 都记录 `git_commit = c68e320`，而实验代码是在后续提交 `0dda2fe` 才进入 Git 的；raw 记录没有代码哈希、没有 dirty 标记、没有命令行。
- ZIP 快照与当前工作区归一化后完全一致，说明“实验时的工作树”很可能就是后来提交的代码；但这是推断，不是可验证的溯源。
- checkpoint 未提交，冷启动结果无法直接复跑。
- `resume` 只比较部分配置字段，不比较 benchmark 设置、模型代码版本、环境；不同环境/代码的结果可能被追加到同一 raw 文件。
- `environment.json` 在每次运行开始时被覆盖，没有按 run 保存环境快照。

这些是 v2 必须修复的结果管理问题。

## 5. 参数统计口径审查

### 5.1 五个口径的定义与当前状态

| 口径 | 严格定义 | 当前实现 | 缺口 |
|---|---|---|---|
| 参数引用总量 | 所有调用点引用的参数元素之和，重复调用重复计数 | 未统计 | 没有 call site/registry；cross 的 3 次调用未计引用 |
| 唯一参数量 | 实际存储的不同参数元素数 | `count_parameters`，正确 | 无 |
| 单次前向实际激活参数量 | 一次 forward 中实际参与计算的不同参数元素数 | `active_parameters`，按当前路由选择 L1+1 expert；cross 加 shared_block | 不乘重复调用；没有多工作流/混合输入；没有缓存后的“实际执行”概念 |
| 理论 FLOPs | 按调用图、每样本、未缓存时的解析计算量 | `analytical_flops = 2*MACs`，单调用正确 | 没有 batch 乘数；没有引用总量/调用次数；没有实际执行 FLOPs |
| 实际执行计算次数 | 运行时每个 block 的 forward 次数与实际 MACs/FLOPs | **未统计** | 没有 block 级计数器；没有 cache；没有实际 FlopCounter 运行时记录 |

补充：旧代码用 `active_parameters` 表示“不同参数元素”，不是“执行次数 × 参数量”。在 cross 中 shared_block 每个样本只调用一次，所以两者恰好接近；一旦一个 block 在同一次 forward 中被多个调用方调用，这两个口径就会分开。提示词要求的正是后者。

### 5.2 small 规模示例

| 模型 | 唯一参数 | 单样本激活参数 | 单样本 MACs | 说明 |
|---|---:|---:|---:|---|
| Independent | 201,222 | 67,074 | 66,560 | 3 个 MLP；单样本只跑 1 个 |
| Shared | 68,102 | 68,102 | 66,560 | 全参数激活 |
| A-Net | 100,230 | 33,922 | 33,536 | L1 + 1 个 expert |
| Cross | 116,742 | 50,434 | 49,920 | A-Net + shared_block |

这张表说明：**不能把 A-Net 的 33,922 除以 Independent 的 201,222 得到 16.9% 并称为“计算量比例”**；同口径应除以 Independent 单任务激活 67,074，得到 50.6%。历史 17% 的错误就是这一点的放大版。

## 6. 实验设计缺口

### 6.1 数据与任务

现有：

- 3 个 2D 二分类任务：easy sine / medium sine / hard checker；
- 12,000 train / 3,000 test 每任务；训练标签噪声 1%/2%；测试标签干净；
- 3 个种子；没有验证集；
- 任务相关性只有“T1/T2 同族、T3 不同族”的定性描述，没有高/中/低三档，也没有量化相关性。

缺口：

- 没有非线性回归任务；
- 没有多分类/复杂决策任务；
- 没有高/中/低任务相关性的受控因子；
- 没有验证集与“测试集绝不参与选择”的流程；
- 没有数据哈希/数据清单，无法独立核验训练/测试张量是否完全一致；
- 没有多任务混合 batch（当前是 task-by-task 顺序训练）；
- 没有 workflow 数量因子（固定 3）。

### 6.2 规模与预算

现有 hidden = 256/896/1792，只有 3 档；提示词建议 64/128/256/512/1024/2048 六档。现有只有 3 seeds，提示词建议至少 5。现有每个 run 50 epoch、batch 512、FP32；训练预算一致，但参数量并不匹配：

- Independent 总参数约为 A-Net 的 2×；
- Shared 总参数约为 A-Net 的 0.68×；
- A-Net 与 Cross 也不等参数。

REPORT 已如实报告差异，但没有等参数/等 FLOPs 控制组，因此不能把“精度相当”直接解释为“结构更优”。

### 6.3 指标

现有：

- accuracy（逐任务/均值/标准差）；
- 唯一参数、激活参数、解析 MACs/FLOPs；
- checkpoint 大小；
- 训练时间、训练 GPU 峰值；
- batch=1/64/256/1024 的 P50、吞吐；
- 冷启动（但 checkpoint 不在仓库）。

缺口：

- 没有 P95 聚合（raw 有，summary 丢弃）；
- 没有延迟/吞吐的标准差、中位数、置信区间；
- 没有 CPU 内存峰值；
- 没有 F1、回归 MSE/MAE/R²、多分类 macro-F1；
- 没有实际执行次数、实际 FLOPs、缓存命中率/查找耗时/缓存内存；
- 没有多工作流推理、多任务混合输入；
- 没有梯度冲突/负迁移指标；
- 没有参数量/FLOPs 匹配的控制组；
- 没有统计显著性检验（3 seeds 也不够）。

### 6.4 延迟测量

- `first_call_after_training_us` 是在 30 次 warmup **之后**测的（`benchmark_inference` 先 warmup，再记录），名字有误导性；它不是冷启动，也不是首次调用。
- 延迟只在训练后测量，且 GPU 与桌面应用共享；没有 GPU 时钟/温度/利用率记录。
- 每个 batch 的 200 次事件被汇总成 P50/P95，但**没有保存逐次样本**，无法复算分布或做配对检验。
- 报告只给 P50 均值；P95 没有进入报告表格。
- 逐 seed 的延迟波动很大：small A-Net batch1 P50 为 221.0/75.9/105.9 µs（CV 0.571），large A-Net 为 207.7/135.9/90.4 µs（CV 0.409）。因此“1.30–1.49× 更慢”是均值之比，不是稳定结论。

## 7. 测试缺口

现有 14 个测试（5 + 6 + 3）：

- `test_cross_layer.py`：对象唯一注册、直接调用公式、只选 expert 梯度、双路径梯度累积、禁用调用；
- `test_scaling_models.py`：参数公式、规模目标、前向/反向/优化、只选 expert 梯度、cross 参数更多、解析 FLOPs vs FlopCounter；
- `test_scaling_tasks.py`：数据形状/平衡/干净测试标签、确定性、噪声只进训练。

这些测试质量不错，但覆盖的是旧架构。提示词要求的 8 个测试文件：

| 要求文件 | 是否存在 | 现状 |
|---|---|---|
| `test_parameter_identity.py` | 否 | 旧 `test_cross_layer` 只覆盖“一个 parent block” |
| `test_gradient_flow.py` | 部分 | 旧测试覆盖双路径累积，但没有缓存路径/多调用方 |
| `test_parameter_version.py` | 否 | 没有版本概念 |
| `test_cache_correctness.py` | 否 | 没有 cache |
| `test_cache_invalidation.py` | 否 | 没有 cache |
| `test_cache_gradient.py` | 否 | 没有 cache |
| `test_workflow_dag.py` | 否 | 没有 DAG |
| `test_output_integration.py` | 否 | 没有末端 L1 |

另外缺少：实际执行计数、CPU 内存、缓存开销、数据泄漏、checkpoint 重载、配置/环境溯源、失败路径测试。

## 8. 工程与结果管理缺口

- `requirements.txt` 只有 `torch>=2.0` 等宽范围，没有锁文件；当前环境的 torch 2.14.1+cu130 是手动恢复的。
- 当前 base conda 环境**没有 torch**；审查时先发现 `ModuleNotFoundError`，后通过工作区内的 wheel 解压恢复。这个过程没有写入仓库，但说明环境不可由 `requirements.txt` 直接重建。
- `environment.json` 没有 pip freeze、没有 CUDA 驱动版本、没有 CPU 型号/内存、没有 GPU 当前负载；且会被覆盖。
- `config.json` 没有数据版本、代码哈希、命令行、随机源分解（数据/模型/优化器）。
- raw 只有聚合后的每 run 记录，没有逐 epoch/逐 step 曲线，没有逐次延迟样本。
- `resume` 的 config 检查不完整；没有 run manifest 锁。
- 失败日志：REPORT 提到的 preflight 失败没有保留原始 traceback；当前 raw 0 失败，但不能证明所有尝试都成功。
- 旧结果与 PDF 的 17% 口径没有勘误文件；读者可能只读 PDF。
- 提交了 1.67 MB ZIP 与 1.22 MB PDF，历史可追溯，但增加仓库体积；可接受，但应在 README 说明。

## 9. 风险清单

| 级别 | 风险 | 证据 | 影响 | 处理 |
|---|---|---|---|---|
| 严重 | 当前实现不是提示词定义的 A-Net | 无末端 L1、无 registry、无 cache、无多工作流调用图 | 旧结论不能回答 A/B/C 的核心问题 | 新建 v2，旧结果只作 baseline/历史证据 |
| 严重 | 跨工作流共享没有“替换重复块”的对照 | `ANetCrossLayer` 是额外加块 | 无法证明唯一参数存储下降 | v2 增加 no-share/direct-share/adapter 三组，共享块替换重复块 |
| 严重 | 延迟结论不稳健 | batch1 逐 seed 比值 0.747–2.607 | “1.30–1.49×”不可复现 | 配对/交错测量、保存逐次样本、P50/P95/CI、空闲 GPU |
| 严重 | 结果溯源不完整 | env commit c68e320 ≠ HEAD；无代码哈希/命令行；checkpoint 缺失 | 无法从 Git 精确重建旧实验 | v2 每次 run 保存 git commit+dirty diff hash+命令行+环境+权重 |
| 高 | 参数统计口径不全 | 无引用总量、无实际执行次数 | 容易再次产生 17% 类错误 | 实现 accounting 模块，五个口径分开 |
| 高 | 数据集/任务不足 | 只有 3 个二分类；无回归/多分类/相关性因子 | 结论外部有效性弱 | v2 三类任务 × 三档相关性 × 5 seeds |
| 高 | 测试覆盖不足 | 缺少 8 个必需测试文件 | 缓存/版本/DAG/整合没有正确性保障 | 先写测试，再写实现（或同步） |
| 高 | 无等参数/等 FLOPs 控制 | Independent 2×、Shared 0.68× | 精度差异不能归因于结构 | 增加参数匹配/FLOPs 匹配控制 |
| 中 | 环境不可重建 | torch 缺失、依赖未锁 | 复现失败 | 提供 conda/venv 锁文件与 bootstrap 脚本 |
| 中 | summary 丢弃 P95/方差 | 聚合脚本只取均值 | 报告不满足指标要求 | 新聚合保留 P50/P95/median/std/CI |
| 中 | resume 配置检查不全 | 只比较部分字段 | 混合不同代码/环境结果 | 配置哈希 + 环境哈希 + 代码哈希 |
| 中 | 无 CPU 内存/混合输入/工作流数 | 现有 benchmark 只 task0 | 工程性能证据不全 | v2 全覆盖 |
| 低 | PDF 17% 未勘误 | 历史 PDF 仍可被单独阅读 | 误导 | 新增 ERRATA.md，不修改 PDF |
| 低 | 日志编码/中文乱码 | PowerShell 默认编码 | 可读性 | 统一 UTF-8，保存原始日志 |

## 10. 对 A/B/C 的当前判断

| 主张 | 当前证据 | 判断 |
|---|---|---|
| A. 基础工作流架构可行 | 旧 A-Net 是共享 L1 + 稀疏 expert，无末端整合/工作流图 | **仅有弱化架构的证据；完整 A 未验证** |
| B. 跨工作流参数块调用有效 | `ANetCrossLayer` 证明同一对象、梯度累积、可训练；但共享块是额外加的，精度增益不一致 | **机制原型可行；存储收益/稳定任务收益未证明** |
| C. 结果复用正确且有性能收益 | 无 cache | **未实现，无证据** |
| D. 两种机制组合 | 无 | **未实现** |
| E. 规模/工作流/任务复杂度扩展 | 3 规模 × 3 seeds × 3 任务；有初步趋势 | **初步但不足；需要 v2 重做** |

## 11. 分阶段实施计划

详细计划见 `audit/stage0/PLAN.md`。核心原则：

1. **不覆盖旧文件**：旧代码、旧结果、旧报告、PDF、ZIP 全部保留；v2 放在 `experiments/anet_v2/`，结果放在 `experiments/anet_v2/results/`。
2. **先测试后扩规模**：先实现 core + 8 个必需测试 + accounting，跑 preflight；通过后再跑规模实验。
3. **五个口径分开**：references / unique / active / theoretical FLOPs / actual executions，全部在 run 记录中。
4. **缓存正确性优先**：对象身份、参数版本、输入哈希、运算配置、训练/评估状态、Dropout/BN 状态、参数更新失效，全部测试。
5. **统计与测量**：5 seeds、P50/P95/median/std、bootstrap CI、配对比较；逐次延迟样本落盘。
6. **结果管理**：每次 run 保存 config、git commit+dirty、环境、命令行、raw samples、failures、summary；append-only。

## 12. 立即可执行的最小改动（阶段 1 的第一步）

不改旧文件，新增：

```text
experiments/anet_v2/
  __init__.py
  core/parameter_block.py     # ParameterBlock, BlockSpec, ParameterRegistry, Grant
  core/workflow.py            # Workflow, CallSite, WorkflowGraph, PreL1, TerminalL1
  core/result_cache.py        # ResultCache, CacheKey, CacheStats, invalidation
  core/accounting.py          # 五个口径 + 实际执行计数
  core/models.py              # Independent, Shared, ANetBasic, ANetShare, ANetReuse, ANetFull
  core/tasks.py               # 分类/回归/多分类 + 三档相关性
  core/train.py               # 统一训练/评估/混合 batch
  core/benchmark.py           # P50/P95/吞吐/内存/逐次样本
  runners/run_stage1.py       # 预检 + 规模
audit/stage0/                 # 本报告与证据
tests/test_parameter_identity.py
tests/test_gradient_flow.py
tests/test_parameter_version.py
tests/test_cache_correctness.py
tests/test_cache_invalidation.py
tests/test_cache_gradient.py
tests/test_workflow_dag.py
tests/test_output_integration.py
```

阶段 1 的验收门槛：

- 8 个必需测试 + accounting/data 测试全部通过；
- 同一 block 对象在 registry 中只注册一次，多个 workflow 的 `id(block)` / 参数 `data_ptr` 相同；
- 两个调用方的梯度之和等于分别反传之和；
- 参数更新后缓存 100% 失效；
- 不同输入/不同 dtype/不同 train-eval/Dropout/BN 状态不得命中；
- DAG 拓扑排序可检测环；
- 末端 L1 只接收 workflow 最终输出，不读取中间隐藏状态；
- 五个口径在小模型上与手工计算/FlopCounter 对齐；
- preflight 在小宽度、1 seed、少量 step 上无失败。

## 13. 证据索引

| 文件 | 内容 |
|---|---|
| `audit/stage0/stage0_evidence.json` | 自动审计的机器可读结果 |
| `audit/stage0/recomputed_runs.csv` | 45 条成功 run 的关键指标 |
| `audit/stage0/recomputed_groups.csv` | 15 个 (scale, model) 组的均值/标准差/P95 |
| `audit/stage0/latency_ratios.csv` | 逐 seed 延迟/吞吐比值与稳定性 |
| `audit/stage0/legacy_manifest.json` | 旧仓库 60 个受控文件的 SHA256、commit、外部产物哈希 |
| `audit/stage0/pytest_stage0.log` | 14 passed |
| `audit/stage0/legacy_demo_cpu.log` | 旧简单任务精确复现 |
| `audit/stage0/repro_small_anet_s2026/` | 旧 scaling 单 run 复现结果 |
| `audit/stage0/cuda_compat_stage0.log` | CUDA 兼容性 0 失败 |
| `audit/stage0/verification.json` | 测试/demo/复杂任务/scaling 复现/CUDA 的机器可读汇总 |
| `ERRATA.md` | 历史 PDF 17% 口径与延迟结论的显式勘误 |

## 14. 最终判断

- **机制正确性**：旧代码的稀疏路由、对象共享、梯度累积是正确且可测的；完整 A-Net 的机制尚未实现。
- **存储收益**：旧 A-Net 相对 Independent 约 50%，这是真实的结构性结果；但旧 cross 没有证明“共享替换重复块”会降低存储。
- **计算收益**：旧 A-Net 的解析 MACs 约为 Independent 的 50%；实际延迟只在部分规模/大 batch 下有优势，batch=1 不稳定且常更慢。
- **工程性能收益**：训练时间在 large 规模有优势，显存峰值较低；但延迟测量噪声大，不能宣称普遍加速。
- **任务性能影响**：旧 A-Net 与 Independent 接近（-0.20/-0.15/+0.04 pp），优于 Shared 但存储更高；没有回归/多分类证据。
- **扩展性**：3 档宽度、3 seeds、3 任务的初步趋势存在；不足以支撑 64–2048 宽、5 seeds、多工作流数的结论。
- **当前局限**：架构定义不完整、缓存缺失、参数口径不全、延迟不稳、溯源不足、测试不足。这些必须在 v2 中解决，不能靠旧结果外推。

**因此，阶段 0 的结论是：旧仓库可以作为“稀疏专家路由”的可靠历史基线，但不能作为提示词定义的 A-Net 的验证；下一步应新建 v2，先实现核心机制与测试，再开展规模/共享/缓存/综合实验。**
