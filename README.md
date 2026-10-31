# A-Net

**Workflow-owned parameter sharing with explicit call graphs, result reuse, and workflow isolation**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-research%20prototype-6B7280)]()

> 这是一个研究原型，不是已证明的大模型结论。当前最强证据来自 50M char-level pilot 和 2D 合成任务；1B/2B 语言能力尚未验证。

## 核心摘要：非对称容量扩展的实测边界

> **打平，但不是“成本腰斩”。**  
> 在 12 层深度、37.5M 单次激活参数、等 FLOPs 的约束下，Isolated A-Net 的 Math Exact Match 为 **0.500 vs 0.517**，Code 语法 Pass@1 为 **0.220 vs 0.247**，与 Transformer-12L 统计上打平。  
> 代价是唯一参数 **99.83M vs 37.50M（2.66×）**、checkpoint **399.5 MB vs 150.1 MB（2.66×）**、训练峰值显存 **+18%**。  
> 因此正确结论是：**隔离工作流用唯一存储换取模块化容量；它不是低成本替代，也不是无条件更省显存。**

| 指标 | Transformer-12L | Isolated A-Net | 实测关系 |
|---|---:|---:|---|
| 唯一参数 | 37.50M | 99.83M | **2.66×** |
| 单次激活参数 | 37.50M | 37.50M | 1.00× |
| 激活 FLOPs/token | 40.59M | 40.59M | 1.00× |
| 深度 | 12 | 12 | 对齐 |
| Math Exact Match | **0.517** | 0.500 | 打平（CI 重叠） |
| Code 语法 Pass@1 | **0.247** | 0.220 | 打平（CI 重叠） |
| 训练峰值 allocated | 1.508 GB | 1.786 GB | **+18%** |
| checkpoint | 150.1 MB | 399.5 MB | **2.66×** |

**术语校正：**

- 本文的“容量扩展”指**唯一参数/checkpoint 容量**，不是“知识量”或“推理能力”的已证明增长。
- 本文的“显存”若无特别说明，指**训练峰值 allocated**；“推理峰值”尚未单独测量。
- 隔离由 **IsolatedWorkflowANetLM 的结构隔离**实现：每个任务一个独立 10 层 workflow，不共享参数、不互相调用；它没有使用 detach_both，也没有把未激活专家流放到 DDR5。
- “冷数据驻留 DDR5 / 存算解耦”是**尚未实现的工程假设**，不是本实验结论。

## 宏观叙事：不严谨推演（欢迎指正）

> **以下内容不是实验结论，而是宣传性/愿景性推演。**  
> 当前可复现的实测结果只支持：唯一参数 2.66×、激活参数/FLOPs 对齐、Math/Code 打平、训练峰值 +18%、checkpoint 2.66×。  
> 下面这些“3 倍、20%、腰斩、1 小时、8GB 绝对、30B 必然”等表述没有对应实验，属于不严谨外推；欢迎在 GitHub Issues 指正。

- **“打破超大容量、极低显存、绝对性能的不可能三角”**：这是 A-Net 的设计愿景，不是已证明的物理结论。
- **“3 倍物理知识库”**：当前实测唯一参数为 2.66×；“知识量”没有测量。把它说成 3 倍是宣传性取整。
- **“推理显存仅增加 20%”**：没有单独测推理峰值；训练峰值实测 +18%。“推理 +20%”是外推，不是测量。
- **“性能平替、成本腰斩”**：Math/Code 打平有实测支持；“成本腰斩”不成立——唯一参数和 checkpoint 反而是 2.66×。若未来实现冷热分离，才可能谈推理成本下降。
- **“未激活专家流驻留 DDR5，实现存算解耦”**：这是尚未实现的工程设想；当前没有 DDR5 offload、没有跨节点加载、没有推理峰值数据。
- **“6 层残血保持 92% 能力，Transformer 同深度会崩塌”**：没有测到 92%，也没有做 Transformer 同深度阉割对照；这是不严谨的直觉判断。
- **“100M 训练 1 小时，Transformer 要 2–5 小时”**：没有 100M Transformer 的等预算对照；当前 A-Net 是 6 个独立 12M-token run，约 35 分钟，不是同体量端到端训练。
- **“8GB 绝对无法训练 100M Transformer”**：不成立/未验证；100M FP32+Adam 静态状态约 1.6 GB，是否 OOM 取决于 batch/seq/实现。
- **“多节点分布式联合训练、即插即用 AGI 进化”**：这是架构愿景，不是已实现功能；当前只有独立训练/独立 checkpoint。
- **“30B+、40–60 层专家流、消费级硬件运行多专家模型”**：这是未来工作假设，不是本轮结果。

**一句话：可以把它当作 A-Net 的“愿景叙事”，不能当作实验证据。**

## 当前结论

2026 年新增的 50M 三架构语言 pilot 得到一个重要的修正结果：

| 模型 | 唯一参数 | 单次激活参数 | 有效深度 | 平均 bits-per-byte |
|---|---:|---:|---:|---:|
| **A-Net Shared-Depth** | 50.18M | 37.50M | 12 | **1.933 ± 0.067** |
| Transformer-12L | 37.50M | 37.50M | 12 | 1.991 ± 0.033 |
| Shared Backbone | 50.18M | 50.18M | 16 | 1.975 ± 0.028 |
| Transformer-16L | 49.92M | 49.92M | 16 | 2.007 ± 0.055 |
| Sparse A-Net（旧实现） | 50.18M | 37.50M | **6** | 2.141 ± 0.107 |

- 在**同深度、同激活参数、同 FLOPs** 下，Shared-Depth A-Net 比 Transformer-12L 低 **0.058 bpc**，Welch p ≈ **0.046**（3 seeds）。
- 这是一个 **pilot 级** 正结果：接近 0.05、只有 3 seeds、3M tokens，不能当作最终显著性。
- 之前的 Sparse A-Net 更差，主要原因是它把有效路径从 12 层降到了 **6 层**；这不是“共享一定伤害能力”的证据。
- 对照 C 中 A-Net 唯一参数 50.18M，仍比 Transformer-12L 多 12.7M（另一个未激活 workflow tail）；它的收益是**相对于多个独立 workflow 的并集，共享块减少了重复存储**，同时在保持 12 层路径时把单次激活参数维持在 37.5M。它不是“比同深度 Transformer 更少唯一参数”，也不是无条件的精度或延迟胜利。
- 成本上：对照 C 的 batch1 P50 为 **3.00 ms vs 4.70 ms**（A-Net 更快）；训练显存/时间与 Transformer-12L 基本持平（1.59 vs 1.51 GB，100.3 vs 101.7 s）。**Sparse A-Net 的训练峰值确实更低（1.69 vs Transformer-16L 1.98 GB）**，但那是用 6 层有效深度和更差能力换来的。详见“成本画像”。

## 事实边界：哪些流行说法不能写入结论

| 拟表述 | 严格判定 |
|---|---|
| “3 倍物理知识库” | 唯一参数为 **2.66×**，不是 3×；没有测量“知识量”。 |
| “推理显存仅增加 20%” | 没有单独测推理峰值；训练峰值 allocated 为 **+18%**。 |
| “1.59 GB vs 1.51 GB” | 这是 Shared-Depth 训练峰值，不是 Isolated A-Net，也不是推理。Isolated 为 **1.786 vs 1.508 GB**。 |
| “detach_both 物理隔离” | Isolated A-Net 是**结构隔离**：独立 workflow、无共享参数；没有使用 detach_both。 |
| “6 层残血保持 92% 能力” | 未测到 92%。6M 预算下 Shared-Depth（12 层）Math EM 为 0.230 vs Transformer 0.340；6 层 Sparse 没有做 EM，不能声称 92%。 |
| “100M 训练 1 小时，Transformer 需 2–5 小时” | 没有 100M Transformer 对照；A-Net 是 6 个独立 12M-token run，总训练约 35 分钟，但这不是同体量端到端训练。 |
| “8GB 绝对无法训练 100M Transformer” | 不成立/未验证。100M FP32+Adam 静态状态约 1.6 GB；是否 OOM 取决于 batch/seq/实现，不能断言“绝对无法”。 |
| “多节点分布式联合训练” | 未实现；当前只是独立 checkpoint/独立训练，没有跨节点通信或整合。 |
| “30B+ 必然可落地” | 未验证的推演，不是实验结论。 |

## 物理边界与失败记录

- 已证明可行：L1 2 层 + 每个任务 10 层隔离 workflow，总路径 12 层；唯一参数 99.83M，激活参数 37.50M，FLOPs 与 Transformer-12L 相同；Math/Code 打平。
- 已确认不可行：**总参数 60M + 3 个完全隔离 workflow + 1.5×/2× 深度 + 等激活参数/FLOPs** 不能同时成立。原因是 3 个隔离 workflow 使总参数约为激活路径的 3 倍，60M 总参数对应约 21M 激活，不可能等于 60M Transformer 的激活。
- 未执行：30B+、40–60 层专家流、跨节点分布式训练、DDR5 冷数据驻留。这些是未来工作，不是本轮结果。

## 未来工作（明确标注为假设）

1. **冷热数据分离**：把未激活专家流放在系统内存/NVMe，仅将 L1 与当前专家流加载到显存；需要实现和测量，当前未做。
2. **分布式增量训练**：各 workflow 独立训练后与 L1 拼接；需要通信、版本对齐和一致性验证，当前未做。
3. **30B+ 专家流扩展**：在总参数充足时把单专家流加深到 40–60 层；这是外推假设，不是已验证结论。
4. **推理显存/延迟**：单独测 isolated A-Net 的推理峰值、P50/P95、吞吐；当前只有训练峰值。

## A-Net 是什么

当前 v2 定义：

1. 前部 L1 负责输入/任务分配；
2. 独立 workflow 拥有自己的参数层；
3. 其他 workflow 可调用某个 workflow 的参数块，但只有 owner 拥有它；
4. 相同 block/input/config 的结果可复用；
5. 末端 L1 只整合最终 workflow 输出；
6. v1 使用固定、显式的调用图，不做自动发现。

~~~text
x -> PreL1 -> workflow w0 --\
          \-> workflow w1 ----> TerminalL1 -> logits
          \-> workflow w2 --/
                |
          shared parameter block
          (owned by w0, granted to others)
~~~

这与仓库早期的“Shared L1 + task-specific experts + Top-1 routing”不是同一个模型。旧实现和历史结果仍保留，但不能直接当作 v2 的证据。

## 设计澄清：一个任务一个专业 workflow

A-Net 的基本组织方式是：

> **一个任务对应一个专业 workflow；workflow 建立在共享基础层 L1 上，用该任务的专业数据训练；一个专业 workflow 可以任意长。**

因此，A-Net 的参数不是“必须切成三个固定小块”，而是按任务组织成可扩展的 workflow。workflow 可以加深、加宽或延长，只要仍保持共享 L1 和 workflow 的所有权/调用关系。

这也意味着一个关键事实：

> **如果把所有参数全部整合进一个 workflow，不再保留跨 workflow 的共享/路由，A-Net 就退化为经典 Transformer。**

所以 A-Net 与 Transformer 的差别不是“层数更多”或“参数更多”，而是：

| 维度 | 经典 Transformer | A-Net |
|---|---|---|
| 参数组织 | 一个统一堆栈 | 一个任务一个专业 workflow |
| 基础层 | 无独立共享 L1 | 共享 L1 |
| workflow 长度 | 固定整体深度 | 每个 workflow 可任意长 |
| 跨 workflow 关系 | 无 | 共享参数/调用关系/隔离 |
| 合并后的极限 | 仍是 Transformer | 全部合并即退化为 Transformer |

这也是为什么能力对比必须控制**有效深度、激活参数和 FLOPs**：如果 A-Net 只是把 12 层拆成两个 6 层 workflow，它当然会输给 12 层 Transformer；这不是共享机制的问题。当前对照 C 的 Shared-Depth A-Net 就是按“共享 L1 + 12 层任务路径”构造的。

50M pilot 为了把预算压在 50M，实际用 2 个 workflow 承载 3 个任务，每个任务仍只路由到一个专业 workflow；这是预算压缩，不改变“一任务一 workflow”的组织原则。完整实现可以让每个任务拥有独立 workflow。

## 1. 50M 语言 pilot（最新）

### 目的

在单张 RTX 5060 Laptop（8 GB）上，用 50M 唯一参数做一个可执行的三架构语言代理实验，检查：

- 参数共享是否损害能力；
- 结果缓存是否有实际收益；
- A-Net 与 Transformer 的差异是否只是激活参数/深度造成的。

### 设置

- 数据：Tiny Shakespeare（GitHub）、CPython 源码（GitHub）、本地生成四则运算/括号表达式
- tokenizer：byte-level，vocab=256
- 训练：3 seeds，每 seed 1500 步，batch 16，seq 128，约 3M tokens
- 优化器：AdamW，FP32，单卡
- 指标：held-out bits-per-byte（越低越好）

### 结果

3 seeds mean ± 95% CI：

| 模型 | text | code | math | 平均 |
|---|---:|---:|---:|---:|
| **A-Net Shared-Depth** | **2.537** | **1.736** | 1.527 | **1.933 ± 0.067** |
| Transformer-12L | 2.635 | 1.815 | **1.523** | 1.991 ± 0.033 |
| Transformer-16L | 2.638 | 1.815 | 1.568 | 2.007 ± 0.055 |
| Shared Backbone | 2.608 | 1.775 | 1.542 | 1.975 ± 0.028 |
| Sparse A-Net | 2.727 | 2.017 | 1.677 | 2.141 ± 0.107 |

### 对照 C：Shared-Depth A-Net

为了消除“只激活 2/3 workflow、有效深度只有 6 层”的混淆：

~~~text
embedding -> 8 shared blocks -> 4 task-workflow blocks -> task head
~~~

| 项目 | A-Net Shared-Depth | Transformer-12L |
|---|---:|---:|
| 唯一参数 | 50,177,024 | 37,503,168 |
| 单次激活参数 | 37,501,632 | 37,503,168 |
| 激活参数差 | 1,536（0.0041%） | — |
| 顺序深度 | 12 | 12 |
| 共享比例 | 49.5% | 0% |
| 参数引用总量 | 75,002,496 | 37,503,168 |
| FP32 唯一存储 | 200.7 MB | 150.0 MB |

- A-Net 平均低 0.058 bpc；text -0.098、code -0.079、math +0.004。
- Welch t ≈ -3.35，p ≈ 0.046。
- A-Net batch1 P50 3.00 ms，Transformer-12L 4.70 ms；batch16 14.89 vs 15.23 ms。
- A-Net 峰值显存 1.59 GB，Transformer-12L 1.51 GB。

详细报告：[SHARED_DEPTH_REPORT.md](experiments/anet_v2/lang50m/SHARED_DEPTH_REPORT.md)  
三架构报告：[REPORT.md](experiments/anet_v2/lang50m/REPORT.md)  
激活匹配对照：[ACTIVE_MATCH_REPORT.md](experiments/anet_v2/lang50m/ACTIVE_MATCH_REPORT.md)

## 2. Stage 4：共享、复用、宽度、workflow 数

540 个训练 run，0 失败；3 seeds；2×2 因子（share × cache）。

| 配置 | 唯一参数比（相对 Basic） | 精度效应 |
|---|---:|---:|
| A-Net Share，W=2 | 0.76 | W=2 容量不足，+0.3–1.1 pp |
| A-Net Share，W=3 | 0.67 | 约中性（-0.16–+0.15 pp） |
| A-Net Share，W=4 | 0.63 | 约中性（-0.10–+0.13 pp） |

- **Reuse 的精度效应严格为 0**：训练每 step 清空 cache，cache 只影响推理。
- **交互项为 0**：共享和复用没有额外精度协同。
- W=2 对 3 个任务容量不足（平均 83.9% vs W=3 95.8%、W=4 95.9%）。
- W=4 相对 W=3 没有稳定精度收益，但唯一参数继续增加。

详细报告：[STAGE4_REPORT.md](experiments/anet_v2/STAGE4_REPORT.md)

## 3. Stage 2：共享的收益与隔离

315 个耦合 run + 315 个隔离 run。

隔离版（detach_both）相对 Independent：

- pair Share 平均 **-0.07 pp**；
- task0 低约 0.22–0.29 pp；
- 回归 MSE +0.0020–0.0022；
- 唯一参数 **0.656×**；
- 激活参数 **1.479×**。

隔离版相对耦合版：

- pair Share **+1.13 pp**；
- dense Share **+1.68 pp**；
- adapter **+1.49 / +1.75 pp**。

代价：detach_both 会冻结共享块，共享块不再接收梯度。若要共享块继续学习，应改用 detach_aux 或梯度手术。

详细报告：[STAGE2_ISOLATED_REPORT.md](experiments/anet_v2/STAGE2_ISOLATED_REPORT.md)

## 4. Stage 3：结果缓存

288 个 cache stream，0 失败。

正确性：

- cache on/off 输出最大绝对差 **0.0**；
- 参数更新后失效 **100%**；
- H2D **0**；
- 所有输入在 cuda:0。

性能：

| 场景 | 结果 |
|---|---|
| batch=1，0–75% 重复 | cache 均更慢，1.14–1.85× |
| 2048 宽，batch=256，75% 重复 | Reuse **0.68×**，Full **0.67×** |
| 2048 宽，batch=256，50% 重复 | Full **0.98×**，接近盈亏平衡 |
| 缓存内存 | 最高 256 MB，最多 786 次 eviction |
| 查找成本 | 约 1.33 µs/次 |

结论：缓存不是普遍加速；只有大宽度、大 batch、高重复率才有净收益。默认关闭。

详细报告：[STAGE3_REPORT.md](experiments/anet_v2/STAGE3_REPORT.md)

## 5. Stage 1：基础架构 scaling

360 个 run，0 失败；3 seeds；h64/h256/h1024/h2048。

相对 Independent：

| 模型 | 平均精度 | 唯一参数比 | 激活参数比 | batch1 P50 | 大 batch 吞吐 |
|---|---:|---:|---:|---:|---:|
| A-Net Basic | -0.067 pp | 0.967 | 0.996 | 2.65× | 0.61× |
| A-Net Full | -0.216 pp | 0.651 | 0.996 | 4.75× | 0.47× |
| Shared Backbone | -1.367 pp | 0.341 | 1.000 | 相近 | 相近 |

- A-Net 在精度上接近 Independent，但**不是计算/延迟胜利**。
- Shared Backbone 存储最少，但精度损失最大。
- A-Net 处在“比 Independent 省存储、比 Shared 保精度”的中间位置。

详细报告：[STAGE1_REPORT.md](experiments/anet_v2/STAGE1_REPORT.md)

## 成本画像：推理、显存、训练时间

这一节只写实测数据，不把“条件性延迟”写成“全面成本优势”。

### 50M 对照 C

| 指标 | A-Net Shared-Depth | Transformer-12L | 比值/结论 |
|---|---:|---:|---|
| batch1 P50 | **3.00 ms** | 4.70 ms | **0.64×（A-Net 更快）** |
| batch16 P50 | **14.89 ms** | 15.23 ms | 0.98×（基本持平） |
| 单次激活参数 | 37.50M | 37.50M | 相同 |
| 有效深度 | 12 | 12 | 相同 |
| 峰值显存 | 1.59 GB | **1.51 GB** | 1.05×（A-Net 更高） |
| 唯一参数存储（FP32） | 200.7 MB | **150.0 MB** | 1.34×（A-Net 更高） |
| 训练时间/seed | 100.3 s | 101.7 s | 0.99×（无实质差异） |

- **推理**：在这个 pilot 中 A-Net 的 batch1 延迟更低（3.00 vs 4.70 ms），但 batch16 基本持平。两者深度/激活参数相同，因此这不是“FLOPs 更少”，更可能是实现/启动开销差异。
- **训练显存**：要分版本，不能一概而论。见下表。
- **训练时间**：Shared-Depth 与 Transformer-12L 基本持平（100.3 vs 101.7 s）；Sparse A-Net 更快（94.6 s），但它有效深度只有 6 层、能力更差。

### 训练显存：Sparse 与 Shared-Depth 不同

| 模型 | 唯一参数 | 单次激活参数 | 训练峰值显存 | 训练时间/seed |
|---|---:|---:|---:|---:|
| Sparse A-Net | 50.18M | 37.50M | **1.69 GB** | 94.6 s |
| Transformer-16L | 49.92M | 49.92M | 1.98 GB | 121.7 s |
| Shared Backbone | 50.18M | 50.18M | 2.11 GB | 120.8 s |
| Shared-Depth A-Net | 50.18M | 37.50M | 1.59 GB | 100.3 s |
| Transformer-12L | 37.50M | 37.50M | **1.51 GB** | 101.7 s |

- **Sparse A-Net 的训练显存确实更低**：1.69 GB vs Transformer-16L 1.98 GB、Shared Backbone 2.11 GB。原因是它只激活 2/3 workflow，激活值更少；但它的有效深度只有 6 层，能力也更差。
- **Shared-Depth A-Net 没有训练显存优势**：1.59 GB vs Transformer-12L 1.51 GB。两者激活参数/深度相同，但 A-Net 唯一参数多 12.7M（另一个 workflow tail），优化器状态更大。
- 所以“训练显存更低”是**Sparse A-Net 的代价性优势**，不是 Shared-Depth A-Net 的优势，也不能直接归因于“共享”本身。

### Stage 1 全矩阵

| 模型 | batch1 P50 | 大 batch 吞吐 | 训练时间 |
|---|---:|---:|---:|
| A-Net Basic | 2.65× Independent | 0.61× | 1.21× |
| A-Net Full | 4.75× Independent | 0.47× | 1.33× |

在 Stage 1 中，A-Net **推理更慢、吞吐更低、训练更慢**；唯一稳定优势是唯一参数存储更低（Basic 0.967×、Full 0.651×）。

### 结果复用的成本

- 75% 重复、2048 宽、batch=256：Reuse 0.68×、Full 0.67×，这是**条件性**推理收益。
- batch=1：cache 反而慢 1.14–1.85×。
- 缓存内存最高 256 MB，查找约 1.33 µs/次。
- 所以 cache 是“高重复/大 batch 才回本”的优化，不是普遍推理成本优势。

**结论：当前证据支持“50M Shared-Depth pilot 的 batch1 推理延迟优势”和“共享的唯一参数收益”；不支持“普遍显存更省”或“训练更快”。**

## 6. 规模边界：2B 不可在本机执行

本机实测：

- GPU：RTX 5060 Laptop，7.93 GB 专用显存
- RAM：15.18 GB total / 3.66 GB available
- 2B 全参数训练仅权重/梯度/Adam 就约 **32 GB**，不含激活
- 2B FP16 推理约 4 GB，但没有训练好的 2B A-Net/语言栈

推荐：

- **最终能力验证：1B 唯一参数 + 20B–100B tokens + 3 seeds + 8×H100 级**
- **本机可执行：50M pilot**
- 50M 不是 MMLU/GSM8K/HumanEval 能力结论

可行性报告：[FEASIBILITY.md](experiments/anet_v2/scale2b/FEASIBILITY.md)  
规模建议：[RECOMMENDATION.md](experiments/anet_v2/scale2b/RECOMMENDATION.md)

## 7. 当前支持的结论

支持：

1. 参数唯一性、所有权、引用、版本、缓存失效机制可正确实现。
2. 参数共享能降低唯一参数：W=3 约 0.67×，W=4 约 0.63×。
3. 在 50M pilot 中，**Shared-Depth A-Net 在同深度/同激活参数下略优于 Transformer-12L**（p≈0.046，pilot 级）。
4. 隔离（detach_both）显著改善多任务负迁移，代价是冻结共享块。
5. 缓存正确，且在高重复/大 batch/大宽度下能降低执行次数和延迟。
6. Sparse A-Net 的训练峰值显存和训练时间更低（1.69 GB / 94.6 s vs Transformer-16L 1.98 GB / 121.7 s），但这是以只激活 2/3 workflow、有效深度 6 层、能力更差为代价的。

不支持：

1. “A-Net 无条件优于 Transformer”；
2. “A-Net 计算更少/延迟更低”；
3. “2B/1B 语言能力已超过 Transformer”；
4. “结果复用显著提升训练能力”；
5. “Sparse A-Net 的失败证明共享机制无效”。

## 8. Quick start

### Install

~~~bash
pip install -r requirements.txt
~~~

测试过的 CUDA 13.0 构建：

~~~bash
pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install numpy matplotlib pytest
~~~

### Tests

~~~bash
python -m pytest -q -p no:cacheprovider tests
~~~

当前：**75 passed**。

### Stage 1

~~~bash
python -m experiments.anet_v2.runners.run_stage1 \
  --out-dir experiments/anet_v2/results/stage1 --resume
~~~

### Stage 2（隔离默认 detach_both）

~~~bash
python -m experiments.anet_v2.runners.run_stage2 \
  --out-dir experiments/anet_v2/results/stage2_isolated_both \
  --gradient-isolation detach_both --resume
~~~

### Stage 3 缓存

~~~bash
python -m experiments.anet_v2.runners.run_stage3 \
  --out-dir experiments/anet_v2/results/stage3 --resume
~~~

### Stage 4 因子实验

~~~bash
python -m experiments.anet_v2.runners.run_stage4 \
  --widths 128 512 2048 --workflows 2 3 4 \
  --models anet_basic anet_share anet_reuse anet_full \
  --correlations low medium high \
  --seeds 2026 2027 2028 2029 2030 \
  --out-dir experiments/anet_v2/results/stage4 --resume
~~~

### 50M 三架构 pilot

~~~bash
python -m experiments.anet_v2.lang50m.run \
  --steps 1500 --seeds 2026 2027 2028 \
  --archs transformer shared anet \
  --out-dir experiments/anet_v2/lang50m/results/run_50m
~~~

### 50M 激活匹配 / 共享深度对照

~~~bash
python -m experiments.anet_v2.lang50m.run \
  --steps 1500 --seeds 2026 2027 2028 \
  --archs transformer37 \
  --out-dir experiments/anet_v2/lang50m/results/run_50m_active37

python -m experiments.anet_v2.lang50m.run \
  --steps 1500 --seeds 2026 2027 2028 \
  --archs anet_shared_depth \
  --out-dir experiments/anet_v2/lang50m/results/run_50m_shared_depth
~~~

### 2B 可行性检查

~~~bash
python experiments/anet_v2/scale2b/feasibility.py
~~~

## 9. Repository layout

~~~text
.
├── README.md
├── multitask_anet_demo.py              # legacy demo, not v2
├── experiments/
│   ├── scaling/                        # legacy 2D scaling study + historical artifacts
│   └── anet_v2/
│       ├── core/                       # blocks, registry, workflows, cache, accounting
│       ├── runners/                    # Stage 1–4 runners
│       ├── analysis/                   # aggregation / figures
│       ├── results/                    # Stage 1–4 results
│       ├── STAGE1_REPORT.md
│       ├── STAGE2_REPORT.md
│       ├── STAGE2_ISOLATED_REPORT.md
│       ├── STAGE3_REPORT.md
│       ├── STAGE4_REPORT.md
│       ├── lang50m/                    # 50M Transformer/Shared/A-Net pilot
│       │   ├── model.py
│       │   ├── run.py
│       │   ├── REPORT.md
│       │   ├── ACTIVE_MATCH_REPORT.md
│       │   ├── SHARED_DEPTH_REPORT.md
│       │   └── results/
│       └── scale2b/                    # 2B feasibility, not a 2B run
└── tests/
~~~

## 10. 历史修正

早期 README 声称 A-Net 激活 Independent 的 17%（8,770 / 51,462）。这是一个特定配置下的真实数字，但混用了 expert width 和 comparison denominator，不能推广。

在受控实验中：

- Stage 1 A-Net Basic 激活参数约 **0.996×** Independent；
- 唯一参数约 **0.967×**；
- A-Net Full 唯一参数约 **0.651×**；
- 早期 A-Net 约 50% 参数的说法只适用于旧共享 L1 + experts 实现。

历史 PDF 和旧结果保留，不覆盖。

## 11. 局限

- 50M 语言 pilot 只有 3 seeds、3M tokens、char-level 三域数据；不是语言能力 benchmark。
- Stage 1–4 是 2D 合成任务；不能直接外推到真实语言/视觉任务。
- 1B/2B 未执行；2B 在本机不可行。
- 缓存收益依赖 batch、宽度、重复率；不是普遍加速。
- detach_both 冻结共享块；共享块可训练性需要单独实验。
- 单 GPU、单 PyTorch/CUDA 栈。
- A-Net Shared-Depth 的正结果仍是 pilot 级，p≈0.046，需要更多 seeds 和更大 token 预算复现。

## 最终免责与复现声明

本项目所有结论均受限于单卡 RTX 5060（8GB）及有限算力预算，结果仅供参考，不作为最终工业定论。恳请拥有充足算力（如 8×H100）的机构复现验证。如有逻辑漏洞或数据相悖，欢迎在 GitHub Issues 指正。

## License

See [LICENSE](LICENSE).

当前实验总结 PDF：[A-Net_Experiment_Summary.pdf](A-Net_Experiment_Summary.pdf)  
历史报告 PDF：[archive/A-Net_Experiment_Report_V3_historical.pdf](archive/A-Net_Experiment_Report_V3_historical.pdf)  
Word 版本：[A-Net_Experiment_Summary.docx](A-Net_Experiment_Summary.docx)
