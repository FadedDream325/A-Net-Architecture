# 对照 C：共享深度 A-Net vs Transformer-12L

- 目的：修正之前“A-Net 只激活 2/3 workflow、有效深度 6 层”的结构混淆。
- 设计：每个任务路径固定 12 层：8 个共享 block + 4 个 workflow 专属 block。
- 对照：Transformer-12L，同样 12 层、同样激活参数/FLOPs。
- 数据、seeds、steps、batch、seq、优化器、精度与 50M pilot 完全一致。
- 原始数据：experiments/anet_v2/lang50m/results/run_50m_shared_depth/raw_runs.jsonl

## 1. 参数与结构

| 项目 | A-Net Shared-Depth | Transformer-12L |
|---|---:|---:|
| 唯一参数 | 50,177,024 | 37,503,168 |
| 单次激活参数 | **37,501,632** | **37,503,168** |
| 激活参数差 | 1,536（0.0041%） | — |
| 顺序深度 | **12 层** | **12 层** |
| 共享 block | 8 个（49.5% unique） | 无 |
| 任务专属 block | 2 个 workflow × 4 个 | 无 |
| 参数引用总量 | 约 75.0M | 37.5M |
| FP32 唯一存储 | 200.7 MB | 150.0 MB |

A-Net 路径：

~~~text
embedding -> 8 shared blocks -> 4 task-workflow blocks -> task head
~~~

每个任务只走一条 12 层路径；共享块同时服务两个 workflow。  
与之前的 sparse A-Net 不同，这里**没有把有效深度砍成 6 层**。

## 2. 结果：bits-per-byte（越低越好）

3 seeds mean ± 95% CI：

| 模型 | text | code | math | 平均 |
|---|---:|---:|---:|---:|
| **A-Net Shared-Depth** | **2.537** | **1.736** | 1.527 | **1.933 ± 0.067** |
| Transformer-12L | 2.635 | 1.815 | **1.523** | **1.991 ± 0.033** |
| 差值（A-Net − 对照） | -0.098 | -0.079 | +0.004 | **-0.058** |

- A-Net 在 text 和 code 上更好；math 基本打平（+0.004，可忽略）。
- 平均低 **0.058 bpc**。
- Welch t ≈ -3.35，**p ≈ 0.046**（3 seeds，pilot 级）。
- 这次是**相同 12 层深度、相同激活参数、相同 FLOPs** 的比较；之前的深度混淆被消除了。

## 3. 工程性能

| 模型 | 训练/seed | batch1 P50 | batch16 P50 | 峰值显存 |
|---|---:|---:|---:|---:|
| A-Net Shared-Depth | 100.3 s | **3.00 ms** | **14.89 ms** | 1.59 GB |
| Transformer-12L | 101.7 s | 4.70 ms | 15.23 ms | 1.51 GB |

- A-Net batch1 延迟明显更低；batch16 略低。
- 峰值显存略高（唯一参数 50M vs 37.5M）。
- 这里也不是“A-Net 计算更少”：激活参数和深度都匹配，差异主要来自共享结构和实现路径。

## 4. 这说明什么

1. **之前 A-Net 更差，很大程度上是因为 sparse A-Net 把有效深度从 12 层降到了 6 层。**
2. 把参数组织成“共享深度 trunk + 任务专属 tail”，在相同深度/激活参数下，A-Net 可以**不弱于甚至优于** Transformer-12L。
3. 参数共享不一定损害表达能力；关键是不能用“减少深度/激活路径”来换共享。
4. 这支持 A-Net 的共享机制，但仅限于这个 50M char-level pilot。

## 5. 与之前结果的关系

| 模型 | 激活参数 | 有效深度 | 平均 bpc |
|---|---:|---:|---:|
| A-Net Shared-Depth | 37.50M | 12 | **1.933** |
| Transformer-12L | 37.50M | 12 | 1.991 |
| Transformer-16L | 49.92M | 16 | 2.007 |
| Shared Backbone | 50.18M | 16 | 1.975 |
| Sparse A-Net | 37.50M | **6** | 2.141 |

- Sparse A-Net 的差，主要是浅路径/路由造成的；不是“共享”本身的判决。
- Shared-Depth A-Net 是三者中最好的，而且唯一参数 50M、激活参数 37.5M。

## 6. 局限

- 只有 3 seeds；p≈0.046 是 pilot 级，接近 0.05，不能当作最终显著性。
- 3M tokens、char-level 三域任务，模型远未收敛。
- 这里的 A-Net 是简化的“shared trunk + task tail”，不是原始多 workflow terminal 结构。
- 没有 SFT、没有 MMLU/GSM8K/HumanEval。
- 没有测试结果缓存；cache 收益仍未验证。
- 唯一参数 50M 中有一部分是另一个 workflow 的 tail，对当前 task 不激活；这是共享/专属折中，不是纯浪费。

## 7. 结论

**对照 C 支持“参数共享可以在不牺牲深度/激活计算的情况下保持能力”**，并且在这个 50M pilot 中，A-Net Shared-Depth 略优于同深度、同激活参数的 Transformer-12L。

但它仍是 pilot：

- 不能外推到 1B/2B；
- 不能宣称 MMLU/GSM8K/HumanEval 能力；
- 不能把 0.058 bpc 的差异当作最终定论。

下一步应做：

1. 更多 seeds（5–10）；
2. 更长 token 预算；
3. dense/shared-depth 消融；
4. 在 1B 规模验证共享深度是否仍成立。
