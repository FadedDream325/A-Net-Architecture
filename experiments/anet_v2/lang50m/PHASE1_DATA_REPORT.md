# Phase 1 数据管线报告（30M 数学 + 30M 编程）

- 分支：experiment/anet-v2
- 脚本：experiments/anet_v2/lang50m/phase1_data.py
- 清单：experiments/anet_v2/lang50m/data60m/manifest.json
- 日志：experiments/anet_v2/lang50m/phase1_data.log

## 1. 数据池

| 池 | 字节数 |
|---|---:|
| math_train | 24,000,000 |
| math_val | 3,000,000 |
| math_test | 3,000,000 |
| code_train | 24,000,000 |
| code_val | 2,980,930 |
| code_test | 3,036,456 |
| 合计 | 60,017,386 |

- 数学：约 30M bytes，模板为 Q/A 四则运算表达式。
- 编程：约 30M bytes，来自 CPython 源码和本地 Anaconda 库中的 1,877 个 .py 文件。
- 数学按 RNG 区间切分；编程按文件级 80/10/10 切分，避免同一文件同时出现在 train/val。
- tokenizer：当前仍是 byte-level；30M bytes 不等于 30M BPE tokens。正式 Phase 1 需要明确 tokenizer 并重新计数。

## 2. 泄漏检查

| 域 | 256-byte 重叠 | train bytes | val bytes |
|---|---:|---:|---:|
| math | 0 | 24,000,000 | 3,000,000 |
| code | 0 | 24,000,000 | 2,980,930 |

code 的 32-byte 公共语法片段不作为泄漏；256-byte 精确重叠为 0。

## 3. 领域比例布局

batch=16，seq=128，每 batch 2,048 tokens：

| 数学比例 | batch 内混合（math/code 样本） | 边界切换（math/code 样本） | 总 tokens |
|---:|---:|---:|---:|
| 100% | 16 / 0 | 16 / 0 | 2,048 |
| 75% | 12 / 4 | 12 / 4 | 2,048 |
| 50% | 8 / 8 | 8 / 8 | 2,048 |
| 25% | 4 / 12 | 4 / 12 | 2,048 |
| 0% | 0 / 16 | 0 / 16 | 2,048 |

- 比例现在是每个 batch 精确匹配，不是随机近似。
- batch 模式表示域在 batch 内交错；boundary 模式表示域在连续 segment 中切换。真正的顺序/索引布局将在训练 sampler 中实现并测试。

## 4. 当前可确认与不可确认

可确认：

- 60M bytes 数据池已生成；
- train/val/test 文件级/区间级隔离；
- 256-byte 泄漏为 0；
- 5 个比例和 2 种混合模式的 batch 计数可复现；
- 数据文件 SHA256 已写入 manifest。

不可确认：

- 这不是 30M + 30M BPE tokens；只是 byte-level 估算；
- 当前还没有训练 sampler，不能证明 batch 内交错/边界切换对梯度的实际影响；
- 还没有数学/编程准确率评测；
- 还没有训练结果；
- code 池来自本地 CPython/Anaconda，不是纯 GitHub 数据集；正式报告需注明来源。

## 5. 对 Phase 1 的影响

- Phase 1 训练脚本必须从 data60m/manifest.json 读取池和 checksum；
- 每个 run 的 token 预算需固定（建议先 6M-token pilot，不直接 60M）；
- 训练 sampler 必须实现并测试两种模式：
  - batch 内交错；
  - batch 间边界切换；
- 评测必须分开报告 math/code BPC，并新增数学答案准确率/代码可解析率；
- 正式实验前需要确认 tokenizer 与 token 计数口径。

## 6. 结论

Phase 1 数据池和比例布局的 smoke 通过，但训练尚未开始。  
下一步是实现带 warmup 的训练 harness：读取 manifest、按比例构造 batch、记录 train/val BPC 与显存/吞吐，然后跑 6M-token pilot。
