# > ⚠️ \*\*项目声明 (Project Disclaimer)\*\*

# > 

# > 这是一个探索性的小型实验验证项目，仅用于测试多任务学习下硬路由（Top-1 Hard Routing）架构的潜力，\*\*并非完整大模型的结论\*\*。

# > 本人是学生，纯属业余时间的学术探索与自嗨。开源仅供交流与学习，实验受限于算力（RTX 5060 Laptop）及模型规模，不代表工业级落地的最终结果。欢迎各位前辈大佬指正与讨论，\*\*求轻喷\*\*。

# 

# \---

# A-Net

**基于动态子网提取与 Top-1 硬路由的多任务学习架构**

[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[!\[CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia\&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[!\[Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
\[!\[Status](https://img.shields.io/badge/status-research%20prototype-6B7280)]()

> 用一个共享浅层接口提取通用特征，再根据 Task ID 只激活对应的一个专家。  
> 不遍历全部专家，不做 all-experts 拼接，不产生无效的稠密计算。

\---

## 1\. 简介

A-Net 是一种面向多任务学习的**动态子网提取架构**。它将模型显式拆分为三个部分：

1. **共享 L1 接口**：所有任务共享的浅层特征提取器；
2. **任务专家（Experts）**：每个任务拥有一组独立的非线性专家参数；
3. **Top-1 硬路由器**：根据输入的 Task ID，只选择并执行一个专家。

核心前向过程可以概括为：

\~\~\~text
x ──► Shared L1 ──┬──► Expert 0
                   ├──► Expert 1   ◄── Task ID k 只选择 1 个专家
                   └──► Expert 2
\~\~\~

\~\~\~python
def forward(self, x, task\_id):
    shared\_features = self.l1(x)
    selected\_expert = self.experts\[task\_id]  # Top-1 硬路由
    return selected\_expert(shared\_features)
\~\~\~

未选中的专家**不参与前向计算，也不会收到当前 batch 的梯度**。因此，A-Net 在保留任务专属非线性容量的同时，避免了“所有专家都算一遍”的浪费。

> 当前实现假设 Task ID 在训练和推理时已知，路由是确定性的 Task-ID 索引；它不是学习得到的 soft gating，也不包含负载均衡器。

\---

## 2\. 核心痛点

传统多任务模型通常面临两个问题：

### 2.1 稠密计算带来的算力浪费

大模型或 MoE 类架构如果对每个输入都执行全部专家，参数量上去了，但单样本有效计算量也同步膨胀。很多任务实际上不需要经过所有子网络。

A-Net 的做法是：**参数可以保存多个专家，但每个样本只执行一个专家**。这把“总参数”和“单样本激活参数”拆成两个独立指标，让模型可以为了容量而扩展存储，同时保持稀疏的计算路径。

### 2.2 共享主干的特征干扰与灾难性遗忘

“共享主干 + 任务专属线性头”在简单任务上很有效，但主干必须同时满足所有任务的特征需求。任务差异变大时，会出现：

* 任务间特征干扰，某任务的梯度破坏另一任务所需的表示；
* 线性任务头容量不足，无法拟合复杂的非线性决策边界；
* 某个随机种子下性能明显塌陷，方差变大。

A-Net 通过**共享浅层 + 任务专属非线性专家**解决这一矛盾：底层保留通用表示，深层由各任务独立拥有，避免任务之间直接争夺同一组深层特征。

\---

## 3\. 实验结果

### 3.1 复杂任务设置

为了避免简单任务“全部接近 100%”导致的饱和，实验使用了 3 个更难的非线性二分类任务：

|任务|决策边界|
|-|-|
|高频正弦|x2 > 0.55·sin(3x1) + 0.28·sin(7.5x1)|
|四同心环|4 个等量环带，标签按 0/1/0/1 交替|
|旋转棋盘|25° 旋转后的 1.1 宽网格奇偶分类|

* 每个任务：**5,000 训练 / 1,500 测试**
* 标签噪声：**2%**
* 随机种子：**2026 / 2027 / 2028**
* 训练：Adam，lr=1e-3，80 epochs，batch=256
* 网络宽度：hidden=128；A-Net 专家为 128-64-2
* 运行设备：CPU 8 线程 / NVIDIA RTX 5060 Laptop GPU（CUDA 13.0）
* CPU 与 GPU 使用完全相同的配置

### 3.2 重点结果

|方案|总参数|激活参数|复杂任务平均准确率（CPU）|跨种子 std|
|-|-:|-:|-:|-:|
|独立基线|51,462|51,462|96.23%|0.19|
|共享主干 + 线性头|17,670|17,670|94.49%|1.25|
|**A-Net（Top-1 硬路由）**|**25,542**|**8,770**|**95.84%**|**0.14**|

> \*\*A-Net 仅激活独立基线 17.0% 的参数（8,770 / 51,462），达到 95.84% 的准确率，跨种子标准差仅 0.14。\*\*

关键对比：

* **对比独立基线**：A-Net 准确率仅低 **0.39 个百分点**，但总参数只有其 **49.6%**，单样本激活参数只有 **17.0%**。
* **对比共享主干**：A-Net 准确率高 **1.35 个百分点**，标准差从 **1.25 降到 0.14**。共享主干在某个种子的“四同心环”任务上跌到 **91.2%**，而 A-Net 保持稳定。
* **CPU vs GPU**：A-Net 的 CPU/GPU 平均准确率分别为 **95.84% / 95.79%**，差异仅 **0.05 个百分点**。

!\[复杂任务准确率对比](assets/complex\_accuracy.png)

*图：简单任务与复杂任务平均准确率对比。误差线为 3 个随机种子的标准差。*

### 3.3 如何解读

在简单任务上，三种结构都能达到约 99.4%–99.5%，差异不足以区分架构。  
但当任务变复杂后：

* 共享主干的线性任务头开始成为瓶颈；
* 独立基线仍然最准，但保存了三套完整网络；
* A-Net 用共享 L1 保留通用特征，用独立专家吸收任务差异，在**准确率、总参数、激活计算和稳定性**之间取得了更好的平衡。

\---

## 4\. 快速开始

### 4.1 安装

\~\~\~bash
pip install torch numpy
\~\~\~

如果你的机器支持 CUDA，可使用与 RTX 5060 / Blackwell 匹配的 CUDA 13.0 轮子：

\~\~\~bash
pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install numpy
\~\~\~

### 4.2 运行复杂任务实验

\~\~\~bash
# 默认 auto：检测到 CUDA 时使用 GPU，否则使用 CPU
python complex\_tasks\_test.py

# 显式指定设备
python complex\_tasks\_test.py --device cpu
python complex\_tasks\_test.py --device cuda
\~\~\~

脚本会自动运行 3 个复杂任务 × 3 个随机种子 × 3 种架构，并输出：

* 每个任务的测试准确率（mean ± std）
* 总参数量与单样本激活参数量
* CPU / GPU 结果
* 平均训练耗时

### 4.3 其他脚本

\~\~\~bash
# 简单任务主实验（原始 3 个任务）
python multitask\_anet\_demo.py --device auto

# CUDA 兼容性 + GPU 基准
python cuda\_compat\_test.py

# CPU 推理延迟基准
python benchmark\_latency.py
\~\~\~

### 4.4 项目结构

\~\~\~text
.
├── multitask\_anet\_demo.py    # 基础模型定义与简单任务实验
├── complex\_tasks\_test.py     # 复杂任务、3 seeds、CPU/GPU 对比
├── cuda\_compat\_test.py       # CUDA 兼容性与 GPU 性能测试
├── benchmark\_latency.py      # CPU 延迟基准
├── assets/
│   └── complex\_accuracy.png
└── README.md
\~\~\~

\---

## 5\. 声明

### 5.1 CUDA 兼容性

A-Net 的实现已在本机 **NVIDIA GeForce RTX 5060 Laptop GPU** 上完成实际验证：

* PyTorch **2.14.1+cu130**
* Compute Capability **12.0（sm\_120）**
* CUDA 13.0 / cuDNN 9.24.0
* FP16、BF16、FP16 autocast、cuDNN 卷积均通过
* A-Net 路由在 GPU 上严格只更新被选中的专家
* 无 CUDA DLL 缺失、驱动不匹配或 OOM

### 5.2 CPU / GPU 精度

> \*\*声明\*\*：A-Net 极度兼容 CUDA。简单任务 50-epoch 主实验的 CPU/GPU 指标完全一致；复杂任务 CPU/GPU 平均准确率差异不超过 \*\*0.05 个百分点\*\*，属于浮点运算的非确定性，不构成精度回退。

已验证的复杂任务结果：

|指标|CPU|GPU|
|-|-:|-:|
|独立基线平均准确率|96.23%|96.20%|
|共享主干平均准确率|94.49%|94.53%|
|A-Net 平均准确率|**95.84%**|**95.79%**|

### 5.3 性能提示

CUDA 兼容并不等于“任何情况下都更快”。本项目的模型很小，实测表明：

* batch=1 的 GPU 延迟反而高于 CPU（kernel launch / WDDM 开销占主导）；
* batch=4096 时 GPU 吞吐可达到约 **37M–41M samples/s**；
* 对于当前规模的网络，GPU 更适合大 batch 吞吐或更大模型，而不是单样本在线推理。

### 5.4 结果口径

* 所有准确率均为测试集 Top-1 准确率；
* 复杂任务结果为 3 个随机种子的 mean ± std；
* “激活参数”指单次 forward 中实际参与计算的参数，不包含未选中的专家；
* 延迟基准仅用于结构对比，不含真实服务中的网络、序列化和预处理开销。

\---

**A-Net：让多任务模型只计算它真正需要的那一部分。**

