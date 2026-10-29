"""
多任务学习（Multi-task Learning）三种架构对比：
1) 独立基线：每个任务单独训练一个 MLP
2) 共享主干 + 任务专属线性头
3) A-Net 极简版：共享 L1 特征提取器 + 3 个专家，Task-ID Top-1 硬路由

运行：
    python multitask_anet_demo.py --device auto
    python multitask_anet_demo.py --device cuda
    python multitask_anet_demo.py --device cpu

配套延迟测试（CPU）：
    python benchmark_latency.py

仅依赖 PyTorch / NumPy；支持 CPU / CUDA，CPU 为单线程，几十秒内可跑完。
"""

from __future__ import annotations

import argparse
import math
import random
import time
import unicodedata
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 全局实验配置
# ---------------------------------------------------------------------------
SEED = 42
N_TRAIN_PER_TASK = 2400
N_TEST_PER_TASK = 800
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 64
EXPERT_HIDDEN_DIM = 32
NUM_TASKS = 3

# 单核 CPU 实验：限制 PyTorch 线程数，避免额外线程影响计时与公平性
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # 若互操作线程池已经初始化，忽略即可
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. 三个 2D 非线性合成分类任务
# ---------------------------------------------------------------------------
def make_sine_data(n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """任务 1：x2 > sin(2*x1) 为正类。"""
    x = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
    y = (x[:, 1] > np.sin(2.0 * x[:, 0])).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_circle_data(n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """任务 2：圆内为 0 类，圆外为 1 类。

    采用平衡的极坐标采样：内圈 r ~ U(0, 0.95)，外圈 r ~ U(1.05, 2.0)，
    避免均匀方形采样带来的类别极度不平衡。
    """
    n_inner = n // 2
    n_outer = n - n_inner

    r_inner = rng.uniform(0.0, 0.95, size=n_inner)
    r_outer = rng.uniform(1.05, 2.0, size=n_outer)
    theta_inner = rng.uniform(0.0, 2.0 * math.pi, size=n_inner)
    theta_outer = rng.uniform(0.0, 2.0 * math.pi, size=n_outer)

    inner = np.stack(
        [r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)], axis=1
    )
    outer = np.stack(
        [r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)], axis=1
    )
    x = np.concatenate([inner, outer], axis=0).astype(np.float32)
    y = np.concatenate(
        [np.zeros(n_inner, dtype=np.int64), np.ones(n_outer, dtype=np.int64)]
    )

    # 打乱内圈 / 外圈的拼接顺序
    perm = rng.permutation(n)
    return torch.from_numpy(x[perm]), torch.from_numpy(y[perm])


def make_xor_data(n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """任务 3：异或，x1*x2 > 0 为正类。"""
    x = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
    y = (x[:, 0] * x[:, 1] > 0.0).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


TaskSplit = tuple[torch.Tensor, torch.Tensor]
TaskData = tuple[TaskSplit, TaskSplit]  # (train, test)


def build_datasets() -> tuple[list[TaskData], list[TaskData]]:
    """返回 (训练任务列表, 测试任务列表)，每个任务均为 (x, y)。"""
    train_rng = np.random.default_rng(SEED)
    test_rng = np.random.default_rng(SEED + 1)

    makers = [make_sine_data, make_circle_data, make_xor_data]
    train_data = [maker(N_TRAIN_PER_TASK, train_rng) for maker in makers]
    test_data = [maker(N_TEST_PER_TASK, test_rng) for maker in makers]
    return train_data, test_data


# ---------------------------------------------------------------------------
# 2. 三种模型架构
# ---------------------------------------------------------------------------
def make_mlp(in_dim: int = 2, hidden_dim: int = HIDDEN_DIM, out_dim: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class IndependentBaseline(nn.Module):
    """3 个完全独立的 MLP，参数不共享。"""

    def __init__(self, num_tasks: int = NUM_TASKS, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.num_tasks = num_tasks
        self.mlps = nn.ModuleList([make_mlp(hidden_dim=hidden_dim) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.mlps[task_id](x)


class SharedTrunkWithHeads(nn.Module):
    """共享两层 MLP 主干，每个任务只有一个线性分类头。"""

    def __init__(self, num_tasks: int = NUM_TASKS, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.num_tasks = num_tasks
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        features = self.trunk(x)
        return self.heads[task_id](features)


class ANetHardRouting(nn.Module):
    """A-Net 极简版：共享 L1 + Task-ID Top-1 硬路由专家。

    与共享主干方案的关键区别：
    - L1 是共享的浅层特征提取器；
    - 每个任务拥有自己的完整专家网络（非线性层 + 输出层）；
    - 前向时根据 task_id 直接索引唯一选中的专家。

    注意：这里不遍历所有专家、不做 all-experts 拼接，也不使用加权求和。
    未被选中的专家既不参与前向计算，也不会收到该 batch 的梯度。
    """

    def __init__(
        self,
        num_tasks: int = NUM_TASKS,
        hidden_dim: int = HIDDEN_DIM,
        expert_hidden_dim: int = EXPERT_HIDDEN_DIM,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        # 共享底层接口 L1：2 -> hidden_dim
        self.l1 = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        # 3 个并行专家：hidden_dim -> expert_hidden_dim -> 2
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(expert_hidden_dim, 2),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"task_id 必须在 [0, {self.num_tasks}) 内，收到 {task_id}")

        shared_features = self.l1(x)

        # Top-1 硬路由核心：根据 Task ID 只取一个专家并计算。
        # 其他专家在此处完全不被调用、不产生中间激活。
        selected_expert = self.experts[task_id]
        return selected_expert(shared_features)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. 训练与评估
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_data: Sequence[TaskSplit],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    device: torch.device | None = None,
) -> None:
    """所有架构共用相同的 Adam、epoch 数、batch 大小与任务采样方式。"""
    device = device or torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _epoch in range(epochs):
        for task_id, (x, y) in enumerate(train_data):
            permutation = torch.randperm(x.size(0))
            for start in range(0, x.size(0), batch_size):
                indices = permutation[start : start + batch_size]
                xb = x[indices].to(device, non_blocking=device.type == "cuda")
                yb = y[indices].to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb, task_id)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()


@torch.no_grad()
def evaluate_per_task(
    model: nn.Module,
    test_data: Sequence[TaskSplit],
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cpu")
    model.eval()
    accuracies: list[float] = []
    for task_id, (x, y) in enumerate(test_data):
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        logits = model(x, task_id)
        predictions = logits.argmax(dim=1)
        accuracies.append((predictions == y).float().mean().item())
    return accuracies


# ---------------------------------------------------------------------------
# 4. 纯文本表格输出（正确处理中文全角字符宽度）
# ---------------------------------------------------------------------------
def display_width(text: str) -> int:
    """终端显示宽度：中日韩全角字符按 2 列计算。"""
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def pad_cell(text: str, width: int, align: str = "left") -> str:
    padding = " " * max(0, width - display_width(text))
    if align == "right":
        return padding + text
    return text + padding


def center_display(text: str, width: int) -> str:
    padding = max(0, width - display_width(text))
    left = padding // 2
    return " " * left + text + " " * (padding - left)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    alignments = ["left"] + ["right"] * (len(headers) - 1)
    widths = []
    for col, header in enumerate(headers):
        content_width = max(display_width(str(row[col])) for row in rows)
        widths.append(max(display_width(header), content_width))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(cells: Sequence[str]) -> str:
        padded = [
            pad_cell(str(cell), widths[i], alignments[i]) for i, cell in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    print(border)
    print(format_row(headers))
    print(border)
    for row in rows:
        print(format_row(row))
    print(border)


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "请求了 --device cuda，但当前 PyTorch 无法使用 CUDA。\n"
            "请检查 NVIDIA 驱动，并安装与本机 GPU 匹配的 CUDA 版 PyTorch。"
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多任务学习三种架构对比实验")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备：auto 自动选择（有 CUDA 时用 GPU），默认 auto",
    )
    return parser.parse_args()


def main(device: torch.device | None = None) -> None:
    device = device or torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用，无法在 GPU 上运行。")
        torch.cuda.init()
        props = torch.cuda.get_device_properties(0)
        device_label = (
            f"CUDA · {props.name} · cc{props.major}.{props.minor} · "
            f"{props.total_memory / 1024**3:.1f} GiB"
        )
    else:
        device_label = "CPU（单线程）"

    set_seed(SEED)
    train_data, test_data = build_datasets()

    print("=" * 92)
    print(center_display("多任务学习：独立基线 / 共享主干 + 任务头 / A-Net Top-1 硬路由专家", 92))
    print("=" * 92)
    print("任务定义：")
    print("  任务1 | 正弦边界：x2 > sin(2*x1)")
    print("  任务2 | 圆形边界：r < 1.0 为类别 0，r > 1.0 为类别 1（平衡极坐标采样）")
    print("  任务3 | XOR     ：x1*x2 > 0")
    print(
        f"数据规模：每任务 train={N_TRAIN_PER_TASK:,} / test={N_TEST_PER_TASK:,}；"
        f"总计 {NUM_TASKS * (N_TRAIN_PER_TASK + N_TEST_PER_TASK):,} 个样本"
    )
    print(
        f"训练设置：Adam(lr={LR})，epochs={EPOCHS}，batch_size={BATCH_SIZE}，"
        f"seed={SEED}，device={device_label}"
    )
    print()

    experiments: list[tuple[str, nn.Module]] = []

    # 每个模型训练前重置随机种子，保证初始化条件一致。
    set_seed(SEED)
    independent = IndependentBaseline()
    experiments.append(("独立基线 (3 x MLP)", independent))

    set_seed(SEED)
    shared = SharedTrunkWithHeads()
    experiments.append(("共享主干 + 3 个线性头", shared))

    set_seed(SEED)
    anet = ANetHardRouting()
    experiments.append(("A-Net 极简版 (Top-1 硬路由)", anet))

    rows: list[list[str]] = []
    for name, model in experiments:
        model.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        train_model(model, train_data, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        accuracies = evaluate_per_task(model, test_data, device=device)
        mean_accuracy = sum(accuracies) / len(accuracies)
        parameter_count = count_trainable_parameters(model)

        print(
            f"[完成] {pad_cell(name, 32)} | 参数量 {parameter_count:>8,d} | "
            f"平均准确率 {mean_accuracy * 100:6.2f}% | 用时 {elapsed:5.1f}s"
        )
        rows.append(
            [
                name,
                f"{parameter_count:,d}",
                f"{accuracies[0] * 100:.2f}%",
                f"{accuracies[1] * 100:.2f}%",
                f"{accuracies[2] * 100:.2f}%",
                f"{mean_accuracy * 100:.2f}%",
            ]
        )

    print()
    print("测试集结果对比")
    print_table(
        ["方案名称", "总参数量", "任务1准确率", "任务2准确率", "任务3准确率", "平均准确率"],
        rows,
    )

    # 参数共享带来的直接收益，便于解读结果。
    independent_params = count_trainable_parameters(independent)
    shared_params = count_trainable_parameters(shared)
    anet_params = count_trainable_parameters(anet)
    print()
    anet_active_params = count_trainable_parameters(anet.l1) + count_trainable_parameters(
        anet.experts[0]
    )
    print("参数量相对独立基线：")
    print(f"  共享主干 + 3 个线性头 : {shared_params / independent_params * 100:6.1f}%")
    print(f"  A-Net 极简版          : {anet_params / independent_params * 100:6.1f}%")
    print(
        f"  A-Net 单样本激活参数（L1 + 1 个专家）: {anet_active_params:,d} / {anet_params:,d}"
    )
    print("\n说明：以上准确率均为单次固定随机种子实验，仅用于结构对比。")


if __name__ == "__main__":
    args = parse_args()
    main(resolve_device(args.device))
