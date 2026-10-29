"""
对 multitask_anet_demo.py 中的三种模型做 CPU 推理延迟基准测试。

测量的是纯 PyTorch eager forward（包含 Python 路由/索引开销），
不包含数据加载、预处理、反向传播和模型保存。

运行：
    python benchmark_latency.py
"""

from __future__ import annotations

import gc
import math
import os
import platform
import statistics
import time

# 必须在 import torch 之前限制 CPU 线程，避免 MKL/OpenMP 额外并行
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

from multitask_anet_demo import (
    ANetHardRouting,
    IndependentBaseline,
    SharedTrunkWithHeads,
    count_trainable_parameters,
    print_table,
)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SEED = 42
TASK_ID = 0
BATCH_SIZES = (1, 64, 256)


def build_models() -> list[tuple[str, torch.nn.Module, int]]:
    """返回 (方案名, 模型, 单样本实际激活参数量)。"""
    torch.manual_seed(SEED)
    independent = IndependentBaseline()
    torch.manual_seed(SEED)
    shared = SharedTrunkWithHeads()
    torch.manual_seed(SEED)
    anet = ANetHardRouting()

    independent_active = count_trainable_parameters(independent.mlps[TASK_ID])
    shared_active = count_trainable_parameters(shared)
    anet_active = count_trainable_parameters(anet.l1) + count_trainable_parameters(
        anet.experts[TASK_ID]
    )

    return [
        ("独立基线 (3 x MLP)", independent, independent_active),
        ("共享主干 + 3 个线性头", shared, shared_active),
        ("A-Net 极简版 (Top-1 硬路由)", anet, anet_active),
    ]


def settings_for(batch_size: int) -> tuple[int, int, int]:
    """返回 (warmup 次数, 每轮测量次数, 轮数)。"""
    if batch_size == 1:
        return 500, 3000, 7
    if batch_size == 64:
        return 200, 1000, 5
    return 100, 500, 5


def percentile(sorted_values: list[int], quantile: float) -> int:
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


def benchmark_batch(
    models: list[tuple[str, torch.nn.Module, int]],
    batch_size: int,
) -> list[tuple[str, int, int, float, float, float]]:
    """对一个 batch size 测量所有模型，轮间轮换顺序以降低频率漂移偏差。"""
    warmup, iterations, rounds = settings_for(batch_size)
    x = torch.randn(batch_size, 2)
    timings: dict[str, list[int]] = {name: [] for name, _, _ in models}

    for _, model, _ in models:
        model.eval()

    with torch.inference_mode():
        for _, model, _ in models:
            for _ in range(warmup):
                model(x, TASK_ID)

        for round_index in range(rounds):
            offset = round_index % len(models)
            order = models[offset:] + models[:offset]
            for name, model, _ in order:
                bucket = timings[name]
                for _ in range(iterations):
                    start = time.perf_counter_ns()
                    model(x, TASK_ID)
                    bucket.append(time.perf_counter_ns() - start)

    results: list[tuple[str, int, int, float, float, float]] = []
    for name, model, active_params in models:
        samples = timings[name]
        samples.sort()
        total_params = count_trainable_parameters(model)
        p50_us = statistics.median(samples) / 1_000
        p95_us = percentile(samples, 0.95) / 1_000
        throughput = batch_size * len(samples) / (sum(samples) / 1e9)
        results.append((name, total_params, active_params, p50_us, p95_us, throughput))
    return results


def main() -> None:
    print("=" * 96)
    print("CPU 推理延迟基准：纯 forward，单线程，无数据加载".center(96))
    print("=" * 96)
    print(f"Python      : {platform.python_version()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CPU 线程数  : {torch.get_num_threads()}")
    print(f"批量大小    : {BATCH_SIZES}")
    print("计时方式    : warmup 后按轮测量，模型顺序轮换；汇总样本取 P50/P95，吞吐=样本数/总耗时")
    print("模型状态    : 随机初始化（延迟只由网络结构和计算图决定，与训练权重无关）")
    print()

    models = build_models()
    results: dict[int, list[tuple[str, int, int, float, float, float]]] = {}

    gc.disable()
    try:
        for batch_size in BATCH_SIZES:
            results[batch_size] = benchmark_batch(models, batch_size)
    finally:
        gc.enable()

    # 表 1：单样本在线推理
    rows = []
    for name, total_params, active_params, p50, p95, throughput in results[1]:
        rows.append(
            [
                name,
                f"{total_params:,d}",
                f"{active_params:,d}",
                f"{p50:.1f}",
                f"{p95:.1f}",
                f"{throughput:,.0f}",
            ]
        )
    print("表 1  单样本延迟（batch_size=1）")
    print_table(
        ["方案", "总参数量", "激活参数量", "P50 延迟(μs)", "P95 延迟(μs)", "吞吐(样本/s)"],
        rows,
    )

    # 表 2：批量吞吐
    rows = []
    for batch_size in (64, 256):
        for name, _total_params, active_params, p50, p95, throughput in results[batch_size]:
            rows.append(
                [
                    name,
                    str(batch_size),
                    f"{active_params:,d}",
                    f"{p50:.1f}",
                    f"{p95:.1f}",
                    f"{throughput:,.0f}",
                ]
            )
    print()
    print("表 2  批量推理延迟与吞吐")
    print_table(
        ["方案", "批量", "激活参数量", "P50 延迟(μs)", "P95 延迟(μs)", "吞吐(样本/s)"],
        rows,
    )

    print()
    print("说明：以上数字是当前 Windows 主机、当前 PyTorch 版本、单核 CPU 下的实测值；")
    print("      换 CPU / 打开多线程 / 使用 GPU 或 TorchScript 后，绝对值和排名都可能变化。")
    print("      总参数量是内存中保存的全部参数；激活参数量是每个样本实际参与计算的参数。")
    print("      A-Net 总参数量包含全部 3 个专家，但每次只执行 Task ID 对应的 1 个专家。")


if __name__ == "__main__":
    main()
