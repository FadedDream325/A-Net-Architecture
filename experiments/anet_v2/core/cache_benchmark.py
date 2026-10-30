"""Request-stream benchmark for the result cache.

The pool is allocated on the GPU before timing.  Repetition is created by
reusing the same pool index (same data_ptr/shape/stride), so the identity key
can hit without any host-to-device copy in the timed loop.
"""

from __future__ import annotations

import statistics
import time

import torch

from .models import ANetModel
from .tasks import SuiteData


def repetition_indices(requests: int, rate: float) -> list[int]:
    if requests <= 0:
        raise ValueError("requests must be positive")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0,1]")
    indices: list[int] = []
    for i in range(requests):
        if i == 0:
            indices.append(0)
        elif rate == 0.0:
            indices.append(i)
        elif rate == 0.25:
            indices.append(indices[-1] if i % 4 == 3 else i)
        elif rate == 0.5:
            indices.append(indices[-1] if i % 2 == 1 else i)
        elif rate == 0.75:
            indices.append(indices[-1] if i % 4 != 0 else i)
        else:
            # General deterministic pattern: repeat the previous input for the
            # first round(rate*requests) slots of each block.
            raise ValueError("supported rates are 0, 0.25, 0.5, 0.75")
    return indices


def _pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def benchmark_cache_stream(
    model: ANetModel,
    suite: SuiteData,
    *,
    device: torch.device,
    batch_size: int,
    requests: int,
    repetition_rate: float,
    cache_enabled: bool,
    task_id: int = 0,
) -> dict:
    model.eval()
    model.set_inference_cache(cache_enabled)
    base = suite.x_test
    needed = requests * batch_size
    repeats = (needed + base.size(0) - 1) // base.size(0)
    pool = base.repeat(repeats, 1)[:needed]
    pool = pool.view(requests, batch_size, base.size(1)).to(device)
    indices = repetition_indices(requests, repetition_rate)

    with torch.inference_mode():
        for _ in range(5):
            model(pool[0], task_id)
    torch.cuda.synchronize()
    model.cache.clear()
    model.cache.stats = {k: (0.0 if isinstance(v, float) else 0) for k, v in model.cache.stats.items()}

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(requests)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(requests)]
    actual_exec = 0
    theoretical_exec = 0
    theoretical_flops = 0.0
    actual_flops = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for i, idx in enumerate(indices):
            x = pool[idx]
            starts[i].record()
            model(x, task_id)
            ends[i].record()
            acc = model.last_accounting
            actual_exec += int(acc.get("actual_executions_per_sample", 0))
            theoretical_exec += int(acc.get("theoretical_executions_per_sample", 0))
            theoretical_flops += float(acc.get("theoretical_flops_per_sample", 0))
            actual_flops += float(acc.get("actual_flops_per_sample", 0))
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    samples = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    samples_sorted = sorted(samples)

    cache_summary = model.cache.summary()
    model.mark_updated()
    invalidation_ok = model.cache.size == 0

    return {
        "batch": batch_size,
        "requests": requests,
        "repetition_rate": repetition_rate,
        "cache_enabled": cache_enabled,
        "input_device": str(pool.device),
        "host_to_device_bytes": 0,
        "p50_us": statistics.median(samples),
        "p95_us": _pct(samples_sorted, 0.95),
        "mean_us": statistics.mean(samples),
        "throughput_samples_s": requests * batch_size / max(wall, 1e-12),
        "wall_seconds": wall,
        "actual_executions": actual_exec,
        "theoretical_executions": theoretical_exec,
        "actual_flops": actual_flops,
        "theoretical_flops": theoretical_flops,
        "cache": cache_summary,
        "cache_invalidation_ok": invalidation_ok,
        "raw_us": samples,
    }
