"""Latency/throughput/memory measurement with raw samples retained."""

from __future__ import annotations

import gc
import statistics
import time

import numpy as np
import torch

from .models import ANetModel
from .tasks import SuiteData


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def summarise(samples_us: list[float]) -> dict:
    values = sorted(float(v) for v in samples_us)
    return {
        "n": len(values),
        "p50_us": statistics.median(values) if values else None,
        "p95_us": _percentile(values, 0.95),
        "mean_us": statistics.mean(values) if values else None,
        "std_us": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_us": min(values) if values else None,
        "max_us": max(values) if values else None,
    }


def _run_timed(fn, device: torch.device, repeats: int) -> tuple[list[float], float]:
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        with torch.inference_mode():
            for i in range(repeats):
                starts[i].record()
                fn()
                ends[i].record()
        torch.cuda.synchronize(device)
        samples = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
        wall = sum(samples) / 1e6
    else:
        samples = []
        with torch.inference_mode():
            for _ in range(repeats):
                start = time.perf_counter_ns()
                fn()
                samples.append((time.perf_counter_ns() - start) / 1000.0)
        wall = sum(samples) / 1e6
    return samples, wall


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def benchmark_inference(
    model: torch.nn.Module,
    suite: SuiteData,
    *,
    device: torch.device,
    batch_sizes: list[int],
    repeats: int,
    modes: list[str],
    warmup: int = 20,
    task_id: int = 0,
    reset_cache_each_repeat: bool = False,
) -> dict:
    model.eval()
    result = {"batches": {}, "modes": modes, "repeats": repeats, "device": str(device)}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        result["peak_allocated_bytes"] = None
        result["peak_reserved_bytes"] = None

    for mode in modes:
        result["batches"][mode] = {}
        for batch_size in batch_sizes:
            if mode == "mixed":
                x = suite.x_test[:batch_size].repeat(len(suite.specs), 1).to(device)
                task_ids = torch.cat(
                    [torch.full((batch_size,), t, dtype=torch.long, device=device) for t in range(len(suite.specs))]
                )
                total_samples = x.size(0)
                call = lambda: model.forward_mixed(x, task_ids)
            else:
                x = suite.x_test[:batch_size].to(device)
                total_samples = batch_size
                if isinstance(model, ANetModel):
                    if mode == "single":
                        active = [task_id]
                    elif mode == "pair":
                        active = [task_id, (task_id + 1) % len(suite.specs)]
                    elif mode == "dense":
                        active = list(range(len(suite.specs)))
                    else:
                        raise KeyError(f"unknown mode {mode}")
                    call = lambda active=active: model(x, task_id, active_ids=active)
                else:
                    call = lambda: model(x, task_id)

            with torch.inference_mode():
                for _ in range(warmup):
                    call()
            _sync(device)
            if isinstance(model, ANetModel):
                model.cache.clear()
                model.cache.stats = {k: (0.0 if isinstance(v, float) else 0) for k, v in model.cache.stats.items()}
            if reset_cache_each_repeat and isinstance(model, ANetModel):
                def call_with_reset(call=call):
                    model.cache.clear()
                    call()
                timed_call = call_with_reset
            else:
                timed_call = call
            samples, wall = _run_timed(timed_call, device, repeats)
            _sync(device)
            summary = summarise(samples)
            result["batches"][mode][str(batch_size)] = {
                **summary,
                "total_samples": total_samples,
                "throughput_samples_s": total_samples * repeats / max(wall, 1e-12),
                "raw_us": samples,
            }
            if isinstance(model, ANetModel):
                result["batches"][mode][str(batch_size)]["cache"] = model.cache.summary()

    if device.type == "cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    result["peak_rss_bytes"] = _rss_bytes()
    gc.collect()
    return result
