"""Latency/throughput/memory measurement with raw samples retained.

Timed calls run only on tensors that are already resident on the target
device.  Any one-time host-to-device transfer happens before warmup and is
reported separately; it is never inside the timed loop.
"""

from __future__ import annotations

import gc
import statistics
import time

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
    """Return per-call CUDA-event times (or CPU perf_counter) and true wall time.

    Throughput uses wall time, not the sum of event durations: the latter
    excludes launch gaps and Python overhead and would overstate throughput.
    """
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        _sync(device)
        wall_start = time.perf_counter()
        with torch.inference_mode():
            for i in range(repeats):
                starts[i].record()
                fn()
                ends[i].record()
        _sync(device)
        wall = time.perf_counter() - wall_start
        samples = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    else:
        samples = []
        wall_start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                start = time.perf_counter_ns()
                fn()
                samples.append((time.perf_counter_ns() - start) / 1000.0)
        wall = time.perf_counter() - wall_start
    return samples, wall


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _to_device_once(t: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    """Move once, report bytes; called before warmup, never in the timed loop."""
    same_type = t.device.type == device.type
    same_index = device.index is None or t.device.index == device.index
    if same_type and same_index:
        return t, 0
    moved = t.to(device)
    return moved, int(t.numel() * t.element_size())


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
    inference_cache: bool = False,
) -> dict:
    model.eval()
    if isinstance(model, ANetModel):
        model.set_inference_cache(inference_cache)
    model.to(device)
    param_device = next(model.parameters()).device
    if device.type == "cuda":
        assert param_device.type == "cuda", f"model is on {param_device}, not CUDA"
    result = {"batches": {}, "modes": modes, "repeats": repeats, "device": str(device),
              "model_device": str(param_device), "host_to_device_bytes": 0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        result["peak_allocated_bytes"] = None
        result["peak_reserved_bytes"] = None

    for mode in modes:
        result["batches"][mode] = {}
        for batch_size in batch_sizes:
            if mode == "mixed":
                x = suite.x_test[:batch_size].repeat(len(suite.specs), 1)
                x, moved = _to_device_once(x, device)
                result["host_to_device_bytes"] += moved
                task_ids = torch.cat(
                    [torch.full((batch_size,), t, dtype=torch.long, device=device) for t in range(len(suite.specs))]
                )
                total_samples = x.size(0)
                call = lambda: model.forward_mixed(x, task_ids)
            else:
                x = suite.x_test[:batch_size]
                x, moved = _to_device_once(x, device)
                result["host_to_device_bytes"] += moved
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

            if device.type == "cuda":
                assert x.is_cuda, "timed input is not on CUDA"
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
                "input_device": str(x.device),
                "throughput_samples_s": total_samples * repeats / max(wall, 1e-12),
                "wall_seconds": wall,
                "raw_us": samples,
            }
            if isinstance(model, ANetModel):
                result["batches"][mode][str(batch_size)]["cache"] = model.cache.summary()

    if device.type == "cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    result["rss_bytes_at_end"] = _rss_bytes()
    gc.collect()
    return result
