"""Measure true process-level cold start for saved scaling checkpoints.

Run through benchmark_cold_starts.py, which spawns one fresh Python process per
checkpoint. The cold forward is the first model forward after loading the
checkpoint; the CUDA context has been initialized but no model kernels have run
yet. Warm latency is measured after explicit warmup.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

_SCRIPT_START = time.perf_counter()

import torch

from .models import build_model

_IMPORT_END = time.perf_counter()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--throughput-batch", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    cuda_start = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    cuda_init_s = time.perf_counter() - cuda_start

    load_start = time.perf_counter()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        payload["model_kind"],
        payload["hidden_dim"],
        payload["expert_hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start
    ready_s = time.perf_counter() - _SCRIPT_START

    x1 = torch.randn(1, 2, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        model(x1, 0)
    torch.cuda.synchronize()
    cold_forward_us = (time.perf_counter_ns() - t0) / 1000.0

    for _ in range(50):
        model(x1, 0)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
    with torch.inference_mode():
        for i in range(args.repeats):
            starts[i].record()
            model(x1, 0)
            ends[i].record()
        torch.cuda.synchronize()
    warm = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    warm_p50 = statistics.median(warm)
    warm_p95 = warm[min(len(warm) - 1, int(len(warm) * 0.95))]

    xb = torch.randn(args.throughput_batch, 2, device=device)
    for _ in range(10):
        model(xb, 0)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(30):
            model(xb, 0)
    torch.cuda.synchronize()
    throughput = args.throughput_batch * 30 / (time.perf_counter() - wall_start)

    result = {
        "checkpoint": str(args.checkpoint),
        "scale": payload["scale"],
        "model": payload["model_kind"],
        "seed": payload["seed"],
        "device": torch.cuda.get_device_name(0),
        "import_end_s": _IMPORT_END - _SCRIPT_START,
        "cuda_init_s": cuda_init_s,
        "model_load_s": model_load_s,
        "ready_s": ready_s,
        "cold_forward_us": cold_forward_us,
        "warm_b1_p50_us": warm_p50,
        "warm_b1_p95_us": warm_p95,
        "throughput_b1024_samples_s": throughput,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
