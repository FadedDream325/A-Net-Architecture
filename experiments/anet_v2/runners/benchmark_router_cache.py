"""Focused benchmark for the router/cache refactor.

Compares the pure eval path (cache off) with the cached eval path on
VRAM-resident inputs.  Refuses to start unless dedicated VRAM is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.anet_v2.core.benchmark import benchmark_inference
from experiments.anet_v2.core.device_guard import enforce_dedicated_vram, snapshot
from experiments.anet_v2.core.models import ANetModel, build_model
from experiments.anet_v2.core.tasks import build_suite
from experiments.anet_v2.core.train import set_seed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--widths", nargs="+", type=int, default=[64, 256, 1024, 2048])
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--correlation", default="low")
    p.add_argument("--models", nargs="+", default=["independent", "anet_basic", "anet_full"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 1024])
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--min-free-vram-gb", type=float, default=2.0)
    p.add_argument("--max-vram-fraction", type=float, default=0.80)
    p.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "refactor_benchmark")
    args = p.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    guard = enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
    print("VRAM guard:", guard, flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_bench.jsonl"
    rows = []

    for seed in args.seeds:
        suite = build_suite(seed, args.correlation, n_train=64, n_val=32, n_test=32).to(device)
        for width in args.widths:
            for kind in args.models:
                cache_flags = [False, True] if kind == "anet_full" else [False]
                for cache_flag in cache_flags:
                    enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
                    set_seed(seed)
                    model = build_model(
                        kind,
                        num_tasks=len(suite.specs),
                        in_dim=suite.meta["in_dim"],
                        hidden_dim=width,
                        out_dims=[s.out_dim for s in suite.specs],
                        active_mode="sparse",
                    ).to(device)
                    model.eval()
                    out = benchmark_inference(
                        model,
                        suite,
                        device=device,
                        batch_sizes=args.batch_sizes,
                        repeats=args.repeats,
                        modes=["single"],
                        warmup=30,
                        reset_cache_each_repeat=False,
                        inference_cache=cache_flag,
                    )
                    for batch, data in out["batches"]["single"].items():
                        rec = {
                            "seed": seed,
                            "width": width,
                            "model": kind,
                            "inference_cache": cache_flag,
                            "batch": int(batch),
                            "p50_us": data["p50_us"],
                            "p95_us": data["p95_us"],
                            "throughput_samples_s": data["throughput_samples_s"],
                            "input_device": data["input_device"],
                            "host_to_device_bytes": out["host_to_device_bytes"],
                            "peak_allocated_bytes": out["peak_allocated_bytes"],
                            "cache": data.get("cache"),
                        }
                        rows.append(rec)
                        print(
                            f"{seed} h{width} {kind} cache={cache_flag} b{batch} "
                            f"p50={data['p50_us']:.1f}us thr={data['throughput_samples_s']:.0f}",
                            flush=True,
                        )
                    del model
                    torch.cuda.empty_cache()

    with open(raw_path, "w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    groups = {}
    for rec in rows:
        groups.setdefault((rec["width"], rec["model"], rec["inference_cache"], rec["batch"]), []).append(rec)
    summary = []
    for key, recs in sorted(groups.items()):
        width, model, cache_flag, batch = key
        summary.append({
            "width": width,
            "model": model,
            "inference_cache": cache_flag,
            "batch": batch,
            "n": len(recs),
            "p50_us_mean": statistics.mean(r["p50_us"] for r in recs),
            "p50_us_std": statistics.stdev(r["p50_us"] for r in recs) if len(recs) > 1 else 0.0,
            "p95_us_mean": statistics.mean(r["p95_us"] for r in recs),
            "throughput_mean": statistics.mean(r["throughput_samples_s"] for r in recs),
            "host_to_device_bytes_max": max(r["host_to_device_bytes"] for r in recs),
            "peak_allocated_gb_max": max(r["peak_allocated_bytes"] for r in recs if r["peak_allocated_bytes"]) / 1024**3,
        })
    with open(args.out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (args.out_dir / "summary.json").write_text(json.dumps({"guard": guard, "rows": summary}, indent=2), encoding="utf-8")
    print("wrote", args.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
