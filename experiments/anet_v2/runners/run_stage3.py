"""Stage 3: result-cache correctness and performance under input repetition."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.anet_v2.core.cache_benchmark import benchmark_cache_stream
from experiments.anet_v2.core.device_guard import enforce_dedicated_vram, snapshot as vram_snapshot
from experiments.anet_v2.core.models import build_model
from experiments.anet_v2.core.tasks import build_suite
from experiments.anet_v2.core.train import set_seed
from experiments.anet_v2.runners.run_stage1 import environment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    p = argparse.ArgumentParser(description="A-Net v2 Stage 3 cache benchmark")
    p.add_argument("--widths", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--models", nargs="+", default=["anet_reuse", "anet_full"])
    p.add_argument("--active-mode", choices=("pair", "dense"), default="pair")
    p.add_argument("--repetition-rates", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 256])
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--correlation", default="medium")
    p.add_argument("--cache-key-mode", choices=("identity", "content"), default="identity")
    p.add_argument("--gradient-isolation", choices=("none", "detach_aux", "detach_shared", "detach_both"), default="detach_both")
    p.add_argument("--min-free-vram-gb", type=float, default=2.0)
    p.add_argument("--max-vram-fraction", type=float, default=0.80)
    p.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage3")
    args = p.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    guard = enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
    print("VRAM guard:", guard, flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_bench.jsonl"
    env = environment()
    config = {
        "widths": args.widths,
        "seeds": args.seeds,
        "models": args.models,
        "active_mode": args.active_mode,
        "repetition_rates": args.repetition_rates,
        "batch_sizes": args.batch_sizes,
        "requests": args.requests,
        "correlation": args.correlation,
        "cache_key_mode": args.cache_key_mode,
        "gradient_isolation": args.gradient_isolation,
        "min_free_vram_gb": args.min_free_vram_gb,
        "max_vram_fraction": args.max_vram_fraction,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (args.out_dir / "config.json").write_text(json.dumps({"config": config, "config_hash": config_hash}, indent=2), encoding="utf-8")
    (args.out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    suite = build_suite(2026, args.correlation, n_train=64, n_val=32, n_test=256).to(device)
    print(f"Stage 3 mode={args.active_mode} key={args.cache_key_mode} rates={args.repetition_rates}", flush=True)

    for seed in args.seeds:
        suite = build_suite(seed, args.correlation, n_train=64, n_val=32, n_test=max(args.batch_sizes) * 4).to(device)
        for width in args.widths:
            for kind in args.models:
                enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
                set_seed(seed)
                model = build_model(
                    kind,
                    num_tasks=len(suite.specs),
                    in_dim=suite.meta["in_dim"],
                    hidden_dim=width,
                    out_dims=[s.out_dim for s in suite.specs],
                    active_mode=args.active_mode,
                    cache_key_mode=args.cache_key_mode,
                    gradient_isolation=args.gradient_isolation,
                ).to(device)
                model.eval()
                if not model.cache.enabled:
                    raise RuntimeError(f"{kind} has no cache")

                # Output equivalence: cache off vs cache on.
                model.set_inference_cache(False)
                x_audit = suite.x_test[: min(64, suite.x_test.size(0))]
                with torch.inference_mode():
                    y_off = model(x_audit, 0)
                model.set_inference_cache(True)
                model.cache.clear()
                with torch.inference_mode():
                    y_on = model(x_audit, 0)
                eq = float((y_off - y_on).abs().max())

                for batch in args.batch_sizes:
                    for rate in args.repetition_rates:
                        for cache_enabled in (False, True):
                            enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
                            model.set_inference_cache(cache_enabled)
                            model.cache.clear()
                            torch.cuda.reset_peak_memory_stats()
                            try:
                                result = benchmark_cache_stream(
                                    model,
                                    suite,
                                    device=device,
                                    batch_size=batch,
                                    requests=args.requests,
                                    repetition_rate=rate,
                                    cache_enabled=cache_enabled,
                                    task_id=0,
                                )
                                rec = {
                                    "run_id": f"h{width:04d}__{args.active_mode}__{kind}__b{batch}__r{rate}__cache{int(cache_enabled)}__s{seed}",
                                    "status": "ok",
                                    "seed": seed,
                                    "width": width,
                                    "model": kind,
                                    "active_mode": args.active_mode,
                                    "batch": batch,
                                    "rate": rate,
                                    "cache_enabled": cache_enabled,
                                    "output_equivalence_max_abs": eq,
                                    "vram_before": vram_snapshot(),
                                    "git_commit": env["git_commit"],
                                    "git_status": env["git_status"],
                                    "code_hash": env["code_hash"],
                                    **result,
                                }
                                rec["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                                print(
                                    f"{seed} h{width} {kind} b{batch} r{rate} cache={cache_enabled} "
                                    f"p50={result['p50_us']:.1f} hit={result['cache']['hit_rate']:.2f} "
                                    f"exec={result['actual_executions']}/{result['theoretical_executions']}",
                                    flush=True,
                                )
                            except Exception as exc:  # noqa: BLE001
                                rec = {
                                    "run_id": f"h{width:04d}__{args.active_mode}__{kind}__b{batch}__r{rate}__cache{int(cache_enabled)}__s{seed}",
                                    "status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(),
                                }
                                print(f"FAILED {rec['error']}", flush=True)
                            with open(raw_path, "a", encoding="utf-8") as fh:
                                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                fh.flush()
                del model
                gc.collect()
                torch.cuda.empty_cache()
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
