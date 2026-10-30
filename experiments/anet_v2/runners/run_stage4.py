"""Stage 4: factorial scaling across sharing, reuse, correlation, width, workflows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.anet_v2.core.benchmark import benchmark_inference
from experiments.anet_v2.core.cache_benchmark import benchmark_cache_stream
from experiments.anet_v2.core.device_guard import enforce_dedicated_vram, snapshot as vram_snapshot
from experiments.anet_v2.core.models import ANetModel, MODEL_LABELS, build_model
from experiments.anet_v2.core.tasks import CORRELATIONS, build_suite
from experiments.anet_v2.core.train import TrainConfig, evaluate_model, set_seed, train_model
from experiments.anet_v2.runners.run_stage1 import PeakRSS, environment, model_cost


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(width: int, workflows: int, corr: str, kind: str, seed: int) -> str:
    return f"h{width:04d}__w{workflows}__{corr}__{kind}__s{seed}"


def main() -> int:
    p = argparse.ArgumentParser(description="A-Net v2 Stage 4")
    p.add_argument("--widths", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--workflows", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--models", nargs="+", default=["anet_basic", "anet_share", "anet_reuse", "anet_full"])
    p.add_argument("--active-mode", choices=("pair", "dense"), default="pair")
    p.add_argument("--correlations", nargs="+", default=["low", "medium", "high"], choices=list(CORRELATIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--val-size", type=int, default=1500)
    p.add_argument("--test-size", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-isolation", choices=("none", "detach_aux", "detach_shared", "detach_both"), default="detach_both")
    p.add_argument("--min-free-vram-gb", type=float, default=2.0)
    p.add_argument("--max-vram-fraction", type=float, default=0.80)
    p.add_argument("--bench-batches", nargs="+", type=int, default=[1, 256])
    p.add_argument("--bench-repeats", type=int, default=100)
    p.add_argument("--cache-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache-eval-width", type=int, default=2048)
    p.add_argument("--cache-eval-workflows", type=int, default=3)
    p.add_argument("--cache-eval-correlation", default="medium")
    p.add_argument("--cache-rates", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    p.add_argument("--cache-batches", nargs="+", type=int, default=[1, 256])
    p.add_argument("--cache-requests", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage4")
    args = p.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    guard = enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
    print("VRAM guard:", guard, flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_runs.jsonl"
    config = {
        "widths": args.widths,
        "workflows": args.workflows,
        "models": args.models,
        "active_mode": args.active_mode,
        "correlations": args.correlations,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_isolation": args.gradient_isolation,
        "min_free_vram_gb": args.min_free_vram_gb,
        "max_vram_fraction": args.max_vram_fraction,
        "bench_batches": args.bench_batches,
        "bench_repeats": args.bench_repeats,
        "cache_eval": args.cache_eval,
        "cache_eval_width": args.cache_eval_width,
        "cache_eval_workflows": args.cache_eval_workflows,
        "cache_eval_correlation": args.cache_eval_correlation,
        "cache_rates": args.cache_rates,
        "cache_batches": args.cache_batches,
        "cache_requests": args.cache_requests,
        "max_steps": args.max_steps,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (out_dir / "config.json").write_text(json.dumps({"config": config, "config_hash": config_hash}, indent=2), encoding="utf-8")
    env = environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    done = set()
    if args.resume and raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add(rec.get("run_id"))
    print(f"Stage 4 widths={args.widths} W={args.workflows} modes={len(args.correlations)} seeds={args.seeds}", flush=True)

    data_cache = {}
    warm = build_suite(args.seeds[0], args.correlations[0], n_train=args.train_size, n_val=args.val_size, n_test=args.test_size).to(device)
    data_cache[(args.seeds[0], args.correlations[0])] = warm
    start = time.perf_counter()
    a = torch.randn(256, 128, device=device)
    b = torch.randn(128, 128, device=device, requires_grad=True)
    (a @ b).sum().backward()
    torch.cuda.synchronize()
    print(f"warmup: {time.perf_counter()-start:.2f}s", flush=True)
    runs = 0

    for corr in args.correlations:
        for seed in args.seeds:
            key = (seed, corr)
            if key not in data_cache:
                data_cache[key] = build_suite(seed, corr, n_train=args.train_size, n_val=args.val_size, n_test=args.test_size).to(device)
            suite = data_cache[key]
            for width in args.widths:
                for workflows in args.workflows:
                    for kind in args.models:
                        rid = run_id(width, workflows, corr, kind, seed)
                        if rid in done:
                            continue
                        if args.max_runs is not None and runs >= args.max_runs:
                            break
                        runs += 1
                        enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
                        print(f"[run] {rid}", flush=True)
                        record = {
                            "run_id": rid,
                            "status": "failed",
                            "started_utc": utc_now(),
                            "width": width,
                            "workflows": workflows,
                            "correlation": corr,
                            "seed": seed,
                            "model": kind,
                            "model_label": MODEL_LABELS[kind],
                            "active_mode": args.active_mode,
                            "gradient_isolation": args.gradient_isolation,
                            "data": suite.meta,
                            "config": config,
                            "config_hash": config_hash,
                            "git_commit": env["git_commit"],
                            "git_status": env["git_status"],
                            "code_hash": env["code_hash"],
                            "vram_before": vram_snapshot(),
                        }
                        model = None
                        try:
                            set_seed(seed)
                            model = build_model(
                                kind,
                                num_tasks=len(suite.specs),
                                num_workflows=workflows,
                                in_dim=suite.meta["in_dim"],
                                hidden_dim=width,
                                out_dims=[s.out_dim for s in suite.specs],
                                active_mode=args.active_mode,
                                gradient_isolation=args.gradient_isolation,
                            ).to(device)
                            record["cost"] = model_cost(model, kind, suite, suite.meta["in_dim"], width, [s.out_dim for s in suite.specs])
                            record["identity"] = record["cost"].get("identity")

                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            with PeakRSS() as rss:
                                train_info = train_model(
                                    model, suite,
                                    TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                                weight_decay=args.weight_decay, max_steps=args.max_steps),
                                    device=device,
                                )
                            torch.cuda.synchronize()
                            train_info["peak_rss_bytes"] = rss.peak
                            train_info["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                            record["train"] = train_info
                            record["val"] = evaluate_model(model, suite, split="val").__dict__
                            record["test"] = evaluate_model(model, suite, split="test").__dict__

                            modes = ["single"]
                            if isinstance(model, ANetModel):
                                modes = ["single", args.active_mode]
                            record["inference"] = benchmark_inference(
                                model, suite, device=device, batch_sizes=args.bench_batches,
                                repeats=args.bench_repeats, modes=modes,
                                reset_cache_each_repeat=True, inference_cache=False,
                            )

                            if (args.cache_eval and isinstance(model, ANetModel)
                                    and kind in {"anet_reuse", "anet_full"}
                                    and width == args.cache_eval_width
                                    and workflows == args.cache_eval_workflows
                                    and corr == args.cache_eval_correlation):
                                streams = []
                                for batch in args.cache_batches:
                                    for rate in args.cache_rates:
                                        model.set_inference_cache(True)
                                        model.cache.clear()
                                        streams.append({
                                            "batch": batch,
                                            "rate": rate,
                                            **benchmark_cache_stream(
                                                model, suite, device=device, batch_size=batch,
                                                requests=args.cache_requests, repetition_rate=rate,
                                                cache_enabled=True,
                                            ),
                                        })
                                record["cache_streams"] = streams

                            record["status"] = "ok"
                            record["finished_utc"] = utc_now()
                            acc = record["test"]["mean"].get("accuracy")
                            print(f"  ok acc={None if acc is None else round(acc*100,2)} params={record['cost']['unique_params']:,} "
                                  f"train={train_info['seconds']:.1f}s", flush=True)
                        except Exception as exc:  # noqa: BLE001
                            record["error"] = f"{type(exc).__name__}: {exc}"
                            record["traceback"] = traceback.format_exc()
                            record["finished_utc"] = utc_now()
                            print(f"  FAILED {record['error']}", flush=True)
                        record["vram_after"] = vram_snapshot()
                        record["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                        with open(raw_path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                            fh.flush()
                        if model is not None:
                            del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    if args.max_runs is not None and runs >= args.max_runs:
                        break
                if args.max_runs is not None and runs >= args.max_runs:
                    break
            if args.max_runs is not None and runs >= args.max_runs:
                break
        if args.max_runs is not None and runs >= args.max_runs:
            break
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
