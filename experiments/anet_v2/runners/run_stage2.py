"""Stage 2: cross-workflow parameter sharing and gradient conflict."""

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

from experiments.anet_v2.core.device_guard import enforce_dedicated_vram, snapshot as vram_snapshot
from experiments.anet_v2.core.gradient_audit import shared_gradient_audit
from experiments.anet_v2.core.models import ANetModel, MODEL_LABELS, build_model
from experiments.anet_v2.core.tasks import CORRELATIONS, build_suite
from experiments.anet_v2.core.train import TrainConfig, evaluate_model, set_seed, train_model
from experiments.anet_v2.runners.run_stage1 import PeakRSS, environment, model_cost


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(width: int, corr: str, mode: str, model: str, seed: int) -> str:
    return f"h{width:04d}__{corr}__{mode}__{model}__s{seed}"


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "ok":
                    done.add(rec.get("run_id"))
    return done


def warmup(args, suite, device) -> float:
    if device.type != "cuda":
        return 0.0
    start = time.perf_counter()
    a = torch.randn(256, 128, device=device)
    b = torch.randn(128, 128, device=device, requires_grad=True)
    (a @ b).sum().backward()
    torch.cuda.synchronize()
    for kind in ["independent", "anet_basic", "anet_share", "anet_adapter"]:
        set_seed(0)
        model = build_model(
            kind,
            num_tasks=len(suite.specs),
            in_dim=suite.meta["in_dim"],
            hidden_dim=128,
            out_dims=[s.out_dim for s in suite.specs],
            active_mode="pair",
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = suite.x_train[: min(64, suite.x_train.size(0))]
        y = suite.y_train[0][: x.size(0)]
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x, 0), y)
        loss.backward()
        opt.step()
        del model, opt, loss, x, y
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return time.perf_counter() - start


def main() -> int:
    p = argparse.ArgumentParser(description="A-Net v2 Stage 2")
    p.add_argument("--widths", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--models", nargs="+", default=["independent", "anet_basic", "anet_share", "anet_adapter"])
    p.add_argument("--active-modes", nargs="+", choices=("pair", "dense"), default=["pair", "dense"])
    p.add_argument("--correlations", nargs="+", default=["low", "medium", "high"], choices=list(CORRELATIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--val-size", type=int, default=1500)
    p.add_argument("--test-size", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--min-free-vram-gb", type=float, default=2.0)
    p.add_argument("--max-vram-fraction", type=float, default=0.80)
    p.add_argument("--bench-batches", nargs="+", type=int, default=[1, 256, 1024])
    p.add_argument("--bench-repeats", type=int, default=100)
    p.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage2")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Stage 2 requires CUDA")
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    guard = enforce_dedicated_vram(min_free_gb=args.min_free_vram_gb, max_fraction=args.max_vram_fraction)
    print("VRAM guard:", guard, flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_runs.jsonl"
    config_path = out_dir / "config.json"
    config = {
        "widths": args.widths,
        "models": args.models,
        "active_modes": args.active_modes,
        "correlations": args.correlations,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "min_free_vram_gb": args.min_free_vram_gb,
        "max_vram_fraction": args.max_vram_fraction,
        "bench_batches": args.bench_batches,
        "bench_repeats": args.bench_repeats,
        "benchmark": args.benchmark,
        "max_steps": args.max_steps,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != config_hash and not args.resume:
            raise SystemExit(f"{out_dir} has a different config")
    else:
        config_path.write_text(json.dumps({"config": config, "config_hash": config_hash}, indent=2), encoding="utf-8")
    env = environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    print(f"Stage 2 modes={args.active_modes} widths={args.widths} models={args.models}", flush=True)
    done = load_done(raw_path) if args.resume else set()
    data_cache = {}
    warm_suite = build_suite(args.seeds[0], args.correlations[0],
                             n_train=args.train_size, n_val=args.val_size, n_test=args.test_size).to(device)
    data_cache[(args.correlations[0], args.seeds[0])] = warm_suite
    print(f"warmup: {warmup(args, warm_suite, device):.2f}s", flush=True)
    runs = 0

    for mode in args.active_modes:
        for corr in args.correlations:
            for seed in args.seeds:
                key = (corr, seed)
                if key not in data_cache:
                    data_cache[key] = build_suite(
                        seed, corr, n_train=args.train_size, n_val=args.val_size, n_test=args.test_size
                    ).to(device)
                suite = data_cache[key]
                for width in args.widths:
                    for kind in args.models:
                        if kind == "independent" and mode != args.active_modes[0]:
                            continue
                        rid = run_id(width, corr, mode, kind, seed)
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
                            "active_mode": mode,
                            "width": width,
                            "correlation": corr,
                            "seed": seed,
                            "model": kind,
                            "model_label": MODEL_LABELS[kind],
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
                                in_dim=suite.meta["in_dim"],
                                hidden_dim=width,
                                out_dims=[s.out_dim for s in suite.specs],
                                active_mode=mode,
                            ).to(device)
                            record["cost"] = model_cost(model, kind, suite, suite.meta["in_dim"], width,
                                                        [s.out_dim for s in suite.specs])
                            record["identity"] = record["cost"].get("identity")

                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            with PeakRSS() as rss:
                                train_info = train_model(
                                    model, suite,
                                    TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                                                lr=args.lr, weight_decay=args.weight_decay,
                                                max_steps=args.max_steps),
                                    device=device,
                                )
                            torch.cuda.synchronize()
                            train_info["peak_rss_bytes"] = rss.peak
                            train_info["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                            train_info["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                            record["train"] = train_info
                            record["val"] = evaluate_model(model, suite, split="val").__dict__
                            record["test"] = evaluate_model(model, suite, split="test").__dict__

                            if isinstance(model, ANetModel):
                                record["gradient_audit"] = shared_gradient_audit(model, suite)
                            else:
                                record["gradient_audit"] = {"shared": False, "reason": "baseline"}

                            if args.benchmark:
                                modes = ["single"]
                                if isinstance(model, ANetModel):
                                    modes = ["single", mode]
                                record["inference"] = benchmark_modes(
                                    model, suite, device, args.bench_batches, args.bench_repeats, modes
                                )
                            else:
                                record["inference"] = None
                            record["status"] = "ok"
                            record["finished_utc"] = utc_now()
                            acc = record["test"]["mean"].get("accuracy")
                            print(f"  ok acc={None if acc is None else round(acc*100,2)} "
                                  f"params={record['cost']['unique_params']:,} "
                                  f"train={train_info['seconds']:.1f}s "
                                  f"grad_cos={record['gradient_audit'].get('mean_pairwise_cosine')}", flush=True)
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


def benchmark_modes(model, suite, device, batches, repeats, modes):
    from experiments.anet_v2.core.benchmark import benchmark_inference

    return benchmark_inference(
        model, suite, device=device, batch_sizes=batches, repeats=repeats,
        modes=modes, warmup=20, reset_cache_each_repeat=True, inference_cache=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
