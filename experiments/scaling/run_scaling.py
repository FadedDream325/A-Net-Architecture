"""Scaling experiment runner for A-Net.

This script performs controlled multi-task training and measurement for:
  - independent per-task MLPs
  - shared trunk + linear heads
  - current A-Net (shared L1 + task experts + Task-ID Top-1)
  - optional A-Net cross-layer parameter-block prototype

Run from the repository root:
    python -m experiments.scaling.run_scaling --help

Preflight example:
    python -m experiments.scaling.run_scaling \
        --scales small --models independent shared anet cross \
        --seeds 2026 --epochs 2 --max-steps 30 \
        --out-dir experiments/scaling/results/preflight

Full example:
    python -m experiments.scaling.run_scaling \
        --scales small medium large \
        --models independent shared anet cross \
        --seeds 2026 2027 2028 \
        --epochs 50 --out-dir experiments/scaling/results/scaling_v1 --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import (
    MODEL_KEYS,
    MODEL_LABELS,
    build_model,
    model_spec_rows,
    count_parameters,
)
from .scales import SCALES, get_scale
from .tasks import TASKS, build_task_suite

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments" / "scaling" / "results" / "scaling_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA but torch.cuda.is_available() == False")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def environment_info(device: torch.device) -> dict:
    info = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_commit": git_commit(),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "disk_free_gb": round(shutil.disk_usage(str(ROOT)).free / 1e9, 2),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_mem, total_mem = torch.cuda.mem_get_info()
        info.update(
            {
                "gpu_name": props.name,
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_sm_count": props.multi_processor_count,
                "gpu_total_memory_gb": round(total_mem / 1024**3, 3),
                "gpu_free_memory_gb": round(free_mem / 1024**3, 3),
                "gpu_arch_list": torch.cuda.get_arch_list(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def make_config(args: argparse.Namespace, device: torch.device) -> dict:
    return {
        "scales": args.scales,
        "models": args.models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "bench_batches": args.bench_batches,
        "bench_repeats": args.bench_repeats,
        "benchmark": args.benchmark,
        "save_checkpoints": args.save_checkpoints,
        "max_steps": args.max_steps,
    }


def train_model(
    model: torch.nn.Module,
    train_data: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    max_steps: int | None = None,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    steps = 0
    last_epoch_losses: list[float] = []
    total_train_samples = sum(x.size(0) for x, _ in train_data)
    stop = False

    for epoch in range(epochs):
        epoch_loss_sums = [0.0 for _ in train_data]
        epoch_counts = [0 for _ in train_data]
        for task_id, (x, y) in enumerate(train_data):
            permutation = torch.randperm(x.size(0), device=device)
            for start in range(0, x.size(0), batch_size):
                idx = permutation[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[idx], task_id)
                loss = F.cross_entropy(logits, y[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch={epoch}, task={task_id}, step={steps}"
                    )
                loss.backward()
                # Check gradients on the first few steps only; checking every step
                # would add timing overhead and hide throughput.
                if steps < 3:
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise FloatingPointError(
                                f"non-finite grad at epoch={epoch}, task={task_id}"
                            )
                optimizer.step()
                epoch_loss_sums[task_id] += float(loss.detach())
                epoch_counts[task_id] += 1
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    stop = True
                    break
            if stop:
                break
        last_epoch_losses = [
            s / max(c, 1) for s, c in zip(epoch_loss_sums, epoch_counts)
        ]
        if stop:
            break

    return {
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_requested": epochs,
        "steps": steps,
        "train_samples_per_epoch": total_train_samples,
        "last_epoch_losses": last_epoch_losses,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_data: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict:
    model.eval()
    accs: list[float] = []
    losses: list[float] = []
    for task_id, (x, y) in enumerate(test_data):
        logits = model(x, task_id)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        accs.append(float((pred == y).float().mean().item()))
        losses.append(float(loss.item()))
    return {
        "accuracy_per_task": accs,
        "accuracy_mean": float(sum(accs) / len(accs)),
        "test_loss_per_task": losses,
        "test_loss_mean": float(sum(losses) / len(losses)),
    }


def warmup_cuda() -> float:
    """Initialize CUDA/cuBLAS before measuring the first real training run."""
    if not torch.cuda.is_available():
        return 0.0
    start = time.perf_counter()
    a = torch.randn(512, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda", requires_grad=True)
    loss = (a @ b).sum()
    loss.backward()
    torch.cuda.synchronize()
    del a, b, loss
    torch.cuda.empty_cache()
    return time.perf_counter() - start


def warmup_models(args: argparse.Namespace, device: torch.device) -> float:
    """Run one real forward/backward per architecture and scale before timing."""
    if device.type != "cuda":
        return 0.0
    start = time.perf_counter()
    torch.randperm(512, device=device)
    x = torch.randn(512, 2, device=device)
    y = torch.randint(0, 2, (512,), device=device)
    for scale_key in args.scales:
        scale = get_scale(scale_key)
        for model_kind in args.models:
            set_seed(0)
            model = build_model(model_kind, scale.hidden_dim, scale.expert_hidden_dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x, 0), y)
            loss.backward()
            optimizer.step()
            del model, optimizer, loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    del x, y
    return time.perf_counter() - start


def _event_times_us(model, x, repeats: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    with torch.inference_mode():
        for i in range(repeats):
            starts[i].record()
            model(x, 0)
            ends[i].record()
        torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]


def _cpu_times_us(model, x, repeats: int) -> list[float]:
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(x, 0)
            times.append((time.perf_counter_ns() - start) / 1000.0)
    return times


def benchmark_inference(
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_sizes: list[int],
    repeats: int,
) -> dict:
    model.eval()
    results: dict[str, dict] = {}
    first_call_us = None
    peak_alloc = None
    peak_reserved = None

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 2, device=device)
        warmup = 30 if batch_size <= 256 else 10
        with torch.inference_mode():
            for _ in range(warmup):
                model(x, 0)
        if device.type == "cuda":
            torch.cuda.synchronize()

        if first_call_us is None:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode():
                    model(x, 0)
                end.record()
                torch.cuda.synchronize()
                first_call_us = start.elapsed_time(end) * 1000.0
            else:
                t0 = time.perf_counter_ns()
                with torch.inference_mode():
                    model(x, 0)
                first_call_us = (time.perf_counter_ns() - t0) / 1000.0

        effective_repeats = repeats if batch_size <= 256 else max(20, repeats // 4)
        wall_start = time.perf_counter()
        if device.type == "cuda":
            times_us = _event_times_us(model, x, effective_repeats)
        else:
            times_us = _cpu_times_us(model, x, effective_repeats)
        wall = time.perf_counter() - wall_start
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_us.sort()
        p50 = statistics.median(times_us)
        p95 = times_us[min(len(times_us) - 1, int(len(times_us) * 0.95))]
        results[str(batch_size)] = {
            "p50_us": p50,
            "p95_us": p95,
            "throughput_samples_s": batch_size * effective_repeats / max(wall, 1e-12),
            "repeats": effective_repeats,
        }

    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

    return {
        "device": str(device),
        "first_call_after_training_us": first_call_us,
        "batches": results,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "note": "First call is measured after training in a warm CUDA process, not process cold start.",
    }


def route_info(model_kind: str, test_size: int, num_tasks: int) -> dict:
    if model_kind == "independent":
        names = [f"mlp[{i}]" for i in range(num_tasks)]
    elif model_kind == "shared":
        names = [f"head[{i}]" for i in range(num_tasks)]
    else:
        names = [f"expert[{i}]" for i in range(num_tasks)]
    return {
        "routing_type": "task_id_deterministic" if model_kind in {"anet", "cross"} else "task_id_select",
        "selected_branch_per_task": names,
        "sample_count_per_task": test_size,
    }


def run_id(scale: str, model_kind: str, seed: int) -> str:
    return f"{scale}__{model_kind}__s{seed}"


def load_completed(raw_path: Path) -> set[str]:
    done: set[str] = set()
    if not raw_path.exists():
        return done
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                done.add(rec.get("run_id"))
    return done


def build_data_for_seed(seed: int, args: argparse.Namespace, device: torch.device):
    train_cpu, test_cpu, meta = build_task_suite(
        seed=seed, n_train=args.train_size, n_test=args.test_size
    )
    train = [(x.to(device), y.to(device)) for x, y in train_cpu]
    test = [(x.to(device), y.to(device)) for x, y in test_cpu]
    return train, test, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="A-Net parameter-scaling experiment")
    parser.add_argument("--scales", nargs="+", default=list(SCALES.keys()), choices=list(SCALES.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODEL_KEYS), choices=list(MODEL_KEYS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--test-size", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bench-batches", nargs="+", type=int, default=[1, 64, 256, 1024])
    parser.add_argument("--bench-repeats", type=int, default=200)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    raw_path = out_dir / "raw_runs.jsonl"
    config_path = out_dir / "config.json"
    config = make_config(args, device)

    if config_path.exists():
        if not args.resume:
            raise SystemExit(
                f"{out_dir} already has config.json; use a new --out-dir or pass --resume"
            )
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "scales",
            "models",
            "seeds",
            "epochs",
            "train_size",
            "test_size",
            "batch_size",
            "lr",
            "weight_decay",
            "max_steps",
        ):
            if old.get(key) != config.get(key):
                raise SystemExit(
                    f"resume config mismatch for {key}: {old.get(key)!r} != {config.get(key)!r}"
                )
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    env = environment_info(device)
    (out_dir / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 96)
    print("A-Net scaling experiment")
    print("=" * 96)
    print(f"device     : {device}")
    print(f"scales     : {args.scales}")
    print(f"models     : {args.models}")
    print(f"seeds      : {args.seeds}")
    print(f"epochs     : {args.epochs}  batch={args.batch_size}  max_steps={args.max_steps}")
    print(f"data       : train={args.train_size} / test={args.test_size}")
    print(f"output     : {out_dir}")
    print()

    done = load_completed(raw_path) if args.resume else set()
    if args.resume:
        print(f"resume: {len(done)} completed runs")

    dataset_cache: dict[int, tuple] = {}
    for seed in args.seeds:
        if seed not in dataset_cache:
            dataset_cache[seed] = build_data_for_seed(seed, args, device)

    warmup_seconds = warmup_cuda() if device.type == "cuda" else 0.0
    if warmup_seconds:
        print(f"CUDA warmup: {warmup_seconds:.2f}s", flush=True)
    model_warmup_seconds = warmup_models(args, device)
    if model_warmup_seconds:
        print(f"model warmup: {model_warmup_seconds:.2f}s", flush=True)

    runs_started = 0
    for scale_key in args.scales:
        scale = get_scale(scale_key)
        for model_kind in args.models:
            for seed in args.seeds:
                rid = run_id(scale_key, model_kind, seed)
                if rid in done:
                    print(f"[skip] {rid} (already completed)", flush=True)
                    continue
                if args.max_runs is not None and runs_started >= args.max_runs:
                    print(f"reached --max-runs={args.max_runs}; stopping", flush=True)
                    break

                runs_started += 1
                train_data, test_data, data_meta = dataset_cache[seed]
                print(f"[run] {rid}", flush=True)
                record = {
                    "run_id": rid,
                    "status": "failed",
                    "started_utc": utc_now(),
                    "scale": scale_key,
                    "model": model_kind,
                    "model_label": MODEL_LABELS[model_kind],
                    "seed": seed,
                    "scale_config": scale.__dict__,
                    "data": data_meta,
                    "config": config,
                }
                model = None
                try:
                    set_seed(seed)
                    model = build_model(
                        model_kind, scale.hidden_dim, scale.expert_hidden_dim
                    ).to(device)
                    cost = model_spec_rows(
                        model_kind,
                        scale.hidden_dim,
                        scale.expert_hidden_dim,
                        model,
                    )
                    record["cost"] = cost
                    record["model_class"] = type(model).__name__

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    train_start = time.perf_counter()
                    train_info = train_model(
                        model,
                        train_data,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        device=device,
                        max_steps=args.max_steps,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    train_seconds = time.perf_counter() - train_start
                    train_info["seconds"] = train_seconds
                    if device.type == "cuda":
                        train_info["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                        train_info["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    record["train"] = train_info

                    evaluation = evaluate_model(model, test_data, device)
                    record["metrics"] = evaluation
                    record["routing"] = route_info(
                        model_kind, args.test_size, len(test_data)
                    )

                    checkpoint_path = checkpoint_dir / f"{rid}.pt"
                    if args.save_checkpoints:
                        torch.save(
                            {
                                "model_kind": model_kind,
                                "scale": scale_key,
                                "seed": seed,
                                "hidden_dim": scale.hidden_dim,
                                "expert_hidden_dim": scale.expert_hidden_dim,
                                "state_dict": model.state_dict(),
                                "cost": cost,
                            },
                            checkpoint_path,
                        )
                        record["checkpoint_bytes"] = checkpoint_path.stat().st_size
                        record["checkpoint_path"] = str(checkpoint_path.resolve().relative_to(ROOT.resolve()))
                    else:
                        record["checkpoint_bytes"] = None

                    if args.benchmark:
                        record["inference"] = benchmark_inference(
                            model,
                            device=device,
                            batch_sizes=args.bench_batches,
                            repeats=args.bench_repeats,
                        )
                    else:
                        record["inference"] = None

                    record["status"] = "ok"
                    record["finished_utc"] = utc_now()
                    print(
                        f"  acc={evaluation['accuracy_mean']*100:.2f}%  "
                        f"params={cost['unique_trainable_parameters']:,}  "
                        f"active={cost['active_parameters_per_sample']:,}  "
                        f"train={train_seconds:.2f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["traceback"] = traceback.format_exc()
                    record["finished_utc"] = utc_now()
                    print(f"  FAILED: {record['error']}", flush=True)

                with open(raw_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                if model is not None:
                    del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if args.max_runs is not None and runs_started >= args.max_runs:
                break
        if args.max_runs is not None and runs_started >= args.max_runs:
            break

    print("finished", flush=True)
    print(f"raw results: {raw_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
