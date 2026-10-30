"""Stage 1 runner: controlled width scaling for the six v2 architectures."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.anet_v2.core.benchmark import benchmark_inference
from experiments.anet_v2.core.device_guard import enforce_dedicated_vram, snapshot as vram_snapshot
from experiments.anet_v2.core.models import ANetModel, MODEL_KEYS, MODEL_LABELS, build_model
from experiments.anet_v2.core.tasks import CORRELATIONS, build_suite
from experiments.anet_v2.core.train import TrainConfig, evaluate_model, set_seed, train_model


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        return hashlib.sha256(diff).hexdigest()
    except Exception:
        return "unknown"


def code_hash() -> str:
    h = hashlib.sha256()
    files = sorted((ROOT / "experiments" / "anet_v2").rglob("*.py")) + sorted((ROOT / "tests").glob("test_*.py")) + [ROOT / "tests" / "conftest.py"]
    for path in files:
        if not path.exists():
            continue
        h.update(str(path.relative_to(ROOT)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    info = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_hash": git_dirty_hash(),
        "git_status": git_status(),
        "code_hash": code_hash(),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "vram": vram_snapshot() if torch.cuda.is_available() else {"cuda": False},
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_mem, total_mem = torch.cuda.mem_get_info()
        info.update(
            {
                "gpu_name": props.name,
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_total_memory_gb": round(total_mem / 1024**3, 3),
                "gpu_free_memory_gb": round(free_mem / 1024**3, 3),
                "gpu_sm_count": props.multi_processor_count,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 2)
        info["ram_available_gb"] = round(vm.available / 1024**3, 2)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except Exception:
        info["cpu_count"] = os.cpu_count()
    return info


class PeakRSS:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            import psutil

            proc = psutil.Process()
            while not self._stop.is_set():
                self.peak = max(self.peak, proc.memory_info().rss)
                time.sleep(self.interval)
        except Exception:
            return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False


def baseline_cost(model: torch.nn.Module, kind: str, in_dim: int, hidden_dim: int, out_dims: list[int], num_tasks: int) -> dict:
    unique = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if kind == "independent":
        one = model.mlps[0]
        active = sum(p.numel() for p in one.parameters())
        flops = 2 * (in_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * out_dims[0])
        refs = unique
        executions = 1
    else:
        active = unique
        flops = 2 * (in_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * out_dims[0])
        refs = unique
        executions = 1
    return {
        "unique_params": unique,
        "workflow_unique_params": None,
        "terminal_params": 0,
        "adapter_params_total": 0,
        "adapter_params_executed": 0,
        "share_mode": "baseline",
        "cache_enabled": False,
        "active_mode": "baseline",
        "accounting": {
            "reference_params": refs,
            "active_params_per_sample": active,
            "theoretical_flops_per_sample": flops,
            "theoretical_executions_per_sample": executions,
            "actual_executions_per_sample": executions,
            "actual_flops_per_sample": flops,
            "adapter_params": 0,
        },
        "cache": {"enabled": False, "hits": 0, "misses": 0, "hit_rate": 0.0},
        "graph": None,
    }


def model_cost(model: torch.nn.Module, kind: str, suite, in_dim: int, hidden_dim: int, out_dims: list[int]) -> dict:
    if isinstance(model, ANetModel):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            x = suite.x_test[: min(8, suite.x_test.size(0))].to(next(model.parameters()).device)
            model(x, 0)
        summary = model.cost_summary()
        if was_training:
            model.train()
        return summary
    return baseline_cost(model, kind, in_dim, hidden_dim, out_dims, len(suite.specs))


def warmup(args, suite, device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    start = time.perf_counter()
    a = torch.randn(256, 128, device=device)
    b = torch.randn(128, 128, device=device, requires_grad=True)
    (a @ b).sum().backward()
    torch.cuda.synchronize()
    for width in args.widths:
        for kind in args.models:
            set_seed(0)
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


def run_id(width: int, corr: str, model_kind: str, seed: int, active_mode: str) -> str:
    return f"h{width:04d}__{corr}__{active_mode}__{model_kind}__s{seed}"


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as fh:
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


def main() -> int:
    p = argparse.ArgumentParser(description="A-Net v2 Stage 1 scaling")
    p.add_argument("--widths", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--models", nargs="+", default=["independent", "shared", "anet_basic", "anet_full"], choices=list(MODEL_KEYS))
    p.add_argument("--correlations", nargs="+", default=["low", "medium", "high"], choices=list(CORRELATIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--val-size", type=int, default=1500)
    p.add_argument("--test-size", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--active-mode", choices=("sparse", "pair", "dense"), default="sparse")
    p.add_argument("--cache-key-mode", choices=("identity", "content"), default="identity")
    p.add_argument("--gradient-isolation", choices=("none", "detach_aux", "detach_shared", "detach_both"), default="detach_both")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--min-free-vram-gb", type=float, default=2.0)
    p.add_argument("--max-vram-fraction", type=float, default=0.80)
    p.add_argument("--bench-batches", nargs="+", type=int, default=[1, 64, 256, 1024])
    p.add_argument("--bench-repeats", type=int, default=100)
    p.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reset-cache-each-repeat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--inference-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage1")
    args = p.parse_args()

    if args.preflight:
        args.widths = [64]
        args.models = ["independent", "shared", "anet_basic", "anet_full"]
        args.correlations = ["medium"]
        args.seeds = [2026]
        args.epochs = 2
        args.max_steps = 30
        args.bench_batches = [1, 64]
        args.bench_repeats = 20
        args.save_checkpoints = False

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    elif device.type == "cuda":
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = False
        vram_guard = enforce_dedicated_vram(
            min_free_gb=args.min_free_vram_gb,
            max_fraction=args.max_vram_fraction,
        )
        print(f"VRAM guard: {vram_guard}", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_runs.jsonl"
    config_path = out_dir / "config.json"
    experiment_config = {
        "widths": args.widths,
        "models": args.models,
        "correlations": args.correlations,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "active_mode": args.active_mode,
        "cache_key_mode": args.cache_key_mode,
        "gradient_isolation": args.gradient_isolation,
        "min_free_vram_gb": args.min_free_vram_gb,
        "max_vram_fraction": args.max_vram_fraction,
        "device": str(device),
        "bench_batches": args.bench_batches,
        "bench_repeats": args.bench_repeats,
        "benchmark": args.benchmark,
        "reset_cache_each_repeat": args.reset_cache_each_repeat,
        "inference_cache": args.inference_cache,
        "save_checkpoints": args.save_checkpoints,
        "max_steps": args.max_steps,
    }
    config_hash = hashlib.sha256(json.dumps(experiment_config, sort_keys=True).encode()).hexdigest()
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != config_hash and not args.resume:
            raise SystemExit(f"{out_dir} has a different config; use --resume or a new directory")
    else:
        config_path.write_text(json.dumps({"config": experiment_config, "config_hash": config_hash}, indent=2), encoding="utf-8")
    env = environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    print("=" * 100)
    print("A-Net v2 Stage 1")
    print("=" * 100)
    print(f"device={device} widths={args.widths} models={args.models} corr={args.correlations} seeds={args.seeds}")
    print(f"epochs={args.epochs} batch={args.batch_size} active_mode={args.active_mode} out={out_dir}")
    print()

    done = load_done(raw_path) if args.resume else set()
    if args.resume:
        print(f"resume: {len(done)} done")
    data_cache = {}
    warm_suite = build_suite(
        args.seeds[0],
        args.correlations[0],
        n_train=args.train_size,
        n_val=args.val_size,
        n_test=args.test_size,
    ).to(device)
    data_cache[(args.correlations[0], args.seeds[0])] = warm_suite
    warm_seconds = warmup(args, warm_suite, device)
    if warm_seconds:
        print(f"warmup: {warm_seconds:.2f}s", flush=True)
    runs_started = 0

    for corr in args.correlations:
        for seed in args.seeds:
            key = (corr, seed)
            if key not in data_cache:
                suite = build_suite(
                    seed,
                    corr,
                    n_train=args.train_size,
                    n_val=args.val_size,
                    n_test=args.test_size,
                ).to(device)
                data_cache[key] = suite
            suite = data_cache[key]
            for width in args.widths:
                for model_kind in args.models:
                    rid = run_id(width, corr, model_kind, seed, args.active_mode)
                    if rid in done:
                        print(f"[skip] {rid}", flush=True)
                        continue
                    if args.max_runs is not None and runs_started >= args.max_runs:
                        print(f"reached max-runs={args.max_runs}", flush=True)
                        break
                    runs_started += 1
                    if device.type == "cuda":
                        enforce_dedicated_vram(
                            min_free_gb=args.min_free_vram_gb,
                            max_fraction=args.max_vram_fraction,
                        )
                    print(f"[run] {rid}", flush=True)
                    record = {
                        "run_id": rid,
                        "status": "failed",
                        "started_utc": utc_now(),
                        "correlation": corr,
                        "seed": seed,
                        "width": width,
                        "model": model_kind,
                        "model_label": MODEL_LABELS[model_kind],
                        "active_mode": args.active_mode,
                        "data": suite.meta,
                        "config": experiment_config,
                        "config_hash": config_hash,
                        "git_commit": env["git_commit"],
                        "git_dirty_hash": env["git_dirty_hash"],
                        "git_status": env["git_status"],
                        "code_hash": env["code_hash"],
                        "vram_before": vram_snapshot() if device.type == "cuda" else {"cuda": False},
                    }
                    model = None
                    try:
                        set_seed(seed)
                        model = build_model(
                            model_kind,
                            num_tasks=len(suite.specs),
                            in_dim=suite.meta["in_dim"],
                            hidden_dim=width,
                            out_dims=[s.out_dim for s in suite.specs],
                            active_mode=args.active_mode,
                            cache_key_mode=args.cache_key_mode,
                            gradient_isolation=args.gradient_isolation,
                        ).to(device)
                        record["cost"] = model_cost(model, model_kind, suite, suite.meta["in_dim"], width, [s.out_dim for s in suite.specs])

                        if device.type == "cuda":
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                        train_cfg = TrainConfig(
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            max_steps=args.max_steps,
                        )
                        with PeakRSS() as rss:
                            train_info = train_model(model, suite, train_cfg, device=device)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        train_info["peak_rss_bytes"] = rss.peak
                        if device.type == "cuda":
                            train_info["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                            train_info["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                        record["train"] = train_info

                        record["val"] = evaluate_model(model, suite, split="val").__dict__
                        record["test"] = evaluate_model(model, suite, split="test").__dict__
                        record["routing"] = {"assign_mode": args.active_mode, "num_workflows": len(suite.specs)}

                        if args.save_checkpoints:
                            ckpt_dir = out_dir / "checkpoints"
                            ckpt_dir.mkdir(exist_ok=True)
                            ckpt_path = ckpt_dir / f"{rid}.pt"
                            torch.save(
                                {
                                    "run_id": rid,
                                    "model_kind": model_kind,
                                    "width": width,
                                    "correlation": corr,
                                    "seed": seed,
                                    "state_dict": model.state_dict(),
                                    "config": experiment_config,
                                },
                                ckpt_path,
                            )
                            record["checkpoint_bytes"] = ckpt_path.stat().st_size
                        else:
                            record["checkpoint_bytes"] = None

                        if args.benchmark:
                            modes = ["single"]
                            if isinstance(model, ANetModel):
                                modes = ["single", "pair", "mixed"]
                            record["inference"] = benchmark_inference(
                                model,
                                suite,
                                device=device,
                                batch_sizes=args.bench_batches,
                                repeats=args.bench_repeats,
                                modes=modes,
                                reset_cache_each_repeat=args.reset_cache_each_repeat,
                                inference_cache=args.inference_cache,
                            )
                            if isinstance(model, ANetModel):
                                record["cost_after_bench"] = model.cost_summary()
                        else:
                            record["inference"] = None

                        record["status"] = "ok"
                        record["finished_utc"] = utc_now()
                        acc = record["test"]["mean"].get("accuracy")
                        print(
                            f"  ok acc={acc if acc is None else round(acc*100,2)} "
                            f"params={record['cost']['unique_params']:,} "
                            f"train={train_info['seconds']:.1f}s",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        record["traceback"] = traceback.format_exc()
                        record["finished_utc"] = utc_now()
                        print(f"  FAILED {record['error']}", flush=True)
                    record["vram_after"] = vram_snapshot() if device.type == "cuda" else {"cuda": False}
                    if device.type == "cuda":
                        record["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                        record["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
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
        if args.max_runs is not None and runs_started >= args.max_runs:
            break

    print("finished", flush=True)
    print(f"raw: {raw_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
