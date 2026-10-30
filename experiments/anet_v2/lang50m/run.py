"""Run the 50M char-level pilot for three architectures."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.anet_v2.lang50m.model import (
    ANetLM,
    SharedBackboneLM,
    TransformerLM,
    parameter_counts,
)

DATA = Path(__file__).resolve().parent / "data"


def load_bytes() -> dict[str, bytes]:
    text = (DATA / "shakespeare.txt").read_bytes()
    code_files = sorted(DATA.glob("*.py"))
    code = b"\n".join(p.read_bytes() for p in code_files)
    math_text = (DATA / "math.txt").read_bytes()
    return {"text": text, "code": code, "math": math_text}


def repeat_to(data: bytes, n: int) -> bytes:
    if not data:
        raise ValueError("empty corpus")
    reps = (n + len(data) - 1) // len(data)
    return (data * reps)[:n]


def make_tasks(chars_per_task: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    raw = load_bytes()
    out = {}
    for name, data in raw.items():
        data = repeat_to(data, chars_per_task)
        arr = torch.tensor(list(data), dtype=torch.uint8)
        cut = int(arr.numel() * 0.9)
        out[name] = (arr[:cut], arr[cut:])
    return out


def get_batch(split: torch.Tensor, batch: int, seq: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n = split.numel() - seq - 1
    starts = torch.randint(0, n, (batch,))
    x = torch.stack([split[s : s + seq] for s in starts]).long()
    y = torch.stack([split[s + 1 : s + 1 + seq] for s in starts]).long()
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tasks, device, seq: int, batches: int = 20) -> dict:
    model.eval()
    out = {}
    for task_id, (name, (_train, val)) in enumerate(tasks.items()):
        total = 0.0
        n = 0
        for _ in range(batches):
            x, y = get_batch(val, 8, seq, device)
            logits = model(x, task_id)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total += float(loss)
            n += 1
        out[name] = total / n / math.log(2)
    model.train()
    return out


@torch.no_grad()
def benchmark(model, device, seq: int, batch: int, repeats: int = 20) -> dict:
    model.eval()
    x = torch.randint(0, 256, (batch, seq), device=device)
    for _ in range(5):
        model(x, 0)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    with torch.inference_mode():
        for i in range(repeats):
            starts[i].record()
            model(x, 0)
            ends[i].record()
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)]
    return {
        "batch": batch,
        "p50_us": statistics.median(samples),
        "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "samples_s": batch * repeats / (sum(samples) / 1e6),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--archs", nargs="+", default=["transformer", "shared", "anet"])
    p.add_argument("--chars-per-task", type=int, default=1_500_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = p.parse_args()
    if args.quick:
        args.steps = 40
        args.seeds = [2026]
        args.chars_per_task = 200_000
        args.batch = 8

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.cuda.init()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(0)
    tasks = make_tasks(args.chars_per_task)
    print("corpora", {k: (v[0].numel(), v[1].numel()) for k, v in tasks.items()}, flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / "raw_runs.jsonl"

    for arch in args.archs:
        for seed in args.seeds:
            torch.manual_seed(seed)
            if arch == "transformer":
                model = TransformerLM()
            elif arch == "transformer37":
                # 12 layers: unique/active ~37.50M, matching A-Net's active count.
                model = TransformerLM(layers=12)
            elif arch == "shared":
                model = SharedBackboneLM()
            elif arch == "anet":
                model = ANetLM()
            else:
                raise KeyError(arch)
            model.to(device)
            stats = parameter_counts(model)
            print(arch, seed, stats["unique_trainable"], f"{stats['unique_trainable']/1e6:.2f}M", flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            model.train()
            # Warmup: exclude first-kernel/allocator effects from the timing.
            for _ in range(2):
                warm_x, warm_y = get_batch(tasks["text"][0], args.batch, args.seq, device)
                opt.zero_grad(set_to_none=True)
                warm_logits = model(warm_x, 0)
                warm_loss = F.cross_entropy(warm_logits.reshape(-1, 256), warm_y.reshape(-1))
                warm_loss.backward()
                opt.step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            losses = []
            for step in range(args.steps):
                names = list(tasks)
                task_id = step % 3
                name = names[task_id]
                train = tasks[name][0]
                x, y = get_batch(train, args.batch, args.seq, device)
                opt.zero_grad(set_to_none=True)
                logits = model(x, task_id)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            torch.cuda.synchronize()
            train_s = time.perf_counter() - start
            bpc = evaluate(model, tasks, device, args.seq, batches=10 if args.quick else 20)
            lat1 = benchmark(model, device, args.seq, 1)
            latb = benchmark(model, device, args.seq, args.batch)
            rec = {
                "arch": arch,
                "seed": seed,
                "params": stats,
                "steps": args.steps,
                "batch": args.batch,
                "seq": args.seq,
                "chars_per_task": args.chars_per_task,
                "train_seconds": train_s,
                "train_loss_last": losses[-1] if losses else None,
                "bits_per_byte": bpc,
                "bits_per_byte_mean": statistics.mean(bpc.values()),
                "latency_batch1": lat1,
                "latency_batch": latb,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
            with open(raw, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            print(arch, seed, "bpc", {k: round(v, 4) for k, v in bpc.items()},
                  "train_s", round(train_s, 1), "peakGB", round(rec["peak_allocated_bytes"] / 1024**3, 2), flush=True)
            del model
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
