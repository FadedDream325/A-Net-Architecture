"""Phase 1 pilot: domain-ratio and batch/boundary mixing for 50M models.

Pilot only: 6M tokens per run, one seed by default.  It is not a final
math/code capability claim; accuracy here is next-byte accuracy, not exact
math-answer or code-execution accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from experiments.anet_v2.lang50m.model import SharedDepthANetLM, TransformerLM, parameter_counts
from experiments.anet_v2.lang50m.phase0 import flops_counts
from experiments.anet_v2.lang50m.run import get_batch

DATA = Path(__file__).resolve().parent / "data60m"
POOLS = {
    "math": (DATA / "math_train.bin", DATA / "math_val.bin", DATA / "math_test.bin"),
    "code": (DATA / "code_train.bin", DATA / "code_val.bin", DATA / "code_test.bin"),
}
TASK_ID = {"math": 0, "code": 1}


def load_split(domain: str, split: str) -> torch.Tensor:
    path = DATA / f"{domain}_{split}.bin"
    # Keep the backing bytearray alive; avoid a large Python int list.
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)


def make_model(arch: str):
    if arch == "transformer12":
        return TransformerLM(layers=12)
    if arch == "shared_depth":
        return SharedDepthANetLM()
    raise KeyError(arch)


def ratio_counts(ratio: float, batch: int) -> tuple[int, int]:
    m = int(round(batch * ratio))
    m = max(0, min(batch, m))
    return m, batch - m


def boundary_task(step: int, ratio: float) -> str:
    if ratio >= 1.0:
        return "math"
    if ratio <= 0.0:
        return "code"
    # Deterministic cycle: 0.75 -> M,M,M,C; 0.5 -> M,C; 0.25 -> M,C,C,C.
    denom = 4 if abs(ratio * 4 - round(ratio * 4)) < 1e-9 else 100
    numer = int(round(ratio * denom))
    return "math" if step % denom < numer else "code"


def batch_groups(step: int, ratio: float, mode: str, train: dict[str, torch.Tensor], batch: int, seq: int, device):
    if mode == "boundary":
        domain = boundary_task(step, ratio)
        x, y = get_batch(train[domain], batch, seq, device)
        return [(x, y, TASK_ID[domain], domain, batch)]
    m, c = ratio_counts(ratio, batch)
    groups = []
    for domain, n in (("math", m), ("code", c)):
        if n:
            x, y = get_batch(train[domain], n, seq, device)
            groups.append((x, y, TASK_ID[domain], domain, n))
    return groups


@torch.no_grad()
def evaluate_domain(model, data: torch.Tensor, task: int, device, seq: int, batches: int = 20):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for _ in range(batches):
        x, y = get_batch(data, 8, seq, device)
        logits = model(x, task)
        total_loss += float(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum"))
        total_correct += int((logits.argmax(dim=-1) == y).sum())
        total_tokens += y.numel()
    model.train()
    return {"bpc": total_loss / total_tokens / math.log(2),
            "next_byte_accuracy": total_correct / total_tokens}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2929)  # 6M tokens at 16*128
    p.add_argument("--seeds", nargs="+", type=int, default=[2026])
    p.add_argument("--archs", nargs="+", default=["transformer12", "shared_depth"])
    p.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.75, 0.5, 0.25, 0.0])
    p.add_argument("--modes", nargs="+", choices=["batch", "boundary"], default=["batch", "boundary"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "phase1_pilot")
    args = p.parse_args()
    if args.quick:
        args.steps = 8
        args.seeds = [2026]
        args.out_dir = args.out_dir.parent / "phase1_quick"

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_runs.jsonl"
    args.out_dir = out
    (out / "config.json").write_text(
        json.dumps({**vars(args), "out_dir": str(out)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train = {d: load_split(d, "train") for d in ("math", "code")}
    val = {d: load_split(d, "val") for d in ("math", "code")}
    test = {d: load_split(d, "test") for d in ("math", "code")}
    print({k: int(v.numel()) for k, v in train.items()}, flush=True)

    for arch in args.archs:
        for seed in args.seeds:
            for ratio in args.ratios:
                for mode in args.modes:
                    torch.manual_seed(seed)
                    model = make_model(arch).to(device)
                    stats = parameter_counts(model)
                    stats.update(flops_counts(model, args.seq))
                    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                    model.train()
                    # Warmup outside timing.
                    for ws in range(2):
                        for x, y, task, domain, n in batch_groups(ws, ratio, mode, train, args.batch, args.seq, device):
                            opt.zero_grad(set_to_none=True)
                            loss = F.cross_entropy(model(x, task).reshape(-1, 256), y.reshape(-1))
                            loss.backward()
                            opt.step()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    losses = []
                    for step in range(args.steps):
                        groups = batch_groups(step, ratio, mode, train, args.batch, args.seq, device)
                        opt.zero_grad(set_to_none=True)
                        loss_sum = 0.0
                        n_tokens = 0
                        for x, y, task, domain, n in groups:
                            logits = model(x, task)
                            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum")
                            loss_sum = loss_sum + loss
                            n_tokens += y.numel()
                        total = loss_sum / max(1, n_tokens)
                        total.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        losses.append(float(total.detach()))
                    torch.cuda.synchronize()
                    train_s = time.perf_counter() - start
                    val_m = evaluate_domain(model, val["math"], 0, device, args.seq, batches=10 if args.quick else 20)
                    val_c = evaluate_domain(model, val["code"], 1, device, args.seq, batches=10 if args.quick else 20)
                    train_m = evaluate_domain(model, train["math"], 0, device, args.seq, batches=5)
                    train_c = evaluate_domain(model, train["code"], 1, device, args.seq, batches=5)
                    rec = {
                        "arch": arch, "seed": seed, "ratio_math": ratio, "mode": mode,
                        "steps": args.steps, "batch": args.batch, "seq": args.seq,
                        "params": stats,
                        "train_seconds": train_s,
                        "train_loss_last": losses[-1] if losses else None,
                        "val": {"math": val_m, "code": val_c},
                        "train_bpc": {"math": train_m, "code": train_c},
                        "train_test_gap_bpc": {
                            "math": val_m["bpc"] - train_m["bpc"],
                            "code": val_c["bpc"] - train_c["bpc"],
                        },
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    }
                    with open(raw, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                    print(arch, seed, ratio, mode, "val", round(val_m["bpc"], 3), round(val_c["bpc"], 3),
                          "t", round(train_s, 1), "peak", round(rec["peak_allocated_bytes"]/1024**3, 2), flush=True)
                    del model
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
