"""Phase 0: smoke-test the 50M language stack before the Phase 1 data pipeline.

Checks:
- parameter/active-parameter/storage accounting;
- approximate per-token FLOPs;
- fixed train/val split and no byte n-gram overlap;
- forward/backward, finite gradients, optimizer step;
- checkpoint save/load with identical logits;
- train/val BPC and train-test gap for each task;
- peak memory and runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
    NoTerminalANetLM,
    SharedDepthANetLM,
    TransformerLM,
    parameter_counts,
)
from experiments.anet_v2.lang50m.run import get_batch, make_tasks


def block_flops(block, seq: int) -> int:
    d = int(block.attn.embed_dim)
    ffn = int(block.mlp[0].out_features)
    # QKV+out projections, FFN, attention score/value products, two norms.
    return 4 * d * d + 2 * d * ffn + 4 * seq * d + 4 * d


def flops_counts(model, seq: int) -> dict:
    if isinstance(model, TransformerLM):
        d = int(model.head.in_features)
        blocks = sum(block_flops(b, seq) for b in model.blocks)
        head = 2 * d * int(model.head.out_features)
        return {"active_flops_per_token": blocks + head,
                "reference_flops_per_token": blocks + head,
                "active_blocks": len(model.blocks), "unique_blocks": len(model.blocks)}
    if isinstance(model, SharedBackboneLM):
        d = int(model.heads[0].in_features)
        blocks = sum(block_flops(b, seq) for b in model.blocks)
        head = 2 * d * int(model.heads[0].out_features)
        return {"active_flops_per_token": blocks + head,
                "reference_flops_per_token": blocks + head,
                "active_blocks": len(model.blocks), "unique_blocks": len(model.blocks)}
    if isinstance(model, NoTerminalANetLM):
        d = int(model.heads[0].in_features)
        active_blocks = list(model.pre) + list(model.workflows[0]) + list(model.shared)
        active = sum(block_flops(b, seq) for b in active_blocks)
        head = 2 * d * int(model.heads[0].out_features)
        all_blocks = list(model.pre) + [b for wf in model.workflows for b in wf] + list(model.shared)
        return {"active_flops_per_token": active + head,
                "reference_flops_per_token": sum(block_flops(b, seq) for b in all_blocks) + head,
                "active_blocks": len(active_blocks), "unique_blocks": len(all_blocks)}
    if isinstance(model, SharedDepthANetLM):
        d = int(model.heads[0].in_features)
        active_blocks = model.shared + model.workflows[0]
        active = sum(block_flops(b, seq) for b in active_blocks)
        head = 2 * d * int(model.heads[0].out_features)
        all_blocks = list(model.shared) + [b for wf in model.workflows for b in wf]
        return {"active_flops_per_token": active + head,
                "reference_flops_per_token": sum(block_flops(b, seq) for b in all_blocks) + head,
                "active_blocks": len(active_blocks), "unique_blocks": len(all_blocks)}
    if isinstance(model, ANetLM):
        d = int(model.heads[0].in_features)
        active_blocks = list(model.pre) + list(model.workflows[0]) + list(model.workflows[1]) + list(model.shared)
        active = sum(block_flops(b, seq) for b in active_blocks)
        head = 2 * d * int(model.heads[0].out_features)
        all_blocks = list(model.pre) + [b for wf in model.workflows for b in wf] + list(model.shared)
        return {"active_flops_per_token": active + head,
                "reference_flops_per_token": sum(block_flops(b, seq) for b in all_blocks) + head,
                "active_blocks": len(active_blocks), "unique_blocks": len(all_blocks)}
    raise TypeError(type(model))


def evaluate_bpc(model, tasks, device, seq: int, batches: int = 2) -> dict:
    model.eval()
    out = {}
    with torch.no_grad():
        for task_id, (name, (train, val)) in enumerate(tasks.items()):
            for split_name, split in (("train", train), ("val", val)):
                total, n = 0.0, 0
                for _ in range(batches):
                    x, y = get_batch(split, 2, seq, device)
                    logits = model(x, task_id)
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
                    total += float(loss)
                    n += 1
                out[f"{name}/{split_name}"] = total / n / math.log(2)
    model.train()
    return out


def ngram_overlap(train: torch.Tensor, val: torch.Tensor, n: int = 32, limit: int = 20000) -> int:
    def grams(x):
        x = x[:limit].tolist()
        return {tuple(x[i:i+n]) for i in range(max(0, len(x)-n))}
    return len(grams(train) & grams(val))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chars-per-task", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "phase0")
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.cuda.init()
    torch.manual_seed(2026)
    tasks = make_tasks(args.chars_per_task)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir()

    split_checks = {}
    for name, (train, val) in tasks.items():
        split_checks[name] = {
            "train_bytes": int(train.numel()),
            "val_bytes": int(val.numel()),
            "overlap_32gram_common_idioms": ngram_overlap(train, val, n=32),
            "overlap_256gram": ngram_overlap(train, val, n=256),
            "split_index_disjoint": train.numel() + val.numel() == args.chars_per_task,
            "min_train": int(train.min()),
            "max_train": int(train.max()),
            "min_val": int(val.min()),
            "max_val": int(val.max()),
            "train_last_index": int(train.numel()) - 1,
            "val_first_index": int(train.numel()),
        }

    specs = {
        "transformer12": lambda: TransformerLM(layers=12),
        "shared_depth": lambda: SharedDepthANetLM(),
        "sparse_anet": lambda: ANetLM(),
        "shared_backbone": lambda: SharedBackboneLM(),
    }
    records = []
    for arch, factory in specs.items():
        torch.manual_seed(2026)
        model = factory().to(device)
        model.train()
        stats = parameter_counts(model)
        stats.update(flops_counts(model, args.seq))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        x, y = get_batch(tasks["math"][0], args.batch, args.seq, device)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        logits = model(x, 0)
        shape_ok = tuple(logits.shape) == (args.batch, args.seq, 256)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        train_s = time.perf_counter() - start
        bpc = evaluate_bpc(model, tasks, device, args.seq, batches=2)
        model.eval()
        with torch.no_grad():
            logits_before = model(x, 0).detach()
        ckpt = ckpt_dir / f"{arch}.pt"
        torch.save({"arch": arch, "state_dict": model.state_dict(),
                    "seq": args.seq, "chars_per_task": args.chars_per_task}, ckpt)
        fresh = factory().to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        fresh.load_state_dict(payload["state_dict"])
        fresh.eval()
        with torch.no_grad():
            logits_after = fresh(x, 0)
        max_diff = float((logits_before - logits_after).abs().max())
        records.append({
            "arch": arch,
            "params": stats,
            "output_shape_ok": shape_ok,
            "loss": float(loss.detach()),
            "grads_finite": finite,
            "checkpoint_bytes": ckpt.stat().st_size,
            "checkpoint_max_abs_diff": max_diff,
            "bpc": bpc,
            "train_step_seconds": train_s,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        })
        print(arch, "params", stats["unique_trainable"], "active", stats["active_parameters_one_forward"],
              "bpc", {k: round(v, 3) for k, v in bpc.items()},
              "ckpt_diff", max_diff, "finite", finite, flush=True)
        del model, fresh
        torch.cuda.empty_cache()

    if not args.keep_checkpoints:
        shutil.rmtree(ckpt_dir)
    result = {
        "config": vars(args) | {"out_dir": str(out_dir)},
        "split_checks": split_checks,
        "records": records,
        "all_shape_ok": all(r["output_shape_ok"] for r in records),
        "all_grads_finite": all(r["grads_finite"] for r in records),
        "all_checkpoints_exact": all(r["checkpoint_max_abs_diff"] == 0.0 for r in records),
        "all_splits_disjoint": all(v["overlap_256gram"] == 0 and v["split_index_disjoint"] for v in split_checks.values()),
    }
    (out_dir / "phase0_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("all_shape_ok", "all_grads_finite", "all_checkpoints_exact", "all_splits_disjoint")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
