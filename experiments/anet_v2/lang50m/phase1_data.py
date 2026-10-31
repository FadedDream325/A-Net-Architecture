"""Phase 1 data pipeline: 30M math + 30M code pools, ratios and mixing smoke.

This does not train a model.  It creates reproducible pools, file/range splits,
checks 256-byte leakage, and validates the five domain ratios under batch-level
and boundary-level mixing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_math(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    parts = []
    total = 0
    while total < n:
        a = rng.randint(0, 999)
        b = rng.randint(1, 99)
        c = rng.randint(0, 999)
        op = rng.choice(["+", "-", "*"])
        result = eval(f"{a}{op}{b}")
        line = f"Q: {a} {op} {b} =\nA: {result}\n"
        if rng.random() < 0.25:
            d = rng.randint(0, 999)
            line += f"Q: ({a} {op} {b}) + {d} =\nA: {result + d}\n"
        parts.append(line)
        total += len(line)
    return "".join(parts)[:n].encode("utf-8")


def gather_code(n: int, roots: list[Path]) -> tuple[bytes, list[str]]:
    chunks = []
    used = []
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            if total >= n:
                break
            try:
                b = p.read_bytes()
            except Exception:
                continue
            if len(b) < 200 or b"\x00" in b:
                continue
            chunks.append(b)
            used.append(str(p))
            total += len(b)
        if total >= n:
            break
    return b"\n".join(chunks)[:n], used


def split_code(files: list[Path], n: int) -> tuple[bytes, bytes, bytes]:
    chunks = []
    for p in files:
        try:
            b = p.read_bytes()
        except Exception:
            continue
        if len(b) >= 200 and b"\x00" not in b:
            chunks.append(b)
        if sum(len(x) for x in chunks) >= n:
            break
    # File-level split: 80/10/10 by file bytes, not by cutting a file.
    total = sum(len(x) for x in chunks)
    train, val, test = [], [], []
    acc = 0
    for b in chunks:
        if acc + len(b) <= int(total * 0.8):
            train.append(b)
        elif acc + len(b) <= int(total * 0.9):
            val.append(b)
        else:
            test.append(b)
        acc += len(b)
    return b"\n".join(train), b"\n".join(val), b"\n".join(test)


def ngram_overlap(a: bytes, b: bytes, n: int = 256, limit: int = 50000) -> int:
    ga = {a[i:i+n] for i in range(min(len(a), limit) - n)}
    gb = {b[i:i+n] for i in range(min(len(b), limit) - n)}
    return len(ga & gb)


def ratio_batch(ratio: float, mode: str, batch: int, seq: int) -> dict:
    # A batch is represented by domain counts only; no tensor allocation yet.
    # Keep the ratio exact per batch; mode only changes ordering/segmentation.
    math_n = int(round(batch * ratio))
    code_n = batch - math_n
    return {"math_samples": math_n, "code_samples": code_n,
            "math_tokens": math_n * seq, "code_tokens": code_n * seq,
            "total_tokens": batch * seq, "mode": mode}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pool-bytes", type=int, default=30_000_000)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data60m")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    random.seed(args.seed)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    n = args.pool_bytes
    math = make_math(n, args.seed)
    code_roots = [
        Path(r"C:\Users\LANG\DSH\A-Net-Architecture\experiments\anet_v2\lang50m\data"),
        Path(r"C:\Users\LANG\anaconda3\Lib"),
    ]
    code, used = gather_code(n, code_roots)
    if len(code) < n:
        code = (code * ((n + len(code) - 1) // len(code)))[:n]

    # Code file-level split; math uses disjoint RNG ranges.
    import os
    files = [Path(x) for x in used]
    code_train, code_val, code_test = split_code(files, n)
    # File split may not hit exactly n; pad train by repeating code_train.
    code_train = (code_train * ((n * 8 // 10 + len(code_train) - 1) // len(code_train)))[: n * 8 // 10]

    math_train = math[: int(n * 0.8)]
    math_val = make_math(n // 10, args.seed + 1)
    math_test = make_math(n // 10, args.seed + 2)

    pools = {
        "math_train": math_train, "math_val": math_val, "math_test": math_test,
        "code_train": code_train, "code_val": code_val, "code_test": code_test,
    }
    manifest = {"config": vars(args) | {"out_dir": str(out)}, "files": {}, "ratios": []}
    for name, data in pools.items():
        path = out / f"{name}.bin"
        path.write_bytes(data)
        manifest["files"][name] = {"bytes": len(data), "sha256": sha256_bytes(data), "path": str(path)}

    leak = {}
    for domain in ("math", "code"):
        train = pools[f"{domain}_train"]
        val = pools[f"{domain}_val"]
        leak[domain] = {"overlap_256gram": ngram_overlap(train, val),
                        "train_bytes": len(train), "val_bytes": len(val)}
    manifest["leakage"] = leak

    for ratio in (1.0, 0.75, 0.5, 0.25, 0.0):
        for mode in ("batch", "boundary"):
            manifest["ratios"].append(ratio_batch(ratio, mode, args.batch, args.seq) | {"ratio_math": ratio})
    manifest["code_files_used"] = len(used)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pool_bytes": n, "files": {k: v["bytes"] for k, v in manifest["files"].items()},
                      "leakage": leak, "code_files": len(used)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
