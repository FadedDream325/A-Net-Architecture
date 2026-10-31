"""Minimal Phase 1 pure-domain evaluation: math Exact Match and code syntax Pass@1.

- Math: parse Q/A examples, prompt up to "A: ", greedily generate the answer,
  compare parsed integer exactly with the target.
- Code: take a syntactically valid held-out code prefix, greedily generate a
  short continuation, and count Pass@1 if ast.parse(prefix + continuation)
  succeeds.  This is syntax-only, not execution/unit tests.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data60m"
CKPT = ROOT / "results" / "phase1_pure" / "checkpoints"
MATH_RE = re.compile(r"Q: ([0-9()+\-* ]+) =\nA: (-?\d+)\n")


def load_model(arch: str, ckpt: Path, device):
    from experiments.anet_v2.lang50m.phase1_train import make_model
    model = make_model(arch).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def generate(model, prompt: str, task: int, device, max_new: int = 16, stop_newline: bool = True) -> str:
    ids = list(prompt.encode("utf-8"))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x, task)[:, -1, :]
            nxt = int(logits.argmax(dim=-1).item())
            if stop_newline and nxt == 10:
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return bytes(out).decode("utf-8", errors="ignore")


def math_examples(n: int, seed: int):
    text = (DATA / "math_test.bin").read_bytes().decode("utf-8", errors="ignore")
    examples = MATH_RE.findall(text)
    rng = random.Random(seed)
    rng.shuffle(examples)
    return [(expr.strip(), int(ans)) for expr, ans in examples[:n]]


def code_prompts(n: int, seed: int):
    text = (DATA / "code_test.bin").read_bytes().decode("utf-8", errors="ignore")
    lines = text.splitlines()
    rng = random.Random(seed)
    starts = [i for i in range(0, max(0, len(lines) - 8)) if lines[i] and not lines[i][0].isspace()]
    rng.shuffle(starts)
    out = []
    for i in starts:
        block = "\n".join(lines[i:i+6]).strip("\n")
        if not 30 <= len(block) <= 400:
            continue
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        out.append(block + "\n")
        if len(out) >= n:
            break
    return out


def evaluate_math(model, n: int, seed: int, device):
    examples = math_examples(n, seed)
    correct = 0
    details = []
    for expr, target in examples:
        prompt = f"Q: {expr} =\nA: "
        raw = generate(model, prompt, 0, device, max_new=12)
        got = None
        try:
            got = int(raw.strip().split()[0])
        except Exception:
            pass
        ok = got == target
        correct += int(ok)
        details.append({"expr": expr, "target": target, "generated": raw, "parsed": got, "correct": ok})
    return {"n": len(examples), "exact_match": correct / max(1, len(examples)), "details": details[:20]}


def evaluate_code(model, n: int, seed: int, device):
    prompts = code_prompts(n, seed)
    passed = 0
    details = []
    for prompt in prompts:
        cont = generate(model, prompt, 1, device, max_new=24)
        full = prompt + cont
        try:
            ast.parse(full)
            ok = bool(cont.strip())  # empty continuation is not a generated pass
        except SyntaxError:
            ok = False
        passed += int(ok)
        details.append({"prompt": prompt[:80], "continuation": cont, "pass": ok})
    return {"n": len(prompts), "pass_at_1": passed / max(1, len(prompts)), "details": details[:20]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "phase1_pure" / "eval.json")
    args = p.parse_args()
    device = torch.device("cuda")
    out = []
    for arch in ("transformer12", "shared_depth"):
        for seed in args.seeds:
            for ratio, domain in ((1.0, "math"), (0.0, "code")):
                ckpt = CKPT / f"{arch}__s{seed}__m{ratio}__batch.pt"
                if not ckpt.exists():
                    print("missing", ckpt, flush=True)
                    continue
                model = load_model(arch, ckpt, device)
                if domain == "math":
                    metrics = evaluate_math(model, args.n, seed, device)
                else:
                    metrics = evaluate_code(model, args.n, seed, device)
                rec = {"arch": arch, "seed": seed, "domain": domain, **metrics}
                out.append(rec)
                key = "exact_match" if domain == "math" else "pass_at_1"
                print(arch, seed, domain, key, round(metrics[key], 3), "n", metrics["n"], flush=True)
                del model
                torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for domain in ("math", "code"):
        vals = [r["exact_match"] if domain == "math" else r["pass_at_1"] for r in out if r["domain"] == domain]
        if vals:
            print(domain, "mean", round(statistics.mean(vals), 4), "std", round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
