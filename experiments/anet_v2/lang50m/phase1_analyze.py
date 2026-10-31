"""Aggregate Phase 1 pilot JSONL into CSV/Markdown."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "phase1_pilot"


def main() -> None:
    recs = [json.loads(l) for l in (RES / "raw_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["arch"], r["ratio_math"], r["mode"])].append(r)
    rows = []
    for (arch, ratio, mode), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2])):
        row = {
            "arch": arch,
            "ratio_math": ratio,
            "mode": mode,
            "seeds": len({r["seed"] for r in rs}),
            "bpc_math_mean": statistics.mean(r["val"]["math"]["bpc"] for r in rs),
            "bpc_code_mean": statistics.mean(r["val"]["code"]["bpc"] for r in rs),
            "acc_math_mean": statistics.mean(r["val"]["math"]["next_byte_accuracy"] for r in rs),
            "acc_code_mean": statistics.mean(r["val"]["code"]["next_byte_accuracy"] for r in rs),
            "gap_math_mean": statistics.mean(r["train_test_gap_bpc"]["math"] for r in rs),
            "gap_code_mean": statistics.mean(r["train_test_gap_bpc"]["code"] for r in rs),
            "train_s_mean": statistics.mean(r["train_seconds"] for r in rs),
            "peak_gb": statistics.mean(r["peak_allocated_bytes"] for r in rs) / 1024**3,
            "unique_params": rs[0]["params"]["unique_trainable"],
            "active_params": rs[0]["params"]["active_parameters_one_forward"],
        }
        row["balanced_bpc"] = (row["bpc_math_mean"] + row["bpc_code_mean"]) / 2
        rows.append(row)
    keys = list(rows[0].keys())
    with open(RES / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (RES / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Phase 1 pilot summary", "",
             "1 seed, 6M tokens/run. BPC lower is better; not a final capability claim.", "",
             "| arch | math ratio | mode | math BPC | code BPC | balanced | math acc | code acc | train s |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arch']} | {r['ratio_math']:.2f} | {r['mode']} | {r['bpc_math_mean']:.3f} | {r['bpc_code_mean']:.3f} | {r['balanced_bpc']:.3f} | {r['acc_math_mean']:.3f} | {r['acc_code_mean']:.3f} | {r['train_s_mean']:.0f} |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(recs), "groups": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
