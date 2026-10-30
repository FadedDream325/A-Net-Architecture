"""Aggregate Stage 3 cache benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = ["anet_reuse", "anet_full"]
LABELS = {"anet_reuse": "A-Net Reuse (no share)", "anet_full": "A-Net Full (share+cache)"}
COLORS = {"anet_reuse": "#C55A11", "anet_full": "#7F5AA2"}
MARKERS = {"anet_reuse": "o", "anet_full": "D"}


def mean_std(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load(path):
    ok, bad = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                (ok if rec.get("status") == "ok" else bad).append(rec)
    return ok, bad


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["width"], r["model"], r["batch"], r["rate"], r["cache_enabled"])].append(r)
    rows = []
    for (width, model, batch, rate, cache_enabled), recs in sorted(groups.items()):
        hits = [r["cache"]["hit_rate"] for r in recs]
        p50 = [r["p50_us"] for r in recs]
        p95 = [r["p95_us"] for r in recs]
        thr = [r["throughput_samples_s"] for r in recs]
        exec_ratio = [r["actual_executions"] / r["theoretical_executions"] for r in recs if r["theoretical_executions"]]
        rows.append({
            "width": width,
            "model": model,
            "model_label": LABELS[model],
            "batch": batch,
            "rate": rate,
            "cache_enabled": cache_enabled,
            "seeds": len(recs),
            "hit_rate_mean": mean_std(hits)[0],
            "hit_rate_std": mean_std(hits)[1],
            "p50_us_mean": mean_std(p50)[0],
            "p50_us_std": mean_std(p50)[1],
            "p95_us_mean": mean_std(p95)[0],
            "throughput_mean": mean_std(thr)[0],
            "execution_ratio_mean": mean_std(exec_ratio)[0],
            "actual_executions_mean": mean_std([r["actual_executions"] for r in recs])[0],
            "theoretical_executions_mean": mean_std([r["theoretical_executions"] for r in recs])[0],
            "cache_bytes_mean": mean_std([r["cache"]["bytes"] for r in recs])[0],
            "lookup_us_mean": mean_std([r["cache"]["lookup_us_mean"] for r in recs])[0],
            "hash_us_mean": mean_std([r["cache"]["hash_us"] for r in recs])[0],
            "equivalence_max_abs": max(r["output_equivalence_max_abs"] for r in recs),
            "invalidation_all_ok": all(r["cache_invalidation_ok"] for r in recs),
            "host_to_device_max": max(r["host_to_device_bytes"] for r in recs),
        })
    return rows


def add_paired_ratios(rows):
    by = {(r["width"], r["model"], r["batch"], r["rate"], r["cache_enabled"]): r for r in rows}
    for row in rows:
        if not row["cache_enabled"]:
            continue
        base = by.get((row["width"], row["model"], row["batch"], row["rate"], False))
        if not base:
            continue
        for key in ("p50_us_mean", "p95_us_mean", "throughput_mean"):
            if base.get(key):
                row[f"{key}_ratio_vs_cache_off"] = row[key] / base[key]
        if base.get("execution_ratio_mean") and row.get("execution_ratio_mean") is not None:
            row["execution_reduction_vs_cache_off"] = 1.0 - row["execution_ratio_mean"] / base["execution_ratio_mean"]
    return rows


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plots(rows, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    rates = sorted({r["rate"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, batch in zip(axes, sorted({r["batch"] for r in rows})):
        for model in MODEL_ORDER:
            pts = sorted([r for r in rows if r["model"] == model and r["cache_enabled"] and r["batch"] == batch and r["width"] == 512], key=lambda r: r["rate"])
            if not pts:
                continue
            ax.plot([r["rate"] * 100 for r in pts], [r["hit_rate_mean"] for r in pts], marker=MARKERS[model], color=COLORS[model], label=LABELS[model])
        ax.set_title(f"batch={batch}")
        ax.set_xlabel("input repetition rate (%)")
        ax.set_ylabel("cache hit rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "hit_rate_vs_repetition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, batch in zip(axes, sorted({r["batch"] for r in rows})):
        for model in MODEL_ORDER:
            pts = sorted([r for r in rows if r["model"] == model and r["cache_enabled"] and r["batch"] == batch and r["width"] == 512], key=lambda r: r["rate"])
            if not pts:
                continue
            ax.plot([r["rate"] * 100 for r in pts], [r.get("p50_us_mean_ratio_vs_cache_off") for r in pts], marker=MARKERS[model], color=COLORS[model], label=LABELS[model])
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_title(f"batch={batch}")
        ax.set_xlabel("input repetition rate (%)")
        ax.set_ylabel("P50 ratio cache-on / cache-off")
        ax.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_ratio_vs_repetition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_md(path, rows, failures, total):
    lines = ["# A-Net v2 Stage 3 summary", "", f"Successful runs: {total}; failed: {len(failures)}.", ""]
    for batch in sorted({r["batch"] for r in rows}):
        lines += [f"## batch={batch}", "", "| width | model | rate | cache | hit rate | exec ratio | P50 us | P50 ratio | throughput |", "|---:|---|---:|---|---:|---:|---:|---:|---:|"]
        for r in sorted([r for r in rows if r["batch"] == batch], key=lambda r: (r["width"], MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99, r["rate"], r["cache_enabled"])):
            ratio = r.get("p50_us_mean_ratio_vs_cache_off")
            lines.append(
                f"| {r['width']} | {r['model_label']} | {r['rate']:.2f} | {r['cache_enabled']} | "
                f"{r['hit_rate_mean']:.2f} | {r['execution_ratio_mean']:.2f} | {r['p50_us_mean']:.1f} | "
                f"{'n/a' if ratio is None else f'{ratio:.2f}'} | {r['throughput_mean']:.0f} |"
            )
        lines.append("")
    if failures:
        lines += ["## Failures", "", "| run_id | error |", "|---|---|"] + [f"| {f.get('run_id')} | {f.get('error','')} |" for f in failures]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage3")
    args = p.parse_args()
    records, failures = load(args.results_dir / "raw_bench.jsonl")
    rows = add_paired_ratios(aggregate(records))
    write_csv(args.results_dir / "summary.csv", rows)
    (args.results_dir / "summary.json").write_text(json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(args.results_dir / "summary.md", rows, failures, len(records))
    plots(rows, args.results_dir)
    print(json.dumps({"ok": len(records), "failed": len(failures), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
