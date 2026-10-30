"""Aggregate Stage 1 raw records into CSV/JSON/Markdown and figures."""

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
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = ["independent", "shared", "anet_basic", "anet_full"]
LABELS = {
    "independent": "Independent",
    "shared": "Shared Backbone",
    "anet_basic": "A-Net Basic",
    "anet_full": "A-Net Full",
}
COLORS = {
    "independent": "#2F5597",
    "shared": "#3C9D8F",
    "anet_basic": "#C55A11",
    "anet_full": "#7F5AA2",
}
MARKERS = {"independent": "o", "shared": "s", "anet_basic": "^", "anet_full": "D"}


def mean_std(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return (None, None)
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


def median(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(values) if values else None


def percentile(values, q):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return None
    idx = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
    return values[idx]


def load_records(path: Path):
    records, failures = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            (records if rec.get("status") == "ok" else failures).append(rec)
    return records, failures


def aggregate(records):
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["width"], rec["correlation"], rec["model"])].append(rec)
    rows = []
    for (width, corr, model), recs in sorted(groups.items()):
        cost = recs[0]["cost"]
        acc = [r["test"]["mean"]["accuracy"] for r in recs if r["test"]["mean"].get("accuracy") is not None]
        f1 = [r["test"]["mean"]["f1"] for r in recs if r["test"]["mean"].get("f1") is not None]
        macro_f1 = [r["test"]["mean"]["macro_f1"] for r in recs if r["test"]["mean"].get("macro_f1") is not None]
        mse = [r["test"]["mean"]["mse"] for r in recs if r["test"]["mean"].get("mse") is not None]
        r2 = [r["test"]["mean"]["r2"] for r in recs if r["test"]["mean"].get("r2") is not None]
        train = [r["train"]["seconds"] for r in recs]
        peak_gpu = [r["train"].get("peak_allocated_bytes") for r in recs if r["train"].get("peak_allocated_bytes")]
        peak_rss = [r["train"].get("peak_rss_bytes") for r in recs if r["train"].get("peak_rss_bytes")]
        row = {
            "width": width,
            "correlation": corr,
            "model": model,
            "model_label": LABELS.get(model, model),
            "seeds": len(recs),
            "unique_params": cost["unique_params"],
            "active_params": cost["accounting"].get("active_params_per_sample"),
            "reference_params": cost["accounting"].get("reference_params"),
            "theoretical_flops": cost["accounting"].get("theoretical_flops_per_sample"),
            "actual_flops": cost["accounting"].get("actual_flops_per_sample"),
            "actual_executions": cost["accounting"].get("actual_executions_per_sample"),
            "theoretical_executions": cost["accounting"].get("theoretical_executions_per_sample"),
            "adapter_params": cost.get("adapter_params_total", 0),
            "accuracy_mean": mean_std(acc)[0],
            "accuracy_std": mean_std(acc)[1],
            "accuracy_median": median(acc),
            "f1_mean": mean_std(f1)[0],
            "macro_f1_mean": mean_std(macro_f1)[0],
            "mse_mean": mean_std(mse)[0],
            "r2_mean": mean_std(r2)[0],
            "train_seconds_mean": mean_std(train)[0],
            "train_seconds_std": mean_std(train)[1],
            "train_peak_gpu_gb": mean_std([x / 1024**3 for x in peak_gpu])[0],
            "train_peak_rss_gb": mean_std([x / 1024**3 for x in peak_rss])[0],
        }
        h2d = [r.get("inference", {}).get("host_to_device_bytes", 0) for r in recs if r.get("inference")]
        row["host_to_device_bytes_max"] = max(h2d) if h2d else None
        row["input_device"] = next(
            (r["inference"]["batches"]["single"]["1"].get("input_device") for r in recs if r.get("inference") and "single" in r["inference"].get("batches", {}) and "1" in r["inference"]["batches"]["single"]),
            None,
        )
        # inference per mode/batch
        for mode in ("single", "pair", "mixed"):
            p50s, p95s, tputs, hits = [], [], [], []
            for r in recs:
                inf = r.get("inference")
                if not inf or mode not in inf.get("batches", {}):
                    continue
                batch = inf["batches"][mode].get("1")
                if batch:
                    p50s.append(batch["p50_us"])
                    p95s.append(batch["p95_us"])
                b1024 = inf["batches"][mode].get("1024")
                if b1024:
                    tputs.append(b1024["throughput_samples_s"])
                cache = inf["batches"][mode].get("1", {}).get("cache")
                if cache:
                    hits.append(cache.get("hit_rate"))
            row[f"{mode}_batch1_p50_us_mean"] = mean_std(p50s)[0]
            row[f"{mode}_batch1_p50_us_std"] = mean_std(p50s)[1]
            row[f"{mode}_batch1_p95_us_mean"] = mean_std(p95s)[0]
            row[f"{mode}_batch1_p95_us_median"] = median(p95s)
            row[f"{mode}_batch1024_throughput_mean"] = mean_std(tputs)[0]
            row[f"{mode}_batch1024_throughput_std"] = mean_std(tputs)[1]
            row[f"{mode}_cache_hit_rate"] = mean_std(hits)[0]
        rows.append(row)
    return rows


def add_ratios(rows):
    by = {(r["width"], r["correlation"], r["model"]): r for r in rows}
    for row in rows:
        base = by.get((row["width"], row["correlation"], "independent"))
        if not base:
            continue
        for key in ("unique_params", "active_params", "theoretical_flops"):
            if base.get(key):
                row[f"{key}_ratio_vs_independent"] = row[key] / base[key]
        for key in ("single_batch1_p50_us_mean", "single_batch1024_throughput_mean"):
            if base.get(key):
                row[f"{key}_ratio_vs_independent"] = row[key] / base[key]
    return rows


def write_csv(path: Path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latency_csv(path: Path, records):
    rows = []
    for rec in records:
        inf = rec.get("inference")
        if not inf:
            continue
        for mode, batches in inf["batches"].items():
            for batch, data in batches.items():
                rows.append({
                    "run_id": rec["run_id"],
                    "width": rec["width"],
                    "correlation": rec["correlation"],
                    "model": rec["model"],
                    "active_mode": rec.get("active_mode"),
                    "mode": mode,
                    "batch": batch,
                    "n": data.get("n"),
                    "p50_us": data.get("p50_us"),
                    "p95_us": data.get("p95_us"),
                    "mean_us": data.get("mean_us"),
                    "std_us": data.get("std_us"),
                    "throughput_samples_s": data.get("throughput_samples_s"),
                    "cache_hit_rate": (data.get("cache") or {}).get("hit_rate"),
                })
    if rows:
        write_csv(path, rows)


def make_plots(rows, out_dir: Path):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    widths = sorted({r["width"] for r in rows})
    corrs = ["low", "medium", "high"]
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, corr in zip(axes, corrs):
        for model in models:
            pts = sorted([r for r in rows if r["correlation"] == corr and r["model"] == model], key=lambda r: r["width"])
            if not pts:
                continue
            ax.errorbar(
                [r["width"] for r in pts], [r["accuracy_mean"] * 100 for r in pts],
                yerr=[r["accuracy_std"] * 100 for r in pts],
                marker=MARKERS[model], color=COLORS[model], label=LABELS[model], capsize=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_title(f"correlation = {corr}")
        ax.set_xlabel("hidden width")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("test accuracy (%)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_vs_width.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for corr in corrs:
        for model in models:
            pts = sorted([r for r in rows if r["correlation"] == corr and r["model"] == model], key=lambda r: r["width"])
            if not pts:
                continue
            ax.plot([r["width"] for r in pts], [r["active_params"] for r in pts], marker=MARKERS[model],
                    color=COLORS[model], alpha=0.45 if corr != "medium" else 1.0,
                    linestyle="--" if corr == "low" else (":" if corr == "high" else "-"),
                    label=f"{LABELS[model]} ({corr})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("hidden width")
    ax.set_ylabel("active parameters / sample")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "active_params_vs_width.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model in models:
        pts = sorted([r for r in rows if r["correlation"] == "medium" and r["model"] == model], key=lambda r: r["width"])
        if not pts:
            continue
        ax.plot([r["width"] for r in pts], [r["single_batch1_p50_us_mean"] for r in pts], marker=MARKERS[model], color=COLORS[model], label=LABELS[model])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("hidden width")
    ax.set_ylabel("batch=1 P50 (us)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "batch1_p50_vs_width.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model in models:
        pts = sorted([r for r in rows if r["correlation"] == "medium" and r["model"] == model], key=lambda r: r["width"])
        if not pts:
            continue
        ax.plot([r["width"] for r in pts], [r["single_batch1024_throughput_mean"] for r in pts], marker=MARKERS[model], color=COLORS[model], label=LABELS[model])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("hidden width")
    ax.set_ylabel("batch=1024 throughput (samples/s)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "throughput1024_vs_width.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, rows, failures, total):
    lines = [
        "# A-Net v2 Stage 1 summary",
        "",
        f"Successful runs: {total}; failed: {len(failures)}.",
        "",
        "## Medium correlation",
        "",
        "| width | model | unique params | active params | test acc | train s | b1 P50 us | b1024 samples/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted([r for r in rows if r["correlation"] == "medium"], key=lambda r: (r["width"], MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99)):
        acc = f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}" if r["accuracy_mean"] is not None else "n/a"
        lines.append(
            f"| {r['width']} | {r['model_label']} | {r['unique_params']:,} | {r['active_params']:,} | "
            f"{acc} | {r['train_seconds_mean']:.2f} | {r['single_batch1_p50_us_mean']:.1f} | "
            f"{r['single_batch1024_throughput_mean']:.0f} |"
        )
    lines += ["", "## Correlations (accuracy, mean ± std)", "", "| width | model | low | medium | high |", "|---:|---|---:|---:|---:|"]
    by = defaultdict(dict)
    for r in rows:
        by[(r["width"], r["model"])][r["correlation"]] = r
    for (width, model), vals in sorted(by.items(), key=lambda kv: (kv[0][0], MODEL_ORDER.index(kv[0][1]) if kv[0][1] in MODEL_ORDER else 99)):
        cells = []
        for corr in ("low", "medium", "high"):
            r = vals.get(corr)
            cells.append(f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}" if r and r["accuracy_mean"] is not None else "n/a")
        lines.append(f"| {width} | {LABELS.get(model, model)} | " + " | ".join(cells) + " |")
    if failures:
        lines += ["", "## Failures", "", "| run_id | error |", "|---|---|"]
        for f in failures:
            lines.append(f"| {f.get('run_id')} | {f.get('error','')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage1")
    args = p.parse_args()
    records, failures = load_records(args.results_dir / "raw_runs.jsonl")
    rows = add_ratios(aggregate(records))
    write_csv(args.results_dir / "summary.csv", rows)
    (args.results_dir / "summary.json").write_text(json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latency_csv(args.results_dir / "latency_summary.csv", records)
    write_markdown(args.results_dir / "summary.md", rows, failures, len(records))
    make_plots(rows, args.results_dir)
    print(json.dumps({"ok": len(records), "failed": len(failures), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
