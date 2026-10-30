"""Aggregate raw scaling runs into summary tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import MODEL_LABELS, MODEL_KEYS
from .scales import SCALES

ROOT = Path(__file__).resolve().parents[2]
MODEL_ORDER = list(MODEL_KEYS)
COLORS = {
    "independent": "#2F5597",
    "shared": "#3C9D8F",
    "anet": "#C55A11",
    "cross": "#7F5AA2",
    "cross_off": "#9B8AAE",
}
MARKERS = {
    "independent": "o",
    "shared": "s",
    "anet": "^",
    "cross": "D",
    "cross_off": "x",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def load_records(results_dir: Path) -> tuple[list[dict], list[dict]]:
    raw = results_dir / "raw_runs.jsonl"
    if not raw.exists():
        raise SystemExit(f"missing {raw}")
    records: list[dict] = []
    failures: list[dict] = []
    with open(raw, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                records.append(rec)
            else:
                failures.append(rec)
    return records, failures


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        groups[(rec["scale"], rec["model"])].append(rec)

    rows: list[dict] = []
    for scale_key in SCALES:
        for model_key in MODEL_ORDER:
            recs = groups.get((scale_key, model_key), [])
            if not recs:
                continue
            base = recs[0]
            cost = base["cost"]
            acc_per_task = [
                [r["metrics"]["accuracy_per_task"][i] * 100 for r in recs]
                for i in range(3)
            ]
            means = [mean_std(v)[0] for v in acc_per_task]
            stds = [mean_std(v)[1] for v in acc_per_task]
            avg_vals = [r["metrics"]["accuracy_mean"] * 100 for r in recs]
            avg_mean, avg_std = mean_std(avg_vals)
            train_times = [r["train"]["seconds"] for r in recs]
            ckpt_mb = [r["checkpoint_bytes"] / 1e6 for r in recs if r.get("checkpoint_bytes")]
            peak_train_gb = [
                r["train"].get("peak_allocated_bytes", 0) / 1024**3 for r in recs
            ]
            batch1 = [
                r["inference"]["batches"].get("1", {}).get("p50_us")
                for r in recs
                if r.get("inference") and r["inference"]["batches"].get("1")
            ]
            throughput_256 = [
                r["inference"]["batches"].get("256", {}).get("throughput_samples_s")
                for r in recs
                if r.get("inference") and r["inference"]["batches"].get("256")
            ]
            throughput_1024 = [
                r["inference"]["batches"].get("1024", {}).get("throughput_samples_s")
                for r in recs
                if r.get("inference") and r["inference"]["batches"].get("1024")
            ]
            first_call = [
                r["inference"].get("first_call_after_training_us")
                for r in recs
                if r.get("inference") and r["inference"].get("first_call_after_training_us")
            ]
            row = {
                "scale": scale_key,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "seeds": len(recs),
                "hidden_dim": cost["hidden_dim"],
                "expert_hidden_dim": cost["expert_hidden_dim"],
                "total_params": cost["unique_trainable_parameters"],
                "active_params": cost["active_parameters_per_sample"],
                "macs": cost["macs_per_sample"],
                "flops": cost["flops_per_sample"],
                "active_fraction": cost["active_parameter_fraction"],
                "macs_fraction_vs_independent": cost["macs_fraction_vs_independent"],
                "accuracy_task1_mean": means[0],
                "accuracy_task1_std": stds[0],
                "accuracy_task2_mean": means[1],
                "accuracy_task2_std": stds[1],
                "accuracy_task3_mean": means[2],
                "accuracy_task3_std": stds[2],
                "accuracy_mean": avg_mean,
                "accuracy_std": avg_std,
                "checkpoint_mb": mean_std(ckpt_mb)[0],
                "train_seconds": mean_std(train_times)[0],
                "peak_train_gb": mean_std(peak_train_gb)[0],
                "batch1_p50_us": mean_std(batch1)[0],
                "first_call_us": mean_std(first_call)[0],
                "throughput_256": mean_std(throughput_256)[0],
                "throughput_1024": mean_std(throughput_1024)[0],
            }
            rows.append(row)
    return rows


def label_params(value: float) -> str:
    if value >= 1e6:
        return f"{value/1e6:.2f}M"
    if value >= 1e3:
        return f"{value/1e3:.0f}k"
    return f"{value:.0f}"


def make_plots(rows: list[dict], out_dir: Path) -> list[Path]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    paths: list[Path] = []
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    x_scales = list(SCALES.keys())

    # Accuracy vs total params
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model in models:
        pts = sorted([r for r in rows if r["model"] == model], key=lambda r: r["total_params"])
        ax.errorbar(
            [r["total_params"] for r in pts],
            [r["accuracy_mean"] for r in pts],
            yerr=[r["accuracy_std"] for r in pts],
            marker=MARKERS[model], linewidth=2, capsize=3,
            label=MODEL_LABELS[model], color=COLORS[model],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total trainable parameters (log scale)")
    ax.set_ylabel("Mean test accuracy (%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = fig_dir / "accuracy_vs_total_params.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Accuracy vs active MACs
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model in models:
        pts = sorted([r for r in rows if r["model"] == model], key=lambda r: r["macs"])
        ax.errorbar(
            [r["macs"] for r in pts],
            [r["accuracy_mean"] for r in pts],
            yerr=[r["accuracy_std"] for r in pts],
            marker=MARKERS[model], linewidth=2, capsize=3,
            label=MODEL_LABELS[model], color=COLORS[model],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Active MACs per sample (log scale)")
    ax.set_ylabel("Mean test accuracy (%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = fig_dir / "accuracy_vs_active_macs.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Throughput at batch 1024 vs active MACs
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model in models:
        pts = sorted([r for r in rows if r["model"] == model and r["throughput_1024"]], key=lambda r: r["macs"])
        if not pts:
            continue
        ax.plot(
            [r["macs"] for r in pts],
            [r["throughput_1024"] for r in pts],
            marker=MARKERS[model], linewidth=2,
            label=MODEL_LABELS[model], color=COLORS[model],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Active MACs per sample (log scale)")
    ax.set_ylabel("Throughput at batch=1024 (samples/s, log)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = fig_dir / "throughput_vs_active_macs.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Peak training memory vs total params
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model in models:
        pts = sorted([r for r in rows if r["model"] == model], key=lambda r: r["total_params"])
        ax.plot(
            [r["total_params"] for r in pts],
            [r["peak_train_gb"] for r in pts],
            marker=MARKERS[model], linewidth=2,
            label=MODEL_LABELS[model], color=COLORS[model],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total trainable parameters (log scale)")
    ax.set_ylabel("Peak training GPU memory (GiB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = fig_dir / "train_memory_vs_params.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def write_outputs(rows: list[dict], failures: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    fields = [
        "scale", "model", "model_label", "seeds", "hidden_dim", "expert_hidden_dim",
        "total_params", "active_params", "macs", "flops", "active_fraction",
        "macs_fraction_vs_independent", "accuracy_task1_mean", "accuracy_task1_std",
        "accuracy_task2_mean", "accuracy_task2_std", "accuracy_task3_mean",
        "accuracy_task3_std", "accuracy_mean", "accuracy_std", "checkpoint_mb",
        "train_seconds", "peak_train_gb", "batch1_p50_us", "first_call_us",
        "throughput_256", "throughput_1024",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    (out_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Scaling summary",
        "",
        "| scale | model | params | active | MACs | acc mean ± std | acc T1 | acc T2 | acc T3 | train s | peak GB | b1 µs | b1024 samples/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scale} | {model_label} | {total_params:,} | {active_params:,} | {macs:,} | "
            "{accuracy_mean:.2f} ± {accuracy_std:.2f} | {accuracy_task1_mean:.2f} | "
            "{accuracy_task2_mean:.2f} | {accuracy_task3_mean:.2f} | "
            "{train_seconds:.2f} | {peak_train_gb:.3f} | {batch1_p50_us:.1f} | "
            "{throughput_1024:.0f} |".format(**row)
        )
    lines.append("")
    lines.append(f"Failed runs: {len(failures)}")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "scaling" / "results" / "scaling_v1")
    args = parser.parse_args()
    records, failures = load_records(args.results_dir)
    rows = aggregate(records)
    write_outputs(rows, failures, args.results_dir)
    paths = make_plots(rows, args.results_dir)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
