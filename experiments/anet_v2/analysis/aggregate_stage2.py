"""Aggregate Stage 2 sharing / gradient-conflict results."""

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
MODEL_ORDER = ["independent", "anet_basic", "anet_share", "anet_adapter"]
LABELS = {
    "independent": "Independent",
    "anet_basic": "A-Net Basic",
    "anet_share": "A-Net Share",
    "anet_adapter": "A-Net Share+Adapter",
}
COLORS = {"independent": "#2F5597", "anet_basic": "#C55A11", "anet_share": "#3C9D8F", "anet_adapter": "#7F5AA2"}
MARKERS = {"independent": "o", "anet_basic": "^", "anet_share": "s", "anet_adapter": "D"}


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
        groups[(r["active_mode"], r["width"], r["correlation"], r["model"])].append(r)
    rows = []
    for (mode, width, corr, model), recs in sorted(groups.items()):
        cost = recs[0]["cost"]
        acc = [r["test"]["mean"]["accuracy"] for r in recs if r["test"]["mean"].get("accuracy") is not None]
        mse = [r["test"]["mean"]["mse"] for r in recs if r["test"]["mean"].get("mse") is not None]
        r2 = [r["test"]["mean"]["r2"] for r in recs if r["test"]["mean"].get("r2") is not None]
        task_acc = []
        task_mse = []
        for t in range(len(recs[0]["test"]["per_task"])):
            task_acc.append(mean_std([r["test"]["per_task"][t].get("accuracy") for r in recs if r["test"]["per_task"][t].get("accuracy") is not None])[0])
            task_mse.append(mean_std([r["test"]["per_task"][t].get("mse") for r in recs if r["test"]["per_task"][t].get("mse") is not None])[0])
        row = {
            "active_mode": mode,
            "width": width,
            "correlation": corr,
            "model": model,
            "model_label": LABELS.get(model, model),
            "seeds": len(recs),
            "unique_params": cost["unique_params"],
            "reference_params": cost["accounting"].get("reference_params"),
            "active_params": cost["accounting"].get("active_params_per_sample"),
            "theoretical_flops": cost["accounting"].get("theoretical_flops_per_sample"),
            "actual_executions": cost["accounting"].get("actual_executions_per_sample"),
            "adapter_params": cost.get("adapter_params_total", 0),
            "accuracy_mean": mean_std(acc)[0],
            "accuracy_std": mean_std(acc)[1],
            "mse_mean": mean_std(mse)[0],
            "r2_mean": mean_std(r2)[0],
            "train_seconds_mean": mean_std([r["train"]["seconds"] for r in recs])[0],
            "gpu_peak_gb": mean_std([r["train"].get("peak_allocated_bytes", 0) / 1024**3 for r in recs])[0],
            "task0_accuracy": task_acc[0],
            "task1_mse": task_mse[1],
            "task2_accuracy": task_acc[2],
        }
        audit = [r.get("gradient_audit", {}) for r in recs]
        shared = [a.get("mean_pairwise_cosine") for a in audit if a.get("shared")]
        conflict = [a.get("conflict_fraction") for a in audit if a.get("shared")]
        row["grad_cosine_mean"] = mean_std(shared)[0]
        row["grad_cosine_std"] = mean_std(shared)[1]
        row["grad_conflict_fraction"] = mean_std(conflict)[0]
        row["grad_norm_task0"] = mean_std([a.get("task_grad_norms", [None, None, None])[0] for a in audit if a.get("shared")])[0]

        for mode_name in ("single", "pair", "dense"):
            p50s, p95s, tputs = [], [], []
            for r in recs:
                inf = r.get("inference")
                if not inf or mode_name not in inf.get("batches", {}):
                    continue
                b1 = inf["batches"][mode_name].get("1")
                if b1:
                    p50s.append(b1["p50_us"])
                    p95s.append(b1["p95_us"])
                b1024 = inf["batches"][mode_name].get("1024")
                if b1024:
                    tputs.append(b1024["throughput_samples_s"])
            row[f"{mode_name}_p50_us_mean"] = mean_std(p50s)[0]
            row[f"{mode_name}_p95_us_mean"] = mean_std(p95s)[0]
            row[f"{mode_name}_throughput_mean"] = mean_std(tputs)[0]
        rows.append(row)
    return rows


def add_negative_transfer(rows):
    ref = {(r["width"], r["correlation"]): r for r in rows if r["active_mode"] == "pair" and r["model"] == "independent"}
    for row in rows:
        base = ref.get((row["width"], row["correlation"]))
        if not base or row["model"] == "independent":
            continue
        if row["accuracy_mean"] is not None and base["accuracy_mean"] is not None:
            row["accuracy_delta_vs_independent_pp"] = (row["accuracy_mean"] - base["accuracy_mean"]) * 100
        if row["mse_mean"] is not None and base["mse_mean"] is not None:
            row["mse_delta_vs_independent"] = row["mse_mean"] - base["mse_mean"]
        for key in ("unique_params", "active_params", "reference_params"):
            if base.get(key):
                row[f"{key}_ratio_vs_independent"] = row[key] / base[key]
        for key, label in (("task0_accuracy", "task0_accuracy_delta_pp"), ("task2_accuracy", "task2_accuracy_delta_pp")):
            if row.get(key) is not None and base.get(key) is not None:
                row[label] = (row[key] - base[key]) * 100
        if row.get("task1_mse") is not None and base.get("task1_mse") is not None:
            row["task1_mse_delta"] = row["task1_mse"] - base["task1_mse"]
    return rows


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plots(rows, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    widths = sorted({r["width"] for r in rows})
    modes = sorted({r["active_mode"] for r in rows})
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        for model in MODEL_ORDER:
            pts = sorted([r for r in rows if r["active_mode"] == mode and r["model"] == model and r["correlation"] == "medium"], key=lambda r: r["width"])
            if not pts:
                continue
            ax.errorbar([r["width"] for r in pts], [r["accuracy_mean"] * 100 for r in pts],
                        yerr=[r["accuracy_std"] * 100 for r in pts], marker=MARKERS[model],
                        color=COLORS[model], label=LABELS[model], capsize=2)
        ax.set_xscale("log", base=2)
        ax.set_title(f"active mode = {mode}")
        ax.set_xlabel("hidden width")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("test accuracy (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_vs_width.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in modes:
        for model in ("anet_share", "anet_adapter"):
            pts = sorted([r for r in rows if r["active_mode"] == mode and r["model"] == model and r["correlation"] == "medium"], key=lambda r: r["width"])
            if not pts:
                continue
            ax.plot([r["width"] for r in pts], [r["grad_cosine_mean"] for r in pts],
                    marker=MARKERS[model], color=COLORS[model], linestyle="-" if mode == "pair" else "--",
                    label=f"{LABELS[model]} ({mode})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width")
    ax.set_ylabel("mean pairwise gradient cosine")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "gradient_cosine_vs_width.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_md(path, rows, failures, total):
    lines = ["# A-Net v2 Stage 2 summary", "", f"Successful runs: {total}; failed: {len(failures)}.", ""]
    for mode in sorted({r["active_mode"] for r in rows}):
        lines += [f"## {mode}", "", "| correlation | width | model | unique params | active params | test acc | grad cos | conflict |", "|---|---:|---|---:|---:|---:|---:|---:|"]
        for r in sorted([r for r in rows if r["active_mode"] == mode], key=lambda r: (r["correlation"], r["width"], MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99)):
            acc = f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}" if r["accuracy_mean"] is not None else "n/a"
            cos = f"{r['grad_cosine_mean']:+.3f}" if r["grad_cosine_mean"] is not None else "n/a"
            conf = f"{r['grad_conflict_fraction']:.2f}" if r["grad_conflict_fraction"] is not None else "n/a"
            lines.append(f"| {r['correlation']} | {r['width']} | {r['model_label']} | {r['unique_params']:,} | {r['active_params']:,} | {acc} | {cos} | {conf} |")
        lines.append("")
    if failures:
        lines += ["## Failures", "", "| run_id | error |", "|---|---|"]
        lines += [f"| {f.get('run_id')} | {f.get('error','')} |" for f in failures]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage2")
    args = p.parse_args()
    records, failures = load(args.results_dir / "raw_runs.jsonl")
    rows = add_negative_transfer(aggregate(records))
    write_csv(args.results_dir / "summary.csv", rows)
    (args.results_dir / "summary.json").write_text(json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(args.results_dir / "summary.md", rows, failures, len(records))
    plots(rows, args.results_dir)
    print(json.dumps({"ok": len(records), "failed": len(failures), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
