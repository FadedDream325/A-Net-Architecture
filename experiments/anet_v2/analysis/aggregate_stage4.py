"""Aggregate Stage 4 factorial results and cache streams."""

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
MODELS = ["anet_basic", "anet_share", "anet_reuse", "anet_full"]
LABELS = {
    "anet_basic": "Basic (no share/cache)",
    "anet_share": "Share only",
    "anet_reuse": "Reuse only",
    "anet_full": "Share+Reuse",
}
COLORS = {"anet_basic": "#C55A11", "anet_share": "#3C9D8F", "anet_reuse": "#2F5597", "anet_full": "#7F5AA2"}
MARKERS = {"anet_basic": "^", "anet_share": "s", "anet_reuse": "o", "anet_full": "D"}


def mean_std(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load(path):
    ok, bad = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            (ok if rec.get("status") == "ok" else bad).append(rec)
    return ok, bad


def aggregate_training(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["width"], r["workflows"], r["correlation"], r["model"])].append(r)
    rows = []
    for (width, wf, corr, model), recs in sorted(groups.items()):
        cost = recs[0]["cost"]
        acc = [r["test"]["mean"]["accuracy"] for r in recs if r["test"]["mean"].get("accuracy") is not None]
        mse = [r["test"]["mean"]["mse"] for r in recs if r["test"]["mean"].get("mse") is not None]
        p50 = []
        thr = []
        for r in recs:
            inf = r.get("inference")
            if not inf:
                continue
            mode = "pair" if "pair" in inf["batches"] else "single"
            b1 = inf["batches"][mode].get("1")
            b256 = inf["batches"][mode].get("256")
            if b1:
                p50.append(b1["p50_us"])
            if b256:
                thr.append(b256["throughput_samples_s"])
        row = {
            "width": width,
            "workflows": wf,
            "correlation": corr,
            "model": model,
            "model_label": LABELS[model],
            "seeds": len(recs),
            "unique_params": cost["unique_params"],
            "active_params": cost["accounting"].get("active_params_per_sample"),
            "theoretical_flops": cost["accounting"].get("theoretical_flops_per_sample"),
            "accuracy_mean": mean_std(acc)[0],
            "accuracy_std": mean_std(acc)[1],
            "mse_mean": mean_std(mse)[0],
            "p50_us_mean": mean_std(p50)[0],
            "throughput_mean": mean_std(thr)[0],
            "train_seconds_mean": mean_std([r["train"]["seconds"] for r in recs])[0],
            "gpu_peak_gb": mean_std([r["train"].get("peak_allocated_bytes", 0) / 1024**3 for r in recs])[0],
        }
        rows.append(row)
    return rows


def factorial(rows):
    by = {(r["width"], r["workflows"], r["correlation"], r["model"]): r for r in rows}
    out = []
    for width in sorted({r["width"] for r in rows}):
        for wf in sorted({r["workflows"] for r in rows}):
            for corr in sorted({r["correlation"] for r in rows}):
                keys = [(width, wf, corr, m) for m in MODELS]
                if not all(k in by for k in keys):
                    continue
                b, s, r, f = [by[k] for k in keys]
                out.append({
                    "width": width,
                    "workflows": wf,
                    "correlation": corr,
                    "share_effect_acc_pp": (s["accuracy_mean"] - b["accuracy_mean"]) * 100 if s["accuracy_mean"] is not None and b["accuracy_mean"] is not None else None,
                    "reuse_effect_acc_pp": (r["accuracy_mean"] - b["accuracy_mean"]) * 100 if r["accuracy_mean"] is not None and b["accuracy_mean"] is not None else None,
                    "interaction_acc_pp": (f["accuracy_mean"] - s["accuracy_mean"] - r["accuracy_mean"] + b["accuracy_mean"]) * 100 if all(x["accuracy_mean"] is not None for x in (b, s, r, f)) else None,
                    "basic_unique": b["unique_params"],
                    "share_unique_ratio": s["unique_params"] / b["unique_params"],
                    "reuse_unique_ratio": r["unique_params"] / b["unique_params"],
                    "full_unique_ratio": f["unique_params"] / b["unique_params"],
                    "basic_p50": b["p50_us_mean"],
                    "share_p50_ratio": s["p50_us_mean"] / b["p50_us_mean"] if b["p50_us_mean"] and s["p50_us_mean"] else None,
                    "reuse_p50_ratio": r["p50_us_mean"] / b["p50_us_mean"] if b["p50_us_mean"] and r["p50_us_mean"] else None,
                    "full_p50_ratio": f["p50_us_mean"] / b["p50_us_mean"] if b["p50_us_mean"] and f["p50_us_mean"] else None,
                })
    return out


def aggregate_cache(records):
    streams = []
    for r in records:
        for s in r.get("cache_streams", []) or []:
            streams.append({
                "width": r["width"],
                "workflows": r["workflows"],
                "correlation": r["correlation"],
                "model": r["model"],
                "seed": r["seed"],
                "batch": s["batch"],
                "rate": s["rate"],
                "hit_rate": s["cache"]["hit_rate"],
                "execution_ratio": s["actual_executions"] / s["theoretical_executions"] if s["theoretical_executions"] else None,
                "p50_us": s["p50_us"],
                "throughput": s["throughput_samples_s"],
                "cache_bytes": s["cache"]["bytes"],
            })
    groups = defaultdict(list)
    for s in streams:
        groups[(s["width"], s["workflows"], s["model"], s["batch"], s["rate"])].append(s)
    rows = []
    for (width, wf, model, batch, rate), recs in sorted(groups.items()):
        rows.append({
            "width": width,
            "workflows": wf,
            "model": model,
            "batch": batch,
            "rate": rate,
            "n": len(recs),
            "hit_rate_mean": mean_std([r["hit_rate"] for r in recs])[0],
            "execution_ratio_mean": mean_std([r["execution_ratio"] for r in recs])[0],
            "p50_us_mean": mean_std([r["p50_us"] for r in recs])[0],
            "throughput_mean": mean_std([r["throughput"] for r in recs])[0],
            "cache_bytes_max": max(r["cache_bytes"] for r in recs),
        })
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


def plots(training, cache, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    widths = sorted({r["width"] for r in training})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, wf in zip(axes, sorted({r["workflows"] for r in training})):
        for model in MODELS:
            pts = sorted([r for r in training if r["workflows"] == wf and r["model"] == model and r["correlation"] == "medium"], key=lambda r: r["width"])
            if not pts:
                continue
            ax.errorbar([r["width"] for r in pts], [r["accuracy_mean"] * 100 for r in pts],
                        yerr=[r["accuracy_std"] * 100 for r in pts], marker=MARKERS[model],
                        color=COLORS[model], label=LABELS[model], capsize=2)
        ax.set_xscale("log", base=2)
        ax.set_title(f"workflows={wf}")
        ax.set_xlabel("hidden width")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("test accuracy (%)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_vs_width_by_workflows.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if cache:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, batch in zip(axes, sorted({r["batch"] for r in cache})):
            for model in ("anet_reuse", "anet_full"):
                pts = sorted([r for r in cache if r["model"] == model and r["batch"] == batch], key=lambda r: r["rate"])
                if not pts:
                    continue
                ax.plot([r["rate"] * 100 for r in pts], [r["hit_rate_mean"] for r in pts], marker=MARKERS[model], color=COLORS[model], label=LABELS[model])
            ax.set_title(f"cache batch={batch}")
            ax.set_xlabel("repetition rate (%)")
            ax.set_ylabel("hit rate")
            ax.grid(alpha=0.25)
        axes[-1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "cache_hit_rate.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_md(path, training, fact, cache, failures, total):
    lines = ["# A-Net v2 Stage 4 summary", "", f"Successful runs: {total}; failed: {len(failures)}.", ""]
    lines += ["## Factorial effects (medium correlation, pair mode)", "", "| width | W | share Δacc | reuse Δacc | interaction | share unique | full unique |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fact:
        if r["correlation"] != "medium":
            continue
        lines.append(f"| {r['width']} | {r['workflows']} | {r['share_effect_acc_pp']:+.2f} | {r['reuse_effect_acc_pp']:+.2f} | {r['interaction_acc_pp']:+.2f} | {r['share_unique_ratio']:.2f} | {r['full_unique_ratio']:.2f} |")
    lines += ["", "## Cache streams (selected)", "", "| width | W | model | batch | rate | hit | exec ratio | P50 us |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in cache:
        lines.append(f"| {r['width']} | {r['workflows']} | {r['model']} | {r['batch']} | {r['rate']:.2f} | {r['hit_rate_mean']:.2f} | {r['execution_ratio_mean']:.2f} | {r['p50_us_mean']:.1f} |")
    if failures:
        lines += ["", "## Failures", "", "| run_id | error |", "|---|---|"] + [f"| {f.get('run_id')} | {f.get('error','')} |" for f in failures]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "anet_v2" / "results" / "stage4")
    args = p.parse_args()
    records, failures = load(args.results_dir / "raw_runs.jsonl")
    training = aggregate_training(records)
    fact = factorial(training)
    cache = aggregate_cache(records)
    write_csv(args.results_dir / "summary.csv", training)
    write_csv(args.results_dir / "factorial.csv", fact)
    if cache:
        write_csv(args.results_dir / "cache_summary.csv", cache)
    (args.results_dir / "summary.json").write_text(json.dumps({"training": training, "factorial": fact, "cache": cache, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(args.results_dir / "summary.md", training, fact, cache, failures, len(records))
    plots(training, cache, args.results_dir)
    print(json.dumps({"ok": len(records), "failed": len(failures), "training_rows": len(training), "cache_rows": len(cache)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
