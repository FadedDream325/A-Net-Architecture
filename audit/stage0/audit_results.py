"""Stage-0 audit of the legacy A-Net scaling results.

Read-only with respect to the legacy result files. Writes only under audit/stage0.
Uses the Python standard library so it can run without PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "scaling" / "results"


def analytic(kind: str, h: int, e: int) -> dict:
    if kind == "independent":
        params = 3 * (h * h + 6 * h + 2)
        active = h * h + 6 * h + 2
        macs = h * h + 4 * h
    elif kind == "shared":
        params = h * h + 10 * h + 6
        active = params
        macs = h * h + 4 * h
    elif kind == "anet":
        params = 3 * h + 3 * h * e + 9 * e + 6
        # active = L1 + one expert
        active = 3 * h + (h * e + 3 * e + 2)
        macs = 2 * h + h * e + 2 * e
    elif kind in {"cross", "cross_off"}:
        params = 3 * h + 3 * h * e + 9 * e + 6 + e * e + e
        active = 3 * h + (h * e + 3 * e + 2)
        macs = 2 * h + h * e + 2 * e
        if kind == "cross":
            active += e * e + e
            macs += e * e
    else:
        raise KeyError(kind)
    return {
        "unique_trainable_parameters": params,
        "active_parameters_per_sample": active,
        "macs_per_sample": macs,
        "flops_per_sample": 2 * macs,
    }


def mean_std(values):
    if not values:
        return (float("nan"), float("nan"))
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def audit_dataset(name: str) -> dict:
    base = RESULTS / name
    raw_path = base / "raw_runs.jsonl"
    records = load_jsonl(raw_path)
    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]
    issues = []

    run_ids = [r.get("run_id") for r in ok]
    if len(run_ids) != len(set(run_ids)):
        issues.append("duplicate successful run_id")

    formulas_checked = 0
    for r in ok:
        h = int(r["scale_config"]["hidden_dim"])
        e = int(r["scale_config"]["expert_hidden_dim"])
        exp = analytic(r["model"], h, e)
        for key in ("unique_trainable_parameters", "active_parameters_per_sample",
                    "macs_per_sample", "flops_per_sample"):
            actual = r["cost"].get(key)
            if actual != exp[key]:
                issues.append(
                    f"{r['run_id']}: cost.{key} actual={actual} expected={exp[key]}"
                )
        formulas_checked += 1

        inf = r.get("inference")
        if not inf:
            issues.append(f"{r['run_id']}: missing inference")
            continue
        for b, d in inf["batches"].items():
            if not (0 < d["p50_us"] <= d["p95_us"]):
                issues.append(f"{r['run_id']}: bad latency batch={b} {d}")
            if not (d["throughput_samples_s"] > 0):
                issues.append(f"{r['run_id']}: bad throughput batch={b}")

        acc = r["metrics"]["accuracy_per_task"]
        if not all(0.0 <= a <= 1.0 for a in acc):
            issues.append(f"{r['run_id']}: accuracy outside [0,1]")
        if abs(sum(acc) / len(acc) - r["metrics"]["accuracy_mean"]) > 1e-12:
            issues.append(f"{r['run_id']}: accuracy_mean mismatch")

        ckpt = r.get("checkpoint_bytes")
        params = r["cost"]["unique_trainable_parameters"]
        if ckpt is not None and not (0.99 * params * 4 <= ckpt <= 1.01 * params * 4 + 65536):
            issues.append(
                f"{r['run_id']}: checkpoint bytes {ckpt} vs fp32 params {params*4}"
            )

    groups = defaultdict(list)
    for r in ok:
        groups[(r["scale"], r["model"])].append(r)

    summary_path = base / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    summary_rows = {(row["scale"], row["model"]): row for row in summary["rows"]} if summary else {}
    summary_issues = []
    group_rows = []
    for (scale, model), recs in sorted(groups.items()):
        acc = [r["metrics"]["accuracy_mean"] * 100 for r in recs]
        m, s = mean_std(acc)
        b1 = [r["inference"]["batches"]["1"]["p50_us"] for r in recs]
        b1p95 = [r["inference"]["batches"]["1"]["p95_us"] for r in recs]
        t1024 = [r["inference"]["batches"]["1024"]["throughput_samples_s"] for r in recs]
        train = [r["train"]["seconds"] for r in recs]
        row = {
            "scale": scale,
            "model": model,
            "seeds": len(recs),
            "accuracy_mean": m,
            "accuracy_std": s,
            "batch1_p50_us_mean": statistics.mean(b1),
            "batch1_p50_us_stdev": statistics.stdev(b1) if len(b1) > 1 else 0.0,
            "batch1_p95_us_mean": statistics.mean(b1p95),
            "throughput_1024_mean": statistics.mean(t1024),
            "train_seconds_mean": statistics.mean(train),
            "train_seconds_cv": (
                statistics.stdev(train) / statistics.mean(train) if len(train) > 1 else 0.0
            ),
        }
        group_rows.append(row)
        if (scale, model) in summary_rows:
            ref = summary_rows[(scale, model)]
            if abs(ref["accuracy_mean"] - m) > 1e-9:
                summary_issues.append(f"{scale}/{model}: accuracy_mean differs from summary.json")
            if abs(ref["accuracy_std"] - s) > 1e-9:
                summary_issues.append(f"{scale}/{model}: accuracy_std differs from summary.json")
            if abs(ref["batch1_p50_us"] - statistics.mean(b1)) > 1e-9:
                summary_issues.append(f"{scale}/{model}: batch1_p50 differs from summary.json")
        else:
            summary_issues.append(f"{scale}/{model}: absent from summary.json")

    issues.extend(summary_issues)

    # Latency/throughput ratio stability against Independent, computed per seed.
    ratio_rows = []
    for scale in sorted({r["scale"] for r in ok}):
        ind = {r["seed"]: r for r in ok if r["scale"] == scale and r["model"] == "independent"}
        for model in sorted({r["model"] for r in ok if r["scale"] == scale and r["model"] != "independent"}):
            cur = {r["seed"]: r for r in ok if r["scale"] == scale and r["model"] == model}
            seeds = sorted(set(ind) & set(cur))
            if not seeds:
                continue
            b1 = [cur[k]["inference"]["batches"]["1"]["p50_us"] / ind[k]["inference"]["batches"]["1"]["p50_us"] for k in seeds]
            t = [cur[k]["inference"]["batches"]["1024"]["throughput_samples_s"] / ind[k]["inference"]["batches"]["1024"]["throughput_samples_s"] for k in seeds]
            ratio_rows.append({
                "scale": scale,
                "model": model,
                "seeds": seeds,
                "b1_ratio_per_seed": b1,
                "b1_ratio_mean": statistics.mean(b1),
                "b1_ratio_median": statistics.median(b1),
                "b1_ratio_min": min(b1),
                "b1_ratio_max": max(b1),
                "b1_ratio_of_means": statistics.mean([cur[k]["inference"]["batches"]["1"]["p50_us"] for k in seeds]) / statistics.mean([ind[k]["inference"]["batches"]["1"]["p50_us"] for k in seeds]),
                "t1024_ratio_per_seed": t,
                "t1024_ratio_mean": statistics.mean(t),
                "t1024_ratio_min": min(t),
                "t1024_ratio_max": max(t),
                "t1024_ratio_of_means": statistics.mean([cur[k]["inference"]["batches"]["1024"]["throughput_samples_s"] for k in seeds]) / statistics.mean([ind[k]["inference"]["batches"]["1024"]["throughput_samples_s"] for k in seeds]),
            })

    env_path = base / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    head = git("rev-parse", "HEAD")
    return {
        "dataset": name,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "records": len(records),
        "ok": len(ok),
        "failed": len(failed),
        "failed_records": [
            {"run_id": r.get("run_id"), "error": r.get("error"), "traceback": r.get("traceback")}
            for r in failed
        ],
        "models": sorted({r["model"] for r in ok}),
        "scales": sorted({r["scale"] for r in ok}),
        "seeds": sorted({r["seed"] for r in ok}),
        "formulas_checked": formulas_checked,
        "issues": issues,
        "group_rows": group_rows,
        "ratio_rows": ratio_rows,
        "environment": env,
        "git_head": head,
        "environment_matches_head": env.get("git_commit") == head,
        "raw_sha256": sha256_file(raw_path),
        "summary_sha256": sha256_file(summary_path) if summary_path.exists() else None,
    }


def write_csv(path: Path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "audit" / "stage0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = ["scaling_v1", "scaling_cross_ablation"]
    report = {
        "root": str(ROOT),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "datasets": [audit_dataset(d) for d in datasets],
    }
    (args.out / "stage0_evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    run_rows = []
    for ds in report["datasets"]:
        base = RESULTS / ds["dataset"]
        for r in load_jsonl(base / "raw_runs.jsonl"):
            if r.get("status") != "ok":
                continue
            b = r["inference"]["batches"]
            run_rows.append({
                "dataset": ds["dataset"],
                "run_id": r["run_id"],
                "scale": r["scale"],
                "model": r["model"],
                "seed": r["seed"],
                "total_params": r["cost"]["unique_trainable_parameters"],
                "active_params": r["cost"]["active_parameters_per_sample"],
                "macs": r["cost"]["macs_per_sample"],
                "accuracy_mean": r["metrics"]["accuracy_mean"],
                "train_seconds": r["train"]["seconds"],
                "batch1_p50_us": b["1"]["p50_us"],
                "batch1_p95_us": b["1"]["p95_us"],
                "batch256_throughput": b["256"]["throughput_samples_s"],
                "batch1024_throughput": b["1024"]["throughput_samples_s"],
                "checkpoint_bytes": r.get("checkpoint_bytes"),
            })
    fields = list(run_rows[0].keys())
    write_csv(args.out / "recomputed_runs.csv", run_rows, fields)

    group_rows = []
    for ds in report["datasets"]:
        for row in ds["group_rows"]:
            group_rows.append({"dataset": ds["dataset"], **row})
    write_csv(args.out / "recomputed_groups.csv", group_rows, list(group_rows[0].keys()))

    ratio_rows = []
    for ds in report["datasets"]:
        for row in ds["ratio_rows"]:
            ratio_rows.append({"dataset": ds["dataset"], **row})
    if ratio_rows:
        write_csv(args.out / "latency_ratios.csv", ratio_rows, list(ratio_rows[0].keys()))

    print(json.dumps({
        "git_head": report["git_head"],
        "git_branch": report["git_branch"],
        "datasets": [
            {
                "dataset": d["dataset"],
                "records": d["records"],
                "ok": d["ok"],
                "failed": d["failed"],
                "issues": d["issues"],
                "environment_matches_head": d["environment_matches_head"],
            }
            for d in report["datasets"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
