"""Collect the stage-0 verification evidence into one JSON file."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "audit" / "stage0"


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("gbk", errors="replace")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out = {"files": {}, "results": {}}
    files = {
        "pytest": AUDIT / "pytest_stage0.log",
        "demo": AUDIT / "legacy_demo_cpu.log",
        "complex_log": AUDIT / "complex_tasks_cpu_stage0.log",
        "complex_json": AUDIT / ".complex_results_cpu.json",
        "repro_log": AUDIT / "repro_small_anet_s2026.log",
        "repro_raw": AUDIT / "repro_small_anet_s2026" / "raw_runs.jsonl",
        "cuda": AUDIT / "cuda_compat_stage0.log",
    }
    for key, path in files.items():
        if path.exists():
            out["files"][key] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}

    pytest_text = read_text(files["pytest"])
    m = re.search(r"(\d+) passed", pytest_text)
    out["results"]["pytest"] = {"passed": int(m.group(1)) if m else None, "exit_ok": "failed" not in pytest_text.lower()}

    demo_text = read_text(files["demo"])
    demo = {}
    for name in ("独立基线", "共享主干", "A-Net"):
        matches = re.findall(rf"{re.escape(name)}.*?平均准确率\s+([\d.]+)%", demo_text)
        if matches:
            demo[name] = float(matches[-1])
    out["results"]["demo_cpu"] = demo

    if files["complex_json"].exists():
        cj = json.loads(files["complex_json"].read_text(encoding="utf-8"))
        out["results"]["complex_cpu"] = {
            name: {
                "average_mean": v.get("average_mean"),
                "average_std": v.get("average_std"),
                "task_mean": v.get("task_mean"),
                "total_params": v.get("total_params"),
                "active_params": v.get("active_params"),
            }
            for name, v in cj.get("summary", {}).items()
        }

    cuda_text = read_text(files["cuda"])
    out["results"]["cuda"] = {
        "failures": int(re.search(r"失败项数量:\s*(\d+)", cuda_text).group(1)) if re.search(r"失败项数量:\s*(\d+)", cuda_text) else None,
        "pass": "CUDA 兼容性: PASS" in cuda_text,
    }

    old = [json.loads(l) for l in (ROOT / "experiments/scaling/results/scaling_v1/raw_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    old = [r for r in old if r["run_id"] == "small__anet__s2026"][0]
    new = json.loads(files["repro_raw"].read_text(encoding="utf-8").splitlines()[0])
    out["results"]["scaling_repro"] = {
        "run_id": new["run_id"],
        "accuracy_old": old["metrics"]["accuracy_per_task"],
        "accuracy_new": new["metrics"]["accuracy_per_task"],
        "accuracy_exact_match": old["metrics"]["accuracy_per_task"] == new["metrics"]["accuracy_per_task"],
        "total_params": new["cost"]["unique_trainable_parameters"],
        "active_params": new["cost"]["active_parameters_per_sample"],
        "train_seconds_old": old["train"]["seconds"],
        "train_seconds_new": new["train"]["seconds"],
        "batch1_p50_old_us": old["inference"]["batches"]["1"]["p50_us"],
        "batch1_p50_new_us": new["inference"]["batches"]["1"]["p50_us"],
    }

    (AUDIT / "verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
