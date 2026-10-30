"""Spawn fresh processes to measure cold-start latency for saved checkpoints.

This intentionally uses one seed (default 2026) because cold-start overhead is a
property of the model/hardware and is not averaged meaningfully over training
seeds. The parent measures process wall time; the child measures CUDA init,
checkpoint load, the first forward, and the warmed latency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "experiments" / "scaling" / "results" / "scaling_v1",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    checkpoint_dir = args.results_dir / "checkpoints"
    out_dir = args.results_dir / "cold_start"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for checkpoint in sorted(checkpoint_dir.glob(f"*__s{args.seed}.pt")):
        child_out = out_dir / f"{checkpoint.stem}.json"
        command = [
            sys.executable,
            "-m",
            "experiments.scaling.benchmark_cold_start",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(child_out),
            "--repeats",
            str(args.repeats),
        ]
        print("running", checkpoint.name, flush=True)
        start = time.perf_counter()
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=__import__("os").environ.copy(),
        )
        wall = time.perf_counter() - start
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(f"child failed for {checkpoint}")
        rec = json.loads(child_out.read_text(encoding="utf-8"))
        rec["process_wall_s"] = wall
        records.append(rec)

    (out_dir / "cold_start.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
