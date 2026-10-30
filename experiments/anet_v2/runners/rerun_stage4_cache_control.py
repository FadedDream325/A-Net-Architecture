"""Re-train the selected Stage 4 cache-eval cells and measure cache on/off.

This produces a paired cache-off control for the Stage 4 combined analysis.
The selected cells are width=2048, workflows=3, medium correlation, seeds
2026-2030, models anet_reuse and anet_full.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.anet_v2.core.cache_benchmark import benchmark_cache_stream
from experiments.anet_v2.core.device_guard import enforce_dedicated_vram
from experiments.anet_v2.core.models import build_model
from experiments.anet_v2.core.tasks import build_suite
from experiments.anet_v2.core.train import TrainConfig, set_seed, train_model
from experiments.anet_v2.runners.run_stage1 import environment


def main() -> int:
    out = ROOT / "experiments" / "anet_v2" / "results" / "stage4" / "cache_control.jsonl"
    device = torch.device("cuda")
    enforce_dedicated_vram()
    env = environment()
    width, workflows, corr = 2048, 3, "medium"
    for seed in [2026, 2027, 2028, 2029, 2030]:
        suite = build_suite(seed, corr, n_train=6000, n_val=1500, n_test=1500).to(device)
        for kind in ["anet_reuse", "anet_full"]:
            set_seed(seed)
            model = build_model(
                kind, num_tasks=3, num_workflows=workflows, in_dim=suite.meta["in_dim"],
                hidden_dim=width, out_dims=[s.out_dim for s in suite.specs],
                active_mode="pair", gradient_isolation="detach_both",
            ).to(device)
            train_model(model, suite, TrainConfig(epochs=30, batch_size=512), device=device)
            for batch in (1, 256):
                for rate in (0.0, 0.25, 0.5, 0.75):
                    for cache_enabled in (False, True):
                        model.set_inference_cache(cache_enabled)
                        model.cache.clear()
                        res = benchmark_cache_stream(
                            model, suite, device=device, batch_size=batch,
                            requests=100, repetition_rate=rate, cache_enabled=cache_enabled,
                        )
                        rec = {
                            "seed": seed, "width": width, "workflows": workflows,
                            "correlation": corr, "model": kind, "batch": batch, "rate": rate,
                            "cache_enabled": cache_enabled, "git_commit": env["git_commit"],
                            "code_hash": env["code_hash"], **res,
                        }
                        with open(out, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        print(seed, kind, batch, rate, cache_enabled,
                              round(res["p50_us"], 1), round(res["cache"]["hit_rate"], 2), flush=True)
            del model
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
