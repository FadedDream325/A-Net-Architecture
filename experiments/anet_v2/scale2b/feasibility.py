"""2B feasibility calculator: memory, compute, and a parameter-matched Transformer config.

This script does not train a 2B model.  It makes the resource arithmetic explicit
so a small proxy cannot be mistaken for a 2B experiment.
"""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import torch

try:
    import psutil
    RAM_TOTAL_GB = psutil.virtual_memory().total / 1024**3
    RAM_AVAILABLE_GB = psutil.virtual_memory().available / 1024**3
    CPU_LOGICAL = psutil.cpu_count(logical=True)
    CPU_PHYSICAL = psutil.cpu_count(logical=False)
except Exception:
    RAM_TOTAL_GB = RAM_AVAILABLE_GB = CPU_LOGICAL = CPU_PHYSICAL = None


def transformer_params(vocab: int, d: int, layers: int, ffn: int, tied: bool = True) -> int:
    # Attention QKVO + FFN + two norms per layer; final norm; token embedding.
    per_layer = 4 * d * d + 2 * d * ffn + 2 * d
    total = vocab * d + layers * per_layer + d
    if not tied:
        total += vocab * d
    return total


def main() -> int:
    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": p.name,
            "total_gb": round(total / 1024**3, 3),
            "free_gb": round(free / 1024**3, 3),
            "capability": f"{p.major}.{p.minor}",
            "sm_count": p.multi_processor_count,
        }
    N = 2_000_000_000
    memory = {
        "fp32_weights_gb": N * 4 / 1e9,
        "fp32_grads_gb": N * 4 / 1e9,
        "adam_fp32_states_gb": N * 8 / 1e9,
        "fp32_training_total_gb": N * 16 / 1e9,
        "mixed_precision_weights_fp16_gb": N * 2 / 1e9,
        "mixed_precision_grads_fp16_gb": N * 2 / 1e9,
        "mixed_precision_master_fp32_gb": N * 4 / 1e9,
        "mixed_precision_adam_fp32_gb": N * 8 / 1e9,
        "mixed_precision_training_total_gb": N * 16 / 1e9,
    }
    tokens = {d: {"tokens_b": d / 1e9, "flops": 6 * N * d} for d in (1e9, 20e9, 100e9)}
    config = {
        "vocab": 32000,
        "d_model": 2240,
        "layers": 32,
        "ffn": 8960,
        "tied_embeddings": True,
        "parameters": transformer_params(32000, 2240, 32, 8960),
    }
    config["deviation_from_2b_pct"] = (config["parameters"] - N) / N * 100
    out = {
        "host": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "ram_total_gb": round(RAM_TOTAL_GB, 2) if RAM_TOTAL_GB else None,
        "ram_available_gb": round(RAM_AVAILABLE_GB, 2) if RAM_AVAILABLE_GB else None,
        "cpu_logical": CPU_LOGICAL,
        "cpu_physical": CPU_PHYSICAL,
        "disk_free_gb": round(shutil.disk_usage(str(Path.cwd())).free / 1024**3, 2),
        "parameter_budget": N,
        "memory_2b": memory,
        "compute_2b": tokens,
        "example_transformer_config": config,
        "verdict": "NO-GO on local hardware: 2B full training needs ~32 GB just for weights/grads/Adam states, plus activations; local GPU has 8 GB and host RAM is 15 GB total.",
    }
    print(json.dumps(out, indent=2))
    (Path(__file__).resolve().parent / "feasibility.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
