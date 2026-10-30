"""Dedicated-VRAM guard for Windows/WDDM experiments.

The goal is to fail fast instead of silently spilling model/data allocations
into Windows shared GPU memory (system RAM).  The guard checks free dedicated
VRAM, caps the PyTorch allocator below the device total, and records memory
snapshots around each run.
"""

from __future__ import annotations

import torch


def snapshot(device: int = 0) -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(device),
        "total_gb": round(total_bytes / 1024**3, 3),
        "free_gb": round(free_bytes / 1024**3, 3),
        "used_gb": round((total_bytes - free_bytes) / 1024**3, 3),
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / 1024**3, 3),
    }


def enforce_dedicated_vram(
    device: int = 0,
    *,
    min_free_gb: float = 2.0,
    max_fraction: float = 0.80,
) -> dict:
    """Check dedicated capacity and cap the allocator.

    The fraction cap makes PyTorch raise OOM before it can consume the whole
    device; the free-memory check rejects a run when desktop applications have
    already left too little dedicated VRAM.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; dedicated-VRAM guard cannot run")
    torch.cuda.init()
    info = snapshot(device)
    if info["free_gb"] < min_free_gb:
        raise RuntimeError(
            f"only {info['free_gb']:.2f} GiB free dedicated VRAM; "
            f"refusing to run below {min_free_gb:.2f} GiB (shared-memory spill risk)"
        )
    if not (0.1 <= max_fraction <= 0.95):
        raise ValueError("max_fraction must be in [0.1, 0.95]")
    torch.cuda.set_per_process_memory_fraction(max_fraction, device=device)
    info["max_fraction"] = max_fraction
    info["min_free_gb"] = min_free_gb
    return info
