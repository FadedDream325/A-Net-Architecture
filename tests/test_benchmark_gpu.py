"""GPU-resident input contract for the latency benchmark."""

from __future__ import annotations

import pytest
import torch

from experiments.anet_v2.core.benchmark import benchmark_inference
from experiments.anet_v2.core.models import build_model
from experiments.anet_v2.core.tasks import build_suite


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_benchmark_does_not_copy_inputs_in_timed_loop():
    device = torch.device("cuda")
    suite = build_suite(2026, "low", n_train=32, n_val=16, n_test=16).to(device)
    torch.manual_seed(0)
    model = build_model(
        "anet_full",
        num_tasks=3,
        in_dim=8,
        hidden_dim=16,
        out_dims=[s.out_dim for s in suite.specs],
        active_mode="sparse",
    ).to(device)
    result = benchmark_inference(
        model,
        suite,
        device=device,
        batch_sizes=[1],
        repeats=2,
        modes=["single"],
        warmup=1,
        reset_cache_each_repeat=True,
    )
    batch = result["batches"]["single"]["1"]
    assert batch["input_device"].startswith("cuda")
    assert result["host_to_device_bytes"] == 0
    assert result["model_device"].startswith("cuda")
