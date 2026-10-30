"""Cache stream repetition and hit behavior."""

from __future__ import annotations

import pytest
import torch

from experiments.anet_v2.core.cache_benchmark import benchmark_cache_stream, repetition_indices
from experiments.anet_v2.core.models import build_model
from experiments.anet_v2.core.tasks import build_suite


def test_repetition_indices_counts():
    assert repetition_indices(8, 0.0) == list(range(8))
    assert len(set(repetition_indices(8, 0.5))) == 4
    assert len(set(repetition_indices(8, 0.25))) == 6
    assert len(set(repetition_indices(8, 0.75))) == 2
    with pytest.raises(ValueError):
        repetition_indices(8, 0.3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cache_stream_reduces_executions_and_invalidates():
    device = torch.device("cuda")
    suite = build_suite(2026, "medium", n_train=32, n_val=16, n_test=64).to(device)
    model = build_model(
        "anet_full", num_tasks=3, in_dim=8, hidden_dim=16, out_dims=[s.out_dim for s in suite.specs],
        active_mode="pair", gradient_isolation="detach_both",
    ).to(device)
    result = benchmark_cache_stream(
        model, suite, device=device, batch_size=2, requests=8,
        repetition_rate=0.5, cache_enabled=True,
    )
    assert result["input_device"].startswith("cuda")
    assert result["host_to_device_bytes"] == 0
    assert result["cache"]["hit_rate"] > 0
    assert result["actual_executions"] < result["theoretical_executions"]
    assert result["cache_invalidation_ok"] is True
