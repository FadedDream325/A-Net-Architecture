"""The five accounting quantities must stay distinct."""

from __future__ import annotations

import torch

from experiments.anet_v2.core.models import ANetModel


def _model(share_mode: str, cache: bool) -> ANetModel:
    torch.manual_seed(0)
    return ANetModel(
        num_tasks=3, in_dim=6, hidden_dim=16, out_dims=[2, 1, 5],
        share_mode=share_mode, cache_enabled=cache, active_mode="pair",
    )


def test_cache_off_counts_every_call():
    model = _model("block", False)
    model.eval()
    x = torch.randn(7, 6)
    model(x, 0)
    acc = model.cost_summary()["accounting"]
    assert acc["reference_params"] >= acc["active_params_per_sample"]
    assert acc["actual_executions_per_sample"] == 5  # pre-L1 + 2 cross + 2 own
    assert acc["actual_flops_per_sample"] == acc["theoretical_flops_per_sample"]


def test_cache_on_reduces_executions_but_not_theoretical_work():
    model = _model("block", True)
    model.eval()
    x = torch.randn(7, 6)
    model(x, 0)
    acc = model.cost_summary()["accounting"]
    assert acc["actual_executions_per_sample"] < acc["theoretical_executions_per_sample"]
    assert acc["actual_flops_per_sample"] < acc["theoretical_flops_per_sample"]
    assert model.cost_summary()["cache"]["hits"] >= 1
    assert acc["reference_params"] > acc["active_params_per_sample"]


def test_unique_params_equal_registry_plus_terminal():
    model = _model("block", False)
    assert model.unique_parameter_count() == (
        model.registry.unique_parameter_count() + model.terminal.parameter_count()
    )


def test_adapter_params_are_separate():
    model = _model("adapter", False)
    model.eval()
    model(torch.randn(4, 6), 0)
    acc = model.cost_summary()["accounting"]
    assert acc["adapter_params"] > 0
    assert acc["adapter_executions"] == 2
    summary = model.cost_summary()
    assert summary["adapter_params_total"] > summary["adapter_params_executed"]
    assert summary["adapter_params_total"] == model.adapter_parameter_count()
