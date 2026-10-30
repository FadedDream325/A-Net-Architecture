"""Variable workflow count independent of task count."""

from __future__ import annotations

import torch

from experiments.anet_v2.core.models import ANetModel


def test_pair_routing_for_workflow_counts():
    for w in (2, 3, 4):
        model = ANetModel(3, 6, 16, [2, 1, 5], num_workflows=w, share_mode="block",
                          cache_enabled=False, active_mode="pair")
        expected = tuple((t % w, (t % w + 1) % w) for t in range(3))
        assert model._active_table == expected
        assert model.terminal.gates.shape == (3, w)
        assert model.graph.terminal_inputs() == {f"w{i}.output" for i in range(w)}
        out = model(torch.randn(4, 6), 0)
        assert out.shape == (4, 2)


def test_dense_uses_all_workflows_and_sparse_uses_primary():
    dense = ANetModel(3, 6, 16, [2, 1, 5], num_workflows=4, active_mode="dense")
    assert dense._active_table == ((0, 1, 2, 3),) * 3
    sparse = ANetModel(3, 6, 16, [2, 1, 5], num_workflows=2, active_mode="sparse")
    assert sparse._active_table == ((0,), (1,), (0,))
    assert sparse(torch.randn(3, 6), 1).shape == (3, 1)


def test_workflow_count_changes_parameter_count():
    small = ANetModel(3, 6, 16, [2, 1, 5], num_workflows=2, share_mode="none")
    large = ANetModel(3, 6, 16, [2, 1, 5], num_workflows=4, share_mode="none")
    assert large.unique_parameter_count() > small.unique_parameter_count()
    assert large.terminal.gates.numel() == 12
