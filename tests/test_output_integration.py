"""Terminal L1 only integrates final workflow outputs."""

from __future__ import annotations

import pytest
import torch

from experiments.anet_v2.core.models import ANetModel
from experiments.anet_v2.core.workflow import TerminalL1


def test_terminal_receives_only_workflow_features():
    torch.manual_seed(0)
    model = ANetModel(
        num_tasks=3, in_dim=6, hidden_dim=16, out_dims=[2, 1, 5],
        share_mode="block", cache_enabled=False, active_mode="pair",
    )
    captured = {}

    def hook(_module, args):
        captured["outputs"] = args[0]
        captured["task_id"] = args[1]
        captured["active_ids"] = args[2]

    handle = model.terminal.register_forward_pre_hook(hook)
    try:
        x = torch.randn(5, 6)
        logits = model(x, 0, active_ids=[0, 2])
    finally:
        handle.remove()

    assert logits.shape == (5, 2)
    assert captured["task_id"] == 0
    assert captured["active_ids"] == [0, 2]
    assert len(captured["outputs"]) == 2
    for z in captured["outputs"]:
        assert z.shape == (5, model.workflow_out_dim)
        assert z.shape[1] != model.hidden_dim


def test_terminal_rejects_mismatched_outputs_and_active_ids():
    terminal = TerminalL1(3, 8, [2, 1, 5])
    z = torch.randn(4, 8)
    with pytest.raises(ValueError):
        terminal([z], 0, [0, 1])
    with pytest.raises(ValueError):
        terminal([], 0, [])


def test_all_task_outputs_have_declared_shapes():
    torch.manual_seed(1)
    model = ANetModel(3, 6, 16, [2, 1, 5], share_mode="block", cache_enabled=False, active_mode="sparse")
    x = torch.randn(3, 6)
    for t, dim in enumerate([2, 1, 5]):
        out = model(x, t)
        assert out.shape == (3, dim), (t, out.shape)
