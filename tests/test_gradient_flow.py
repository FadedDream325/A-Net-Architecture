"""Gradient accumulation across multiple callers of one parameter block."""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.anet_v2.core.models import ANetModel
from experiments.anet_v2.core.parameter_block import BlockSpec, ParameterBlock


def make_block(seed: int = 0) -> ParameterBlock:
    torch.manual_seed(seed)
    return ParameterBlock(
        BlockSpec("b", "w0", 4, 4, kind="test"),
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
    ).double()


def test_two_callers_gradient_equals_sum_of_separate_gradients():
    block = make_block()
    x1 = torch.randn(3, 4, dtype=torch.float64)
    x2 = torch.randn(5, 4, dtype=torch.float64)
    p = block.op[0].weight

    block.zero_grad(set_to_none=True)
    (block(x1).sum() + block(x2).sum()).backward()
    g_both = p.grad.detach().clone()

    block.zero_grad(set_to_none=True)
    block(x1).sum().backward()
    g1 = p.grad.detach().clone()

    block.zero_grad(set_to_none=True)
    block(x2).sum().backward()
    g2 = p.grad.detach().clone()

    assert torch.allclose(g_both, g1 + g2, atol=1e-12, rtol=1e-12)


def test_shared_block_receives_gradient_in_pair_mode():
    torch.manual_seed(0)
    model = ANetModel(
        num_tasks=3,
        in_dim=6,
        hidden_dim=16,
        out_dims=[2, 1, 5],
        share_mode="block",
        cache_enabled=False,
        active_mode="pair",
    )
    x = torch.randn(7, 6)
    model.zero_grad(set_to_none=True)
    logits = model(x, 0)
    loss = logits.square().mean()
    loss.backward()
    shared = model.registry.get(model.cross_ids[0])
    grad = shared.op[0].weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0
    assert torch.isfinite(grad).all()


def test_only_active_workflows_receive_gradient():
    torch.manual_seed(1)
    model = ANetModel(
        num_tasks=3,
        in_dim=6,
        hidden_dim=16,
        out_dims=[2, 1, 5],
        share_mode="none",
        cache_enabled=False,
        active_mode="sparse",
    )
    x = torch.randn(5, 6)
    model.zero_grad(set_to_none=True)
    model(x, 1).sum().backward()
    assert model.registry.get(model.own_ids[1]).op[0].weight.grad is not None
    assert model.registry.get(model.own_ids[0]).op[0].weight.grad is None
    assert model.registry.get(model.own_ids[2]).op[0].weight.grad is None
