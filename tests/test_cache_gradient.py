"""Cached reuse must preserve gradients for every caller."""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.anet_v2.core.models import ANetModel
from experiments.anet_v2.core.parameter_block import BlockSpec, ParameterBlock
from experiments.anet_v2.core.result_cache import CacheKey, ResultCache


def _run(cache_enabled: bool, same_input: bool = True):
    torch.manual_seed(0)
    block = ParameterBlock(
        BlockSpec("shared", "w0", 4, 4),
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
    )
    c1 = nn.Linear(4, 1)
    c2 = nn.Linear(4, 1)
    x1 = torch.randn(6, 4)
    x2 = x1.clone() if same_input else torch.randn(6, 4)
    cache = ResultCache(enabled=cache_enabled)
    key1 = CacheKey("shared", block.version, "x1", "cfg")
    key2 = CacheKey("shared", block.version, "x2" if not same_input else "x1", "cfg")
    z1, _ = cache.get_or_compute(key1, lambda: block(x1))
    z2, hit2 = cache.get_or_compute(key2, lambda: block(x2))
    loss = (z1 * c1.weight).sum() + (z2 * c2.weight).sum()
    loss.backward()
    return block.op[0].weight.grad.detach().clone(), hit2


def test_cache_on_gradient_equals_cache_off_for_shared_input():
    g_off, _ = _run(False)
    g_on, hit = _run(True)
    assert hit is True
    assert torch.allclose(g_on, g_off, atol=1e-6, rtol=1e-6)


def test_cache_on_gradient_equals_cache_off_for_distinct_inputs():
    g_off, _ = _run(False, same_input=False)
    g_on, hit = _run(True, same_input=False)
    assert hit is False
    assert torch.allclose(g_on, g_off, atol=1e-6, rtol=1e-6)


def test_model_full_and_share_have_same_gradient_for_shared_block():
    torch.manual_seed(2)
    share = ANetModel(3, 6, 16, [2, 1, 5], share_mode="block", cache_enabled=False, active_mode="pair")
    torch.manual_seed(2)
    full = ANetModel(3, 6, 16, [2, 1, 5], share_mode="block", cache_enabled=True, active_mode="pair")
    full.load_state_dict(share.state_dict())
    x = torch.randn(9, 6)
    losses = []
    grads = []
    for model in (share, full):
        model.zero_grad(set_to_none=True)
        logits = model(x, 0)
        loss = logits.square().mean()
        loss.backward()
        shared = model.registry.get(model.cross_ids[0])
        grads.append(shared.op[0].weight.grad.detach().clone())
        losses.append(loss.detach().clone())
    assert torch.allclose(losses[0], losses[1], atol=1e-12)
    assert torch.allclose(grads[0], grads[1], atol=1e-12)
