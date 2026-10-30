"""Parameter updates must invalidate cached results."""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.anet_v2.core.parameter_block import BlockSpec, ParameterBlock
from experiments.anet_v2.core.result_cache import CacheKey, ResultCache


def test_manual_update_and_version_invalidate():
    torch.manual_seed(0)
    block = ParameterBlock(BlockSpec("b", "w0", 3, 1), nn.Linear(3, 1))
    cache = ResultCache(enabled=True)
    x = torch.randn(5, 3)
    key = CacheKey("b", block.version, "x", "cfg")
    old = block(x).detach().clone()
    cache.put(key, old)
    assert cache.get(key) is not None

    with torch.no_grad():
        block.op.weight.add_(2.0)
        block.op.bias.add_(0.5)
    block.mark_updated()
    new_key = CacheKey("b", block.version, "x", "cfg")
    assert cache.get(new_key) is None
    cache.invalidate_block("b")
    assert cache.get(key) is None


def test_optimizer_step_changes_output_and_version():
    torch.manual_seed(1)
    block = ParameterBlock(BlockSpec("b", "w0", 4, 1), nn.Linear(4, 1))
    opt = torch.optim.SGD(block.parameters(), lr=0.1)
    x = torch.randn(8, 4)
    before = block(x).detach().clone()
    opt.zero_grad()
    loss = block(x).square().mean()
    loss.backward()
    opt.step()
    block.mark_updated()
    after = block(x).detach().clone()
    assert not torch.allclose(before, after)
    assert block.version == 1


def test_cache_clear_removes_everything():
    cache = ResultCache(enabled=True)
    for i in range(3):
        cache.put(CacheKey("b", i, "x", "cfg"), torch.ones(2))
    assert cache.size == 3
    cache.clear()
    assert cache.size == 0
    assert cache.stats["clears"] == 1
