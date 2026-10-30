"""Parameter versions and cache invalidation."""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.anet_v2.core.models import ANetModel
from experiments.anet_v2.core.parameter_block import BlockSpec, ParameterBlock
from experiments.anet_v2.core.result_cache import CacheKey, ResultCache


def test_version_increments_and_param_token_changes():
    block = ParameterBlock(BlockSpec("b", "w0", 4, 3), nn.Linear(4, 3))
    assert block.version == 0
    token0 = block.param_version_token()
    block.mark_updated()
    assert block.version == 1
    with torch.no_grad():
        block.op.weight.add_(1.0)
    assert block.param_version_token() != token0


def test_cache_key_version_change_misses_old_entry():
    cache = ResultCache(enabled=True)
    key0 = CacheKey("b", 0, "input", "cfg")
    value = torch.ones(2)
    cache.put(key0, value)
    assert cache.get(key0) is not None
    assert cache.stats["hits"] == 1
    key1 = CacheKey("b", 1, "input", "cfg")
    assert cache.get(key1) is None
    assert cache.stats["misses"] == 1


def test_model_mark_updated_clears_cache_and_bumps_every_block():
    torch.manual_seed(0)
    model = ANetModel(
        num_tasks=3,
        in_dim=6,
        hidden_dim=16,
        out_dims=[2, 1, 5],
        share_mode="block",
        cache_enabled=True,
        active_mode="sparse",
    )
    model.eval()
    x = torch.randn(4, 6)
    model(x, 0)
    assert model.cache.size > 0
    before = {i: model.registry.get(i).version for i in model.registry.block_ids()}
    model.mark_updated()
    after = {i: model.registry.get(i).version for i in model.registry.block_ids()}
    assert all(after[i] == before[i] + 1 for i in before)
    assert model.cache.size == 0


def test_begin_step_clears_training_cache():
    torch.manual_seed(1)
    model = ANetModel(
        num_tasks=3,
        in_dim=6,
        hidden_dim=16,
        out_dims=[2, 1, 5],
        share_mode="block",
        cache_enabled=True,
        active_mode="pair",
    )
    model.train()
    x = torch.randn(4, 6)
    model.begin_step(0)
    model(x, 0)
    assert model.cache.size > 0
    model.begin_step(1)
    assert model.cache.size == 0
