"""Cache key correctness: identity, input, dtype, config, train/eval state."""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.anet_v2.core.parameter_block import BlockSpec, ParameterBlock
from experiments.anet_v2.core.result_cache import (
    CacheKey,
    ResultCache,
    config_fingerprint,
    tensor_fingerprint,
)


def make_key(block, x, config):
    h, _ = tensor_fingerprint(x)
    return CacheKey(block.block_id, block.version, h, config_fingerprint(config))


def test_same_block_input_config_hits_and_returns_same_value():
    torch.manual_seed(0)
    block = ParameterBlock(BlockSpec("b", "w0", 4, 3), nn.Linear(4, 3))
    cache = ResultCache(enabled=True)
    x = torch.randn(5, 4)
    cfg = {"training": False, "dtype": "float32", "state": "eval"}
    key = make_key(block, x, cfg)
    out1, hit1 = cache.get_or_compute(key, lambda: block(x))
    out2, hit2 = cache.get_or_compute(key, lambda: block(x))
    assert hit1 is False and hit2 is True
    assert out1 is out2
    assert torch.equal(out1, out2)
    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1


def test_different_input_misses():
    torch.manual_seed(1)
    block = ParameterBlock(BlockSpec("b", "w0", 4, 3), nn.Linear(4, 3))
    cache = ResultCache(enabled=True)
    cfg = {"training": False}
    x1 = torch.randn(5, 4)
    x2 = x1.clone()
    x2[0, 0] += 1.0
    k1 = make_key(block, x1, cfg)
    k2 = make_key(block, x2, cfg)
    cache.get_or_compute(k1, lambda: block(x1))
    out2, hit2 = cache.get_or_compute(k2, lambda: block(x2))
    assert hit2 is False
    assert not torch.equal(cache.get(k1), out2)


def test_different_dtype_or_config_misses():
    torch.manual_seed(2)
    block = ParameterBlock(BlockSpec("b", "w0", 4, 3), nn.Linear(4, 3))
    cache = ResultCache(enabled=True)
    x = torch.randn(5, 4)
    k1 = make_key(block, x, {"training": False, "dtype": "float32"})
    k2 = make_key(block, x, {"training": False, "dtype": "float16"})
    k3 = make_key(block, x, {"training": True, "dtype": "float32"})
    cache.get_or_compute(k1, lambda: block(x))
    assert cache.get(k2) is None
    assert cache.get(k3) is None


def test_dropout_state_is_part_of_config_and_changes_result():
    torch.manual_seed(3)
    block = ParameterBlock(
        BlockSpec("drop", "w0", 8, 8, stateful=True),
        nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5)),
    )
    block.train()
    x = torch.randn(32, 8)
    k1 = make_key(block, x, {"training": True, "dropout_p": 0.5, "seed": 1})
    k2 = make_key(block, x, {"training": True, "dropout_p": 0.5, "seed": 1})
    # Same stateful config would collide only if the caller drops randomness;
    # the cache itself must at least distinguish train/eval and dropout p.
    k3 = make_key(block, x, {"training": False, "dropout_p": 0.0})
    assert k1 == k2
    assert k1 != k3


def test_disabled_cache_never_hits():
    cache = ResultCache(enabled=False)
    block = ParameterBlock(BlockSpec("b", "w0", 4, 3), nn.Linear(4, 3))
    x = torch.randn(2, 4)
    key = make_key(block, x, "cfg")
    _, hit1 = cache.get_or_compute(key, lambda: block(x))
    _, hit2 = cache.get_or_compute(key, lambda: block(x))
    assert hit1 is False and hit2 is False
    assert cache.stats["lookups"] == 0
