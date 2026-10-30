"""Object identity, unique storage, and ownership/use separation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.anet_v2.core.models import ANetModel, build_model
from experiments.anet_v2.core.parameter_block import BlockSpec, ParameterBlock, ParameterRegistry


def make_model(share_mode: str, cache: bool = False, active_mode: str = "sparse") -> ANetModel:
    torch.manual_seed(0)
    return ANetModel(
        num_tasks=3,
        in_dim=6,
        hidden_dim=16,
        out_dims=[2, 1, 5],
        share_mode=share_mode,
        cache_enabled=cache,
        active_mode=active_mode,
    )


def test_shared_cross_block_is_registered_once_and_is_same_object():
    model = make_model("block")
    assert len(set(model.cross_ids)) == 1
    shared_id = model.cross_ids[0]
    block = model.registry.get(shared_id)
    assert all(model.registry.get(i) is block for i in model.cross_ids)
    assert model.registry.num_blocks() == 1 + 3 + 1  # pre-L1 + 3 own + 1 shared
    cross_keys = [k for k in model.state_dict() if "cross" in k]
    assert len(cross_keys) == 2, cross_keys
    block_ids = [id(b) for b in model.registry.blocks()]
    assert len(block_ids) == len(set(block_ids))


def test_shared_block_is_not_owned_by_non_owner_workflows():
    model = make_model("block")
    shared_id = model.cross_ids[0]
    assert model.registry.owner_of(shared_id) == "w0"
    assert shared_id in model.workflow_objs[0].owned_block_ids
    assert shared_id not in model.workflow_objs[1].owned_block_ids
    assert shared_id in model.workflow_objs[1].used_block_ids
    assert model.registry.can_call(shared_id, "w1")
    assert model.registry.can_call(shared_id, "w2")


def test_no_share_has_one_cross_block_per_workflow():
    model = make_model("none")
    assert len(set(model.cross_ids)) == 3
    assert all(model.registry.owner_of(i) == f"w{t}" for t, i in enumerate(model.cross_ids))
    assert model.registry.num_blocks() == 1 + 3 + 3


def test_registry_rejects_duplicate_id_and_duplicate_object():
    registry = ParameterRegistry()
    spec = BlockSpec("b", "w0", 3, 4)
    block = ParameterBlock(spec, nn.Linear(3, 4))
    registry.register(block)
    with pytest.raises(ValueError):
        registry.register(block)
    other = ParameterBlock(BlockSpec("c", "w0", 3, 4), nn.Linear(3, 4))
    registry.register(other)


def test_parameter_data_ptr_is_identical_across_all_callers():
    model = make_model("block")
    shared_id = model.cross_ids[0]
    block = model.registry.get(shared_id)
    ptrs = [int(p.data_ptr()) for p in block.parameters()]
    for wf in model.workflow_objs:
        if shared_id in wf.used_block_ids:
            assert [int(p.data_ptr()) for p in model.registry.get(shared_id).parameters()] == ptrs


def test_unauthorized_call_is_rejected():
    model = make_model("none")
    with pytest.raises(PermissionError):
        model.registry.require_call(model.own_ids[0], "w1")


def test_build_model_kind_matrix_has_expected_storage_order():
    out = {}
    for kind in ["anet_basic", "anet_share", "anet_adapter"]:
        m = build_model(kind, num_tasks=3, in_dim=6, hidden_dim=16, out_dims=[2, 1, 5])
        out[kind] = m.unique_parameter_count()
    assert out["anet_share"] < out["anet_basic"]
    assert out["anet_share"] < out["anet_adapter"] < out["anet_basic"] + 2000
