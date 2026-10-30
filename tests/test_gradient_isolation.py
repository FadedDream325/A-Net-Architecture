"""Gradient isolation between task workflows and the shared block."""

from __future__ import annotations

import torch

from experiments.anet_v2.core.models import ANetModel


def _model(isolation: str) -> ANetModel:
    torch.manual_seed(0)
    return ANetModel(
        3, 6, 16, [2, 1, 5],
        share_mode="block", cache_enabled=False, active_mode="pair",
        gradient_isolation=isolation,
    )


def _grads(model: ANetModel, task_id: int = 0) -> dict:
    model.zero_grad(set_to_none=True)
    model(torch.randn(5, 6), task_id).square().mean().backward()
    shared = model.registry.get(model.cross_ids[0]).op[0].weight
    return {
        "shared": shared.grad,
        "w0_own": model.registry.get(model.own_ids[0]).op[0].weight.grad,
        "w1_own": model.registry.get(model.own_ids[1]).op[0].weight.grad,
        "w2_own": model.registry.get(model.own_ids[2]).op[0].weight.grad,
    }


def test_coupled_mode_updates_auxiliary_workflow_and_shared_block():
    grads = _grads(_model("none"), 0)
    assert grads["shared"] is not None
    assert grads["w0_own"] is not None
    assert grads["w1_own"] is not None  # auxiliary workflow in pair mode
    assert grads["w2_own"] is None


def test_detach_aux_isolates_non_primary_workflows():
    grads = _grads(_model("detach_aux"), 0)
    assert grads["w0_own"] is not None
    assert grads["w1_own"] is None
    assert grads["w2_own"] is None
    assert grads["shared"] is not None  # trained through the primary path


def test_detach_shared_blocks_gradient_into_shared_block():
    grads = _grads(_model("detach_shared"), 0)
    assert grads["shared"] is None
    assert grads["w0_own"] is not None
    assert grads["w1_own"] is not None


def test_detach_both_isolates_shared_and_auxiliary_paths():
    grads = _grads(_model("detach_both"), 0)
    assert grads["shared"] is None
    assert grads["w0_own"] is not None
    assert grads["w1_own"] is None


def test_isolation_does_not_change_forward_values():
    base = _model("none")
    x = torch.randn(7, 6)
    outputs = []
    for isolation in ("none", "detach_aux", "detach_shared", "detach_both"):
        model = _model(isolation)
        model.load_state_dict(base.state_dict())
        model.eval()
        outputs.append(model(x, 0))
    for out in outputs[1:]:
        assert torch.allclose(outputs[0], out, atol=1e-6, rtol=1e-6)
