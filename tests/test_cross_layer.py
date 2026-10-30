"""Mechanism tests for the cross-layer parameter-block prototype."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.scaling.models import ANetCrossLayer, analytical_parameters


def make_model() -> ANetCrossLayer:
    torch.manual_seed(0)
    model = ANetCrossLayer(hidden_dim=16, expert_hidden_dim=8).double()
    return model


def test_shared_block_is_registered_once_and_object_is_shared():
    model = make_model()
    shared_modules = [m for m in model.modules() if m is model.shared_block]
    assert len(shared_modules) == 1
    shared_keys = [k for k in model.state_dict() if "shared_block" in k]
    assert len(shared_keys) == 2, shared_keys
    total = sum(p.numel() for p in model.parameters())
    assert total == analytical_parameters("cross", 16, 8)

    calls = {"n": 0}

    def hook(_module, _inputs, _output):
        calls["n"] += 1

    handle = model.shared_block.register_forward_hook(hook)
    try:
        x0 = torch.randn(7, 2, dtype=torch.float64)
        x1 = torch.randn(5, 2, dtype=torch.float64)
        model(x0, 0)
        model(x1, 1)
    finally:
        handle.remove()
    assert calls["n"] == 2


def test_direct_call_matches_manual_formula():
    model = make_model().eval()
    x = torch.randn(11, 2, dtype=torch.float64)
    with torch.no_grad():
        out = model(x, 2)
        h = model.l1(x)
        expert = model.experts[2]
        z = F.relu(expert.fc1(h))
        z = z + model.shared_block(z)
        expected = expert.fc2(z)
    assert out.shape == (11, 2)
    assert torch.allclose(out, expected, atol=1e-12, rtol=1e-12)


def test_only_selected_expert_receives_gradient_but_shared_does():
    model = make_model()
    x = torch.randn(13, 2, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    model(x, 0).sum().backward()

    assert model.experts[0].fc1.weight.grad is not None
    assert model.experts[1].fc1.weight.grad is None
    assert model.experts[2].fc1.weight.grad is None
    assert model.shared_block[0].weight.grad is not None
    assert model.shared_block[0].weight.grad.abs().sum().item() > 0


def test_gradient_from_two_paths_accumulates_in_same_shared_parameters():
    model = make_model()
    x0 = torch.randn(9, 2, dtype=torch.float64)
    x1 = torch.randn(9, 2, dtype=torch.float64)
    p = model.shared_block[0].weight

    model.zero_grad(set_to_none=True)
    model(x0, 0).sum().backward()
    g0 = p.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    model(x1, 1).sum().backward()
    g1 = p.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    (model(x0, 0).sum() + model(x1, 1).sum()).backward()
    g_both = p.grad.detach().clone()

    assert g0.abs().sum() > 0
    assert g1.abs().sum() > 0
    assert torch.allclose(g_both, g0 + g1, atol=1e-12, rtol=1e-12)


def test_toggle_skips_block_and_matches_manual_no_cross_path():
    model = make_model().eval()
    model.use_cross_layer = False
    x = torch.randn(6, 2, dtype=torch.float64)

    calls = {"n": 0}

    def hook(_module, _inputs, _output):
        calls["n"] += 1

    handle = model.shared_block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(x, 1)
            h = model.l1(x)
            expert = model.experts[1]
            z = F.relu(expert.fc1(h))
            expected = expert.fc2(z)
    finally:
        handle.remove()

    assert calls["n"] == 0
    assert torch.allclose(out, expected, atol=1e-12, rtol=1e-12)
