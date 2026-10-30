"""Correctness tests for the scaling model zoo and cost accounting."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.scaling.models import (
    MODEL_KEYS,
    active_parameters,
    analytical_flops,
    analytical_parameters,
    build_model,
    count_parameters,
    model_spec_rows,
)
from experiments.scaling.scales import SCALES


def test_parameter_formulas_small_width():
    h, e = 32, 16
    expected = {
        "independent": 3 * (h * h + 6 * h + 2),
        "shared": h * h + 10 * h + 6,
        "anet": 3 * h + 3 * h * e + 9 * e + 6,
        "cross": 3 * h + 3 * h * e + 9 * e + 6 + e * e + e,
        "cross_off": 3 * h + 3 * h * e + 9 * e + 6 + e * e + e,
    }
    for kind in MODEL_KEYS:
        torch.manual_seed(0)
        model = build_model(kind, h, e)
        assert count_parameters(model) == expected[kind]
        assert count_parameters(model) == analytical_parameters(kind, h, e)
        rows = model_spec_rows(kind, h, e, model)
        assert rows["unique_trainable_parameters"] == expected[kind]
        assert rows["active_parameters_per_sample"] <= expected[kind]


def test_scale_registry_parameter_targets():
    for scale in SCALES.values():
        model = build_model("anet", scale.hidden_dim, scale.expert_hidden_dim)
        total = count_parameters(model)
        assert 80_000 <= total <= 6_000_000
        assert active_parameters(model, "anet") < total


def test_forward_backward_and_shapes_for_all_models():
    for kind in MODEL_KEYS:
        torch.manual_seed(1)
        model = build_model(kind, 24, 12)
        x = torch.randn(19, 2)
        y = torch.randint(0, 2, (19,))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, 0)
        assert logits.shape == (19, 2), (kind, logits.shape)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)


def test_anet_only_selected_expert_gets_gradient():
    torch.manual_seed(2)
    model = build_model("anet", 24, 12)
    x = torch.randn(8, 2)
    model.zero_grad(set_to_none=True)
    model(x, 1).sum().backward()
    assert model.experts[1][0].weight.grad is not None
    assert model.experts[0][0].weight.grad is None
    assert model.experts[2][0].weight.grad is None
    assert model.l1[0].weight.grad is not None


def test_cross_layer_has_more_params_than_anet_but_same_forward_contract():
    torch.manual_seed(3)
    anet = build_model("anet", 32, 16)
    cross = build_model("cross", 32, 16)
    assert count_parameters(cross) > count_parameters(anet)
    x = torch.randn(5, 2)
    assert cross(x, 0).shape == (5, 2)
    assert anet(x, 0).shape == (5, 2)


def test_analytical_flops_match_torch_flop_counter():
    from torch.utils.flop_counter import FlopCounterMode

    for kind in MODEL_KEYS:
        torch.manual_seed(4)
        model = build_model(kind, 16, 8).eval()
        x = torch.randn(1, 2)
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x, 0)
        measured = int(counter.get_total_flops())
        expected = analytical_flops(kind, 16, 8)
        assert measured == expected, (kind, measured, expected)
