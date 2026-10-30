"""Task-ID routing, sorted mixed batches, and the inference cache switch."""

from __future__ import annotations

import torch

from experiments.anet_v2.core.models import ANetModel


def _model(mode: str = "pair", cache: bool = False) -> ANetModel:
    torch.manual_seed(0)
    return ANetModel(3, 6, 16, [2, 1, 5], share_mode="block", cache_enabled=cache, active_mode=mode)


def test_forward_mixed_matches_per_task_loop_in_original_order():
    model = _model("pair")
    model.eval()
    x = torch.randn(17, 6)
    task_ids = torch.tensor([0, 2, 1, 0, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 2, 1, 0])
    mixed = model.forward_mixed(x, task_ids)
    max_dim = max(model.out_dims)
    expected = []
    for i in range(x.size(0)):
        t = int(task_ids[i])
        val = model(x[i : i + 1], t)
        if val.size(1) < max_dim:
            val = torch.cat([val, val.new_zeros(1, max_dim - val.size(1))], dim=1)
        expected.append(val)
    expected = torch.cat(expected, dim=0)
    assert mixed.shape == expected.shape
    assert torch.allclose(mixed, expected, atol=1e-6, rtol=1e-6)


def test_inference_cache_switch_off_is_pure_and_on_reuses():
    model = _model("sparse", cache=True)
    model.eval()
    x = torch.randn(4, 6)
    model.set_inference_cache(False)
    y_off = model(x, 0)
    assert model.cache.size == 0
    model.set_inference_cache(True)
    y_on_1 = model(x, 0)
    y_on_2 = model(x, 0)
    assert model.cache.size > 0
    assert torch.allclose(y_off, y_on_1, atol=1e-6)
    assert torch.allclose(y_on_1, y_on_2, atol=1e-6)


def test_route_table_has_no_per_forward_branching():
    sparse = _model("sparse")
    assert sparse._active_table == ((0,), (1,), (2,))
    pair = _model("pair")
    assert pair._active_table == ((0, 1), (1, 2), (2, 0))
    dense = _model("dense")
    assert dense._active_table == ((0, 1, 2), (0, 1, 2), (0, 1, 2))


def test_forward_rejects_cpu_input_when_model_is_cuda():
    if not torch.cuda.is_available():
        return
    model = _model("sparse").to("cuda")
    try:
        model(torch.randn(2, 6), 0)
    except RuntimeError as exc:
        assert "device" in str(exc)
    else:
        raise AssertionError("CPU input should be rejected for a CUDA model")

def test_forward_mixed_dense_matches_loop():
    model = _model("dense")
    model.eval()
    x = torch.randn(23, 6)
    task_ids = torch.randint(0, 3, (23,))
    mixed = model.forward_mixed(x, task_ids)
    max_dim = max(model.out_dims)
    expected = []
    for i in range(x.size(0)):
        t = int(task_ids[i])
        val = model(x[i : i + 1], t)
        if val.size(1) < max_dim:
            val = torch.cat([val, val.new_zeros(1, max_dim - val.size(1))], dim=1)
        expected.append(val)
    expected = torch.cat(expected, dim=0)
    assert torch.allclose(mixed, expected, atol=1e-6, rtol=1e-6)
