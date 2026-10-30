"""Data-generation tests for the controlled scaling task suite."""

from __future__ import annotations

import numpy as np
import torch

from experiments.scaling.tasks import TASKS, build_task_suite, make_task_arrays


def test_suite_shapes_balance_and_clean_test_labels():
    train, test, meta = build_task_suite(2026, n_train=512, n_test=128)
    assert len(train) == len(test) == len(TASKS) == 3
    for i, ((x_tr, y_tr), (x_te, y_te)) in enumerate(zip(train, test)):
        assert x_tr.shape == (512, 2) and y_tr.shape == (512,)
        assert x_te.shape == (128, 2) and y_te.shape == (128,)
        assert x_tr.dtype == torch.float32 and y_tr.dtype == torch.int64
        assert 0.35 < float((y_tr == 1).float().mean()) < 0.65
        assert 0.35 < float((y_te == 1).float().mean()) < 0.65

        # Test labels are generated without noise, so the exact formulas must hold.
        x_np = x_te.numpy()
        y_np = y_te.numpy()
        if i == 0:
            expected = (x_np[:, 1] > 0.8 * np.sin(2.0 * x_np[:, 0])).astype(np.int64)
        elif i == 1:
            expected = (
                x_np[:, 1] > 0.55 * np.sin(3.0 * x_np[:, 0])
                + 0.28 * np.sin(7.5 * x_np[:, 0])
            ).astype(np.int64)
        else:
            theta = np.deg2rad(25.0)
            u = x_np[:, 0] * np.cos(theta) + x_np[:, 1] * np.sin(theta)
            v = -x_np[:, 0] * np.sin(theta) + x_np[:, 1] * np.cos(theta)
            expected = (
                (np.floor((u + 4.0) / 1.1) + np.floor((v + 4.0) / 1.1)).astype(
                    np.int64
                )
                % 2
            )
        assert np.array_equal(y_np, expected)

    assert len(meta["tasks"]) == 3
    assert meta["note"].startswith("Label noise is applied to train only")


def test_generation_is_deterministic_for_same_seed():
    train_a, test_a, meta_a = build_task_suite(7, n_train=64, n_test=32)
    train_b, test_b, meta_b = build_task_suite(7, n_train=64, n_test=32)
    for (xa, ya), (xb, yb) in zip(train_a, train_b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    for (xa, ya), (xb, yb) in zip(test_a, test_b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert meta_a == meta_b


def test_injected_noise_changes_some_train_labels_but_not_test():
    x_tr, y_tr = make_task_arrays(
        "easy_sine", 2000, np.random.default_rng(0), noise=0.05
    )
    y_clean = (x_tr[:, 1] > 0.8 * np.sin(2.0 * x_tr[:, 0])).astype(np.int64)
    changed = int(np.sum(y_tr != y_clean))
    assert 40 <= changed <= 160  # ~5% of 2,000, with generous binomial bounds
