"""Data contract: deterministic splits, no overlap, controlled correlation."""

from __future__ import annotations

import numpy as np
import torch

from experiments.anet_v2.core.tasks import CORRELATIONS, build_suite


def _row_hashes(x: torch.Tensor) -> set[bytes]:
    arr = x.numpy()
    return {row.tobytes() for row in arr}


def test_shapes_splits_and_no_overlap():
    suite = build_suite(2026, "medium", n_train=300, n_val=100, n_test=100)
    assert suite.x_train.shape == (300, 8)
    assert suite.x_val.shape == (100, 8)
    assert suite.x_test.shape == (100, 8)
    assert _row_hashes(suite.x_train).isdisjoint(_row_hashes(suite.x_val))
    assert _row_hashes(suite.x_train).isdisjoint(_row_hashes(suite.x_test))
    assert _row_hashes(suite.x_val).isdisjoint(_row_hashes(suite.x_test))
    for i, spec in enumerate(suite.specs):
        assert suite.y_train[i].shape == (300,)
        assert suite.y_val[i].shape == (100,)
        assert suite.y_test[i].shape == (100,)
        assert suite.y_test[i].dtype == (torch.float32 if spec.task_type == "regression" else torch.int64)


def test_correlation_levels_are_ordered():
    values = {}
    for level in ("low", "medium", "high"):
        suite = build_suite(2026, level, n_train=512, n_val=128, n_test=128)
        values[level] = suite.meta["latent_corr_abs_offdiag_mean"]
    assert values["low"] < values["medium"] < values["high"]
    assert values["high"] > 0.75
    assert values["low"] < 0.2


def test_multiclass_classes_are_balanced():
    suite = build_suite(2026, "medium", n_train=1000, n_val=100, n_test=100)
    counts = np.asarray(suite.meta["tasks"][2]["train_class_counts"])
    assert counts.min() > 0
    assert counts.min() / counts.max() > 0.6


def test_binary_threshold_is_balanced():
    suite = build_suite(2026, "high", n_train=1000, n_val=100, n_test=100)
    rate = suite.meta["tasks"][0]["train_positive_rate"]
    assert 0.45 < rate < 0.55


def test_same_seed_is_deterministic_and_new_seed_changes_data():
    a = build_suite(7, "medium", n_train=128, n_val=32, n_test=32)
    b = build_suite(7, "medium", n_train=128, n_val=32, n_test=32)
    c = build_suite(8, "medium", n_train=128, n_val=32, n_test=32)
    assert torch.equal(a.x_train, b.x_train)
    assert torch.equal(a.y_train[0], b.y_train[0])
    assert not torch.equal(a.x_train, c.x_train)
    assert a.meta["hashes"] == b.meta["hashes"]


def test_meta_records_hashes_and_noise_policy():
    suite = build_suite(2026, "low", n_train=64, n_val=16, n_test=16)
    meta = suite.meta
    assert meta["note"].startswith("Noise is applied to train only")
    assert len(meta["hashes"]["y_train"]) == 3
    assert "x_test" in meta["hashes"]
