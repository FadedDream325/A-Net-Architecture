"""Gradient-conflict audit contract."""

from __future__ import annotations

import torch

from experiments.anet_v2.core.gradient_audit import shared_gradient_audit
from experiments.anet_v2.core.models import ANetModel
from experiments.anet_v2.core.tasks import build_suite


def test_shared_gradient_audit_reports_pairwise_cosines():
    torch.manual_seed(0)
    suite = build_suite(2026, "low", n_train=96, n_val=32, n_test=32)
    model = ANetModel(3, 8, 16, [2, 1, 5], share_mode="block", cache_enabled=False, active_mode="pair")
    audit = shared_gradient_audit(model, suite, max_samples=32)
    assert audit["shared"] is True
    assert audit["shared_block_id"] == "w0.cross"
    assert len(audit["task_grad_norms"]) == 3
    assert all(n > 0 for n in audit["task_grad_norms"])
    assert audit["num_pairs"] == 3
    assert audit["mean_pairwise_cosine"] is not None
    assert -1.0 <= audit["mean_pairwise_cosine"] <= 1.0


def test_no_share_audit_is_not_applicable():
    torch.manual_seed(0)
    suite = build_suite(2026, "low", n_train=32, n_val=16, n_test=16)
    model = ANetModel(3, 8, 16, [2, 1, 5], share_mode="none", cache_enabled=False, active_mode="pair")
    audit = shared_gradient_audit(model, suite, max_samples=16)
    assert audit["shared"] is False
