"""Per-task gradient conflict audit on a shared parameter block."""

from __future__ import annotations

import math
from itertools import combinations

import torch
import torch.nn.functional as F

from .models import ANetModel
from .tasks import SuiteData


def _loss(logits: torch.Tensor, target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "regression":
        return F.mse_loss(logits.squeeze(-1), target.float())
    return F.cross_entropy(logits, target.long())


def shared_gradient_audit(
    model: ANetModel,
    suite: SuiteData,
    *,
    max_samples: int = 512,
) -> dict:
    """Compute per-task gradients of the shared cross block.

    The model is put in train mode but no optimizer step is taken.  The same
    input batch is used for every task so the comparison isolates task-target
    conflict rather than input variation.
    """
    shared_ids = [i for i in set(model.cross_ids) if model.cross_ids.count(i) > 1]
    if not shared_ids:
        return {"shared": False, "reason": "no cross-workflow shared block"}
    shared_id = shared_ids[0]
    block = model.registry.get(shared_id)
    params = [p for p in block.parameters() if p.requires_grad]

    was_training = model.training
    model.train()
    x = suite.x_train[: min(max_samples, suite.x_train.size(0))]
    grads: list[torch.Tensor] = []
    norms: list[float] = []
    losses: list[float] = []
    for task_id, spec in enumerate(suite.specs):
        model.zero_grad(set_to_none=True)
        logits = model(x, task_id)
        loss = _loss(logits, suite.y_train[task_id][: x.size(0)], spec.task_type)
        loss.backward()
        flat = torch.cat([
            (p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1))
            for p in params
        ])
        grads.append(flat)
        norms.append(float(flat.norm()))
        losses.append(float(loss.detach()))

    cosine = [[None for _ in suite.specs] for _ in suite.specs]
    values = []
    for i, j in combinations(range(len(grads)), 2):
        a, b = grads[i], grads[j]
        denom = float(a.norm() * b.norm())
        c = float(torch.dot(a, b) / denom) if denom > 1e-12 else 0.0
        c = max(-1.0, min(1.0, c))
        cosine[i][j] = cosine[j][i] = c
        values.append(c)

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()

    return {
        "shared": True,
        "shared_block_id": shared_id,
        "shared_params": block.num_params,
        "task_grad_norms": norms,
        "task_losses": losses,
        "cosine_matrix": cosine,
        "mean_pairwise_cosine": sum(values) / len(values) if values else None,
        "min_pairwise_cosine": min(values) if values else None,
        "conflict_fraction": sum(1 for v in values if v < 0.0) / len(values) if values else None,
        "num_pairs": len(values),
    }
