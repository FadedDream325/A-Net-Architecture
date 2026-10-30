"""Unified training, evaluation, and metrics for A-Net v2."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .models import ANetModel
from .tasks import SuiteData, TaskSpec


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_steps: int | None = None
    grad_clip: float | None = 5.0
    check_grad_steps: int = 3


@dataclass
class EvalResult:
    per_task: list[dict]
    mean: dict
    predictions: list[torch.Tensor] | None = None


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_for_task(logits: torch.Tensor, target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "regression":
        return F.mse_loss(logits.squeeze(-1), target.float())
    return F.cross_entropy(logits, target.long())


def train_model(
    model: torch.nn.Module,
    suite: SuiteData,
    config: TrainConfig,
    *,
    device: torch.device,
    log_every: int = 0,
) -> dict:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    model.train()
    steps = 0
    losses_by_task = [[] for _ in suite.specs]
    start = time.perf_counter()
    stop = False

    for epoch in range(config.epochs):
        for task_id, spec in enumerate(suite.specs):
            x = suite.x_train
            y = suite.y_train[task_id]
            permutation = torch.randperm(x.size(0), device=device)
            for offset in range(0, x.size(0), config.batch_size):
                idx = permutation[offset : offset + config.batch_size]
                if isinstance(model, ANetModel):
                    model.begin_step(steps)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[idx], task_id)
                loss = _loss_for_task(logits, y[idx], spec.task_type)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss epoch={epoch} task={task_id} step={steps}"
                    )
                loss.backward()
                if steps < config.check_grad_steps:
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise FloatingPointError(
                                f"non-finite grad epoch={epoch} task={task_id} step={steps}"
                            )
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                if isinstance(model, ANetModel):
                    model.mark_updated()
                losses_by_task[task_id].append(float(loss.detach()))
                steps += 1
                if log_every and steps % log_every == 0:
                    print(f"  step {steps} task {task_id} loss {float(loss.detach()):.5f}", flush=True)
                if config.max_steps is not None and steps >= config.max_steps:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return {
        "optimizer": "AdamW",
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "epochs_requested": config.epochs,
        "steps": steps,
        "seconds": seconds,
        "last_loss_per_task": [
            float(np.mean(v[-20:])) if v else None for v in losses_by_task
        ],
        "loss_curve": losses_by_task,
    }


def _binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.long()
    target = target.long()
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _multiclass_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    pred = pred.long()
    target = target.long()
    accuracy = float((pred == target).float().mean())
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (target == c)).sum())
        fp = int(((pred == c) & (target != c)).sum())
        fn = int(((pred != c) & (target == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1s))}


def _regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    mse = float(F.mse_loss(pred, target))
    mae = float(F.l1_loss(pred, target))
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    suite: SuiteData,
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> EvalResult:
    model.eval()
    xs = {"train": suite.x_train, "val": suite.x_val, "test": suite.x_test}[split]
    ys = {"train": suite.y_train, "val": suite.y_val, "test": suite.y_test}[split]
    per_task = []
    for task_id, spec in enumerate(suite.specs):
        x = xs
        y = ys[task_id]
        if max_samples is not None and x.size(0) > max_samples:
            x = x[:max_samples]
            y = y[:max_samples]
        logits = model(x, task_id)
        if spec.task_type == "regression":
            metrics = _regression_metrics(logits, y)
        else:
            pred = logits.argmax(dim=1)
            if spec.task_type == "binary":
                metrics = _binary_metrics(pred, y)
            else:
                metrics = _multiclass_metrics(pred, y, spec.out_dim)
        metrics.update({"task_id": task_id, "task_key": spec.key, "task_type": spec.task_type})
        per_task.append(metrics)

    def mean_of(key: str):
        vals = [m[key] for m in per_task if key in m]
        return float(np.mean(vals)) if vals else None

    mean = {
        "accuracy": mean_of("accuracy"),
        "f1": mean_of("f1"),
        "macro_f1": mean_of("macro_f1"),
        "mse": mean_of("mse"),
        "mae": mean_of("mae"),
        "r2": mean_of("r2"),
    }
    return EvalResult(per_task=per_task, mean=mean)


@torch.no_grad()
def evaluate_mixed(
    model: ANetModel,
    suite: SuiteData,
    *,
    batch_size: int = 512,
) -> dict:
    model.eval()
    x = suite.x_test
    task_ids = torch.cat(
        [torch.full((x.size(0),), t, dtype=torch.long, device=x.device) for t in range(len(suite.specs))]
    )
    x_all = x.repeat(len(suite.specs), 1)
    preds = model.forward_mixed(x_all, task_ids)
    offset = 0
    per_task = []
    for task_id, spec in enumerate(suite.specs):
        y = suite.y_test[task_id]
        out = preds[offset : offset + x.size(0), : spec.out_dim]
        offset += x.size(0)
        if spec.task_type == "regression":
            metrics = _regression_metrics(out, y)
        else:
            pred = out.argmax(dim=1)
            metrics = _binary_metrics(pred, y) if spec.task_type == "binary" else _multiclass_metrics(pred, y, spec.out_dim)
        metrics.update({"task_id": task_id, "task_key": spec.key, "task_type": spec.task_type})
        per_task.append(metrics)
    return {"per_task": per_task}
