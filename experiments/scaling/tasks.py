"""Controlled multi-task suite for A-Net scaling experiments.

Three 2D binary tasks spanning easy / medium / hard difficulty:

0. easy_sine     -- low-frequency sine boundary, correlated with task 1
1. medium_sine   -- multi-frequency sine boundary, correlated with task 0
2. hard_checker  -- rotated checkerboard, uncorrelated with the sine tasks

Train labels optionally contain label noise; test labels are clean. All tasks
use the same 2D input domain and are generated once per seed so every model
sees exactly the same train/test tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TaskSpec:
    key: str
    name: str
    family: str
    difficulty: str
    train_noise: float
    note: str


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="easy_sine",
        name="Easy Sine",
        family="sine",
        difficulty="easy",
        train_noise=0.01,
        note="x2 > 0.8*sin(2*x1); correlated with medium sine",
    ),
    TaskSpec(
        key="medium_sine",
        name="Medium Sine",
        family="sine",
        difficulty="medium",
        train_noise=0.02,
        note="x2 > 0.55*sin(3*x1) + 0.28*sin(7.5*x1); correlated with easy sine",
    ),
    TaskSpec(
        key="hard_checker",
        name="Hard Checker",
        family="checkerboard",
        difficulty="hard",
        train_noise=0.02,
        note="25-degree rotated 1.1-wide checkerboard; uncorrelated with sine tasks",
    ),
)


def _flip_labels(y: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    if rate <= 0:
        return y
    mask = rng.random(y.shape[0]) < rate
    out = y.copy()
    out[mask] = 1 - out[mask]
    return out


def make_task_arrays(
    task_key: str,
    n: int,
    rng: np.random.Generator,
    *,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one task. Returns float32 x and int64 y."""
    x = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)

    if task_key == "easy_sine":
        threshold = 0.8 * np.sin(2.0 * x[:, 0])
        y = (x[:, 1] > threshold).astype(np.int64)
    elif task_key == "medium_sine":
        threshold = 0.55 * np.sin(3.0 * x[:, 0]) + 0.28 * np.sin(7.5 * x[:, 0])
        y = (x[:, 1] > threshold).astype(np.int64)
    elif task_key == "hard_checker":
        theta = math.radians(25.0)
        u = x[:, 0] * math.cos(theta) + x[:, 1] * math.sin(theta)
        v = -x[:, 0] * math.sin(theta) + x[:, 1] * math.cos(theta)
        cell = 1.1
        y = (
            np.floor((u + 4.0) / cell) + np.floor((v + 4.0) / cell)
        ).astype(np.int64) % 2
    else:
        raise KeyError(f"unknown task: {task_key}")

    y = _flip_labels(y, rng, noise)
    return x, y


def build_task_suite(
    seed: int,
    n_train: int = 12_000,
    n_test: int = 3_000,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """Build deterministic train/test tensors for all tasks for one seed."""
    train_rng = np.random.default_rng(seed + 100_000)
    test_rng = np.random.default_rng(seed + 900_000)

    train: list[tuple[torch.Tensor, torch.Tensor]] = []
    test: list[tuple[torch.Tensor, torch.Tensor]] = []
    task_stats: list[dict] = []

    for spec in TASKS:
        x_tr, y_tr = make_task_arrays(
            spec.key, n_train, train_rng, noise=spec.train_noise
        )
        # Test labels are clean (no injected label noise).
        x_te, y_te = make_task_arrays(spec.key, n_test, test_rng, noise=0.0)
        train.append((torch.from_numpy(x_tr), torch.from_numpy(y_tr)))
        test.append((torch.from_numpy(x_te), torch.from_numpy(y_te)))
        task_stats.append(
            {
                "key": spec.key,
                "name": spec.name,
                "family": spec.family,
                "difficulty": spec.difficulty,
                "train_noise": spec.train_noise,
                "n_train": n_train,
                "n_test": n_test,
                "train_positive_rate": float((y_tr == 1).mean()),
                "test_positive_rate": float((y_te == 1).mean()),
            }
        )

    meta = {
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "tasks": task_stats,
        "note": "Label noise is applied to train only; test labels are clean.",
    }
    return train, test, meta
