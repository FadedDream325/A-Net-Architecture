"""Controlled synthetic suite: binary classification, regression, multiclass.

Three tasks share one input distribution and one shared latent component.  The
correlation level rho controls how much of each task's latent representation is
shared.  Train labels/targets may contain noise; validation and test are clean.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch


TASK_TYPES = ("binary", "regression", "multiclass")
OUT_DIMS = {"binary": 2, "regression": 1, "multiclass": 5}
CORRELATIONS = {"high": 0.95, "medium": 0.70, "low": 0.20}
CORRELATION_ORDER = ("low", "medium", "high")


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    key: str
    task_type: str
    out_dim: int
    display_name: str


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(0, "binary_nonlinear", "binary", 2, "Nonlinear binary"),
    TaskSpec(1, "regression_nonlinear", "regression", 1, "Nonlinear regression"),
    TaskSpec(2, "multiclass_nonlinear", "multiclass", 5, "Nonlinear 5-class"),
)


def hash_array(a: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(a.shape).encode("ascii"))
    h.update(str(a.dtype).encode("ascii"))
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@dataclass
class SuiteData:
    specs: list[TaskSpec]
    x_train: torch.Tensor
    x_val: torch.Tensor
    x_test: torch.Tensor
    y_train: list[torch.Tensor]
    y_val: list[torch.Tensor]
    y_test: list[torch.Tensor]
    meta: dict

    @property
    def num_tasks(self) -> int:
        return len(self.specs)

    def to(self, device: torch.device) -> "SuiteData":
        return SuiteData(
            specs=self.specs,
            x_train=self.x_train.to(device),
            x_val=self.x_val.to(device),
            x_test=self.x_test.to(device),
            y_train=[y.to(device) for y in self.y_train],
            y_val=[y.to(device) for y in self.y_val],
            y_test=[y.to(device) for y in self.y_test],
            meta=self.meta,
        )


def _latent_corr(us: list[np.ndarray]) -> list[list[float]]:
    feats = [u - u.mean(axis=0, keepdims=True) for u in us]
    feats = [f / (f.std(axis=0, keepdims=True) + 1e-8) for f in feats]
    mat = np.zeros((len(us), len(us)))
    for i in range(len(us)):
        for j in range(len(us)):
            a = feats[i].reshape(-1)
            b = feats[j].reshape(-1)
            mat[i, j] = float(np.corrcoef(a, b)[0, 1])
    return mat.tolist()


def build_suite(
    seed: int,
    correlation: str,
    *,
    n_train: int = 6000,
    n_val: int = 1500,
    n_test: int = 1500,
    in_dim: int = 8,
    latent_dim: int = 8,
    binary_noise: float = 0.01,
    multiclass_noise: float = 0.01,
    regression_noise: float = 0.05,
) -> SuiteData:
    if correlation not in CORRELATIONS:
        raise KeyError(f"unknown correlation {correlation}; use {sorted(CORRELATIONS)}")
    rho = CORRELATIONS[correlation]
    rng = np.random.default_rng(seed)

    W_shared = rng.normal(0.0, 1.0 / math.sqrt(in_dim), (in_dim, latent_dim))
    W_tasks = [
        rng.normal(0.0, 1.0 / math.sqrt(in_dim), (in_dim, latent_dim))
        for _ in TASKS
    ]
    # Multiclass uses quantile bins of an angular score, which keeps the five
    # classes approximately balanced while the shared latent still controls the
    # boundary.  No class-collapsing random projection is used.
    class_phase = rng.uniform(0.0, 2.0 * math.pi, OUT_DIMS["multiclass"])

    def make_x(n: int) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(n, in_dim)).astype(np.float32)

    x_train = make_x(n_train)
    x_val = make_x(n_val)
    x_test = make_x(n_test)

    def latents(x: np.ndarray) -> list[np.ndarray]:
        z_shared = np.tanh(x @ W_shared)
        out = []
        for W_t in W_tasks:
            z_t = np.tanh(x @ W_t)
            out.append(rho * z_shared + math.sqrt(max(0.0, 1.0 - rho * rho)) * z_t)
        return out

    u_train = latents(x_train)
    u_val = latents(x_val)
    u_test = latents(x_test)

    def binary_scores(u: np.ndarray) -> np.ndarray:
        return (
            np.sin(1.8 * u[:, 0])
            + 0.8 * u[:, 1] ** 2
            - 0.5 * u[:, 2]
            + 0.4 * u[:, 3] * u[:, 4]
        )

    def regression_values(u: np.ndarray) -> np.ndarray:
        return (
            np.sin(1.5 * u[:, 0])
            + 0.6 * u[:, 1] ** 2
            + 0.3 * u[:, 2]
            - 0.2 * u[:, 3]
            + 0.1 * u[:, 0] * u[:, 1]
        ).astype(np.float32)

    def multiclass_score(u: np.ndarray) -> np.ndarray:
        return np.arctan2(u[:, 0], u[:, 1]) + 0.5 * np.sin(2.0 * u[:, 2] + class_phase[0])

    # Binary: threshold from clean training scores only.
    s_train = binary_scores(u_train[0])
    threshold = float(np.median(s_train))
    y_bin_train = (s_train > threshold).astype(np.int64)
    y_bin_val = (binary_scores(u_val[0]) > threshold).astype(np.int64)
    y_bin_test = (binary_scores(u_test[0]) > threshold).astype(np.int64)
    flip = rng.random(y_bin_train.shape[0]) < binary_noise
    y_bin_train = y_bin_train.copy()
    y_bin_train[flip] = 1 - y_bin_train[flip]

    # Regression: train observation noise only; val/test clean.
    y_reg_train = regression_values(u_train[1])
    y_reg_val = regression_values(u_val[1])
    y_reg_test = regression_values(u_test[1])
    y_reg_train = y_reg_train + rng.normal(0.0, regression_noise, size=y_reg_train.shape).astype(np.float32)

    # Multiclass: quantile bins from the clean training score; train flips only.
    mc_train_score = multiclass_score(u_train[2])
    cuts = np.quantile(mc_train_score, np.linspace(0.0, 1.0, OUT_DIMS["multiclass"] + 1)[1:-1])
    y_mc_train = np.digitize(mc_train_score, cuts).astype(np.int64)
    y_mc_val = np.digitize(multiclass_score(u_val[2]), cuts).astype(np.int64)
    y_mc_test = np.digitize(multiclass_score(u_test[2]), cuts).astype(np.int64)
    flip = rng.random(y_mc_train.shape[0]) < multiclass_noise
    y_mc_train = y_mc_train.copy()
    y_mc_train[flip] = (y_mc_train[flip] + 1 + rng.integers(0, OUT_DIMS["multiclass"] - 1, size=flip.sum())) % OUT_DIMS["multiclass"]

    y_train = [y_bin_train, y_reg_train, y_mc_train]
    y_val = [y_bin_val, y_reg_val, y_mc_val]
    y_test = [y_bin_test, y_reg_test, y_mc_test]

    corr = _latent_corr(u_test)
    off_diag = [corr[i][j] for i in range(len(corr)) for j in range(len(corr)) if i != j]
    meta = {
        "seed": seed,
        "correlation": correlation,
        "rho": rho,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "in_dim": in_dim,
        "latent_dim": latent_dim,
        "tasks": [
            {
                "task_id": s.task_id,
                "key": s.key,
                "task_type": s.task_type,
                "out_dim": s.out_dim,
                "train_noise": {
                    "binary": binary_noise,
                    "regression": regression_noise,
                    "multiclass": multiclass_noise,
                }[s.task_type],
                "train_positive_rate": float((y_train[i] == 1).mean()) if s.task_type == "binary" else None,
                "test_positive_rate": float((y_test[i] == 1).mean()) if s.task_type == "binary" else None,
                "train_class_counts": np.bincount(y_train[i].astype(np.int64), minlength=s.out_dim).tolist() if s.task_type != "regression" else None,
                "test_class_counts": np.bincount(y_test[i].astype(np.int64), minlength=s.out_dim).tolist() if s.task_type != "regression" else None,
                "test_target_mean": float(y_test[i].mean()) if s.task_type == "regression" else None,
                "test_target_std": float(y_test[i].std()) if s.task_type == "regression" else None,
            }
            for i, s in enumerate(TASKS)
        ],
        "latent_corr_matrix": corr,
        "latent_corr_abs_offdiag_mean": float(np.mean(np.abs(off_diag))),
        "hashes": {
            "x_train": hash_array(x_train),
            "x_val": hash_array(x_val),
            "x_test": hash_array(x_test),
            "y_train": [hash_array(y) for y in y_train],
            "y_val": [hash_array(y) for y in y_val],
            "y_test": [hash_array(y) for y in y_test],
        },
        "note": "Noise is applied to train only; validation/test are clean. Splits are disjoint by construction.",
    }

    return SuiteData(
        specs=list(TASKS),
        x_train=torch.from_numpy(x_train),
        x_val=torch.from_numpy(x_val),
        x_test=torch.from_numpy(x_test),
        y_train=[torch.from_numpy(y.copy()) for y in y_train],
        y_val=[torch.from_numpy(y.copy()) for y in y_val],
        y_test=[torch.from_numpy(y.copy()) for y in y_test],
        meta=meta,
    )


def out_dims(suite: SuiteData) -> list[int]:
    return [s.out_dim for s in suite.specs]
