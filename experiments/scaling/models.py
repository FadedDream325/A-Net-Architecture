"""Model zoo and cost accounting for the A-Net scaling study.

The current A-Net implementation is imported from the repository root instead
of being duplicated, so the scaling study measures the code that actually ships
in multitask_anet_demo.py.

The only new architecture here is ANetCrossLayer: a minimal prototype in which
every task expert directly calls the same parameter block object at a fixed
location. The shared block is owned by the parent model exactly once; it is not
copied into each expert. This avoids duplicate parameter registration and makes
gradient sharing testable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitask_anet_demo import (  # noqa: E402
    ANetHardRouting,
    IndependentBaseline,
    SharedTrunkWithHeads,
)

MODEL_KEYS = ("independent", "shared", "anet", "cross", "cross_off")

MODEL_LABELS = {
    "independent": "Independent",
    "shared": "Shared trunk + head",
    "anet": "A-Net (current)",
    "cross": "A-Net + cross-layer block",
    "cross_off": "A-Net + block (disabled)",
}


class CrossLayerExpert(nn.Module):
    """Expert that can directly call an externally owned shared block."""

    def __init__(self, hidden_dim: int, expert_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim)
        self.fc2 = nn.Linear(expert_hidden_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        shared_block: nn.Module | None = None,
    ) -> torch.Tensor:
        z = F.relu(self.fc1(x))
        if shared_block is not None:
            # Direct parameter-block call, not a hidden-state-only handoff.
            # The same module object is supplied by the parent to every task.
            z = z + shared_block(z)
        return self.fc2(z)


class ANetCrossLayer(nn.Module):
    """A-Net with one shared parameter block called from every task path.

    Ownership:
        self.shared_block is registered once on this model.
    Call graph:
        x -> shared L1 -> expert[t].fc1 -> shared_block -> expert[t].fc2
    Ablation:
        setting use_cross_layer=False leaves the parameters registered but skips
        the direct call, allowing a same-parameter-count compute ablation.
    """

    def __init__(
        self,
        num_tasks: int = 3,
        hidden_dim: int = 128,
        expert_hidden_dim: int = 64,
        use_cross_layer: bool = True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_cross_layer = use_cross_layer
        self.l1 = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.shared_block = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.ReLU(),
        )
        self.experts = nn.ModuleList(
            [
                CrossLayerExpert(hidden_dim, expert_hidden_dim)
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"task_id must be in [0, {self.num_tasks}), got {task_id}")
        h = self.l1(x)
        block = self.shared_block if self.use_cross_layer else None
        return self.experts[task_id](h, block)


def build_model(kind: str, hidden_dim: int, expert_hidden_dim: int) -> nn.Module:
    if kind == "independent":
        return IndependentBaseline(hidden_dim=hidden_dim)
    if kind == "shared":
        return SharedTrunkWithHeads(hidden_dim=hidden_dim)
    if kind == "anet":
        return ANetHardRouting(
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
        )
    if kind == "cross":
        return ANetCrossLayer(
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
        )
    if kind == "cross_off":
        return ANetCrossLayer(
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            use_cross_layer=False,
        )
    raise KeyError(f"unknown model kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analytical_parameters(kind: str, hidden_dim: int, expert_hidden_dim: int) -> int:
    h = hidden_dim
    e = expert_hidden_dim
    if kind == "independent":
        return 3 * (h * h + 6 * h + 2)
    if kind == "shared":
        return h * h + 10 * h + 6
    if kind == "anet":
        return 3 * h + 3 * h * e + 9 * e + 6
    if kind in {"cross", "cross_off"}:
        return 3 * h + 3 * h * e + 9 * e + 6 + e * e + e
    raise KeyError(kind)


def active_parameters(model: nn.Module, kind: str) -> int:
    if kind == "independent":
        return count_parameters(model.mlps[0])
    if kind == "shared":
        return count_parameters(model)
    if kind == "anet":
        return count_parameters(model.l1) + count_parameters(model.experts[0])
    if kind == "cross":
        return (
            count_parameters(model.l1)
            + count_parameters(model.experts[0])
            + count_parameters(model.shared_block)
        )
    if kind == "cross_off":
        return count_parameters(model.l1) + count_parameters(model.experts[0])
    raise KeyError(kind)


def analytical_macs(kind: str, hidden_dim: int, expert_hidden_dim: int) -> int:
    """Forward MACs for one sample, ignoring biases and activation functions.

    independent / shared: 2*h (input) + h*h (hidden) + 2*h (output)
    A-Net:                2*h (L1) + h*e (expert fc1) + 2*e (expert fc2)
    cross:                A-Net + e*e (shared block)
    """
    h = hidden_dim
    e = expert_hidden_dim
    if kind in {"independent", "shared"}:
        return h * h + 4 * h
    if kind == "anet":
        return 2 * h + h * e + 2 * e
    if kind == "cross":
        return 2 * h + h * e + 2 * e + e * e
    if kind == "cross_off":
        return 2 * h + h * e + 2 * e
    raise KeyError(kind)


def analytical_flops(kind: str, hidden_dim: int, expert_hidden_dim: int) -> int:
    return 2 * analytical_macs(kind, hidden_dim, expert_hidden_dim)


def model_spec_rows(
    kind: str,
    hidden_dim: int,
    expert_hidden_dim: int,
    model: nn.Module,
) -> dict:
    total = count_parameters(model)
    expected = analytical_parameters(kind, hidden_dim, expert_hidden_dim)
    if total != expected:
        raise AssertionError(
            f"{kind}: actual params {total} != analytical {expected}"
        )
    active = active_parameters(model, kind)
    macs = analytical_macs(kind, hidden_dim, expert_hidden_dim)
    return {
        "model": kind,
        "hidden_dim": hidden_dim,
        "expert_hidden_dim": expert_hidden_dim,
        "unique_trainable_parameters": total,
        "active_parameters_per_sample": active,
        "macs_per_sample": macs,
        "flops_per_sample": 2 * macs,
        "active_parameter_fraction": active / total,
        "macs_fraction_vs_independent": macs
        / analytical_macs("independent", hidden_dim, expert_hidden_dim),
    }
