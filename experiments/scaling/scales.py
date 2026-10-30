"""Pre-registered model scales for the A-Net scaling experiment.

Width is kept identical across architectures within a scale. A-Net's total
parameter count is the scale anchor; the other models' counts are reported,
not forced to match. This preserves each architecture's natural parameter
allocation while allowing a controlled width/depth scaling comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ScaleSpec:
    key: str
    hidden_dim: int
    expert_hidden_dim: int
    target_anet_params: str
    note: str


SCALES: dict[str, ScaleSpec] = {
    "small": ScaleSpec(
        key="small",
        hidden_dim=256,
        expert_hidden_dim=128,
        target_anet_params="~100k",
        note="A-Net total = 100,230 parameters; independent = 201,222",
    ),
    "medium": ScaleSpec(
        key="medium",
        hidden_dim=896,
        expert_hidden_dim=448,
        target_anet_params="~1.2M",
        note="A-Net total = 1,210,950 parameters; independent = 2,424,582",
    ),
    "large": ScaleSpec(
        key="large",
        hidden_dim=1792,
        expert_hidden_dim=896,
        target_anet_params="~4.8M",
        note="A-Net total = 4,830,342 parameters; independent = 9,666,054",
    ),
}


def get_scale(key: str) -> ScaleSpec:
    if key not in SCALES:
        raise KeyError(f"unknown scale {key}; choose from {sorted(SCALES)}")
    return SCALES[key]


def scale_config(key: str) -> dict:
    return asdict(get_scale(key))
