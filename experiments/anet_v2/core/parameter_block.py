"""Parameter ownership, identity and versioning for A-Net v2.

The central rule: a parameter block has exactly one owner and exactly one
registered object.  Other workflows receive a *grant* and call that same
object; they never receive a copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    owner: str
    in_features: int
    out_features: int
    kind: str = "linear_relu"
    cacheable: bool = True
    stateful: bool = False
    note: str = ""


class ParameterBlock(nn.Module):
    """A named, owned parameter block with an explicit monotonic version."""

    def __init__(self, spec: BlockSpec, op: nn.Module):
        super().__init__()
        if not spec.block_id:
            raise ValueError("block_id must be non-empty")
        self.spec = spec
        self.op = op
        self._version = 0
        self._forward_calls = 0
        self._last_input_shape: tuple[int, ...] | None = None

    @property
    def block_id(self) -> str:
        return self.spec.block_id

    @property
    def owner(self) -> str:
        return self.spec.owner

    @property
    def version(self) -> int:
        return self._version

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def macs(self) -> int:
        return self.spec.in_features * self.spec.out_features

    @property
    def flops(self) -> int:
        return 2 * self.macs

    @property
    def forward_calls(self) -> int:
        return self._forward_calls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._forward_calls += 1
        self._last_input_shape = tuple(x.shape)
        return self.op(x)

    def mark_updated(self) -> int:
        """Call after any parameter update (optimizer step or manual edit)."""
        self._version += 1
        return self._version

    def param_version_token(self) -> tuple[tuple[int, int], ...]:
        """Tensor-level version token, useful to detect unmarked in-place edits."""
        return tuple((id(p), int(p._version)) for p in self.parameters())

    def identity_report(self) -> dict:
        return {
            "block_id": self.block_id,
            "owner": self.owner,
            "object_id": id(self),
            "version": self.version,
            "num_params": self.num_params,
            "param_data_ptrs": [int(p.data_ptr()) for p in self.parameters()],
            "param_versions": list(self.param_version_token()),
            "kind": self.spec.kind,
            "cacheable": self.spec.cacheable,
            "stateful": self.spec.stateful,
        }

    def extra_repr(self) -> str:
        return (
            f"id={self.block_id}, owner={self.owner}, version={self.version}, "
            f"params={self.num_params}, kind={self.spec.kind}"
        )


@dataclass
class Grant:
    block_id: str
    workflow_id: str
    note: str = ""


class ParameterRegistry:
    """One storage location per block id, plus explicit use grants."""

    def __init__(self) -> None:
        self._blocks: dict[str, ParameterBlock] = {}
        self._owners: dict[str, str] = {}
        self._grants: dict[str, set[str]] = {}
        self._object_to_id: dict[int, str] = {}

    def register(self, block: ParameterBlock, owner: str | None = None) -> ParameterBlock:
        block_id = block.block_id
        if block_id in self._blocks:
            raise ValueError(f"duplicate block_id: {block_id}")
        if id(block) in self._object_to_id:
            other = self._object_to_id[id(block)]
            raise ValueError(f"same object registered twice as {other!r} and {block_id!r}")
        owner = owner or block.owner
        self._blocks[block_id] = block
        self._owners[block_id] = owner
        self._grants[block_id] = {owner}
        self._object_to_id[id(block)] = block_id
        return block

    def grant(self, block_id: str, workflow_id: str, note: str = "") -> Grant:
        if block_id not in self._blocks:
            raise KeyError(f"unknown block {block_id}")
        self._grants[block_id].add(workflow_id)
        return Grant(block_id=block_id, workflow_id=workflow_id, note=note)

    def can_call(self, block_id: str, workflow_id: str) -> bool:
        return workflow_id in self._grants.get(block_id, set())

    def require_call(self, block_id: str, workflow_id: str) -> ParameterBlock:
        if not self.can_call(block_id, workflow_id):
            raise PermissionError(
                f"workflow {workflow_id!r} has no grant for block {block_id!r}"
            )
        return self._blocks[block_id]

    def get(self, block_id: str) -> ParameterBlock:
        return self._blocks[block_id]

    def owner_of(self, block_id: str) -> str:
        return self._owners[block_id]

    def grants_for(self, block_id: str) -> set[str]:
        return set(self._grants[block_id])

    def block_ids(self) -> list[str]:
        return list(self._blocks)

    def blocks(self) -> Iterable[ParameterBlock]:
        return self._blocks.values()

    def num_blocks(self) -> int:
        return len(self._blocks)

    def unique_parameter_count(self) -> int:
        return sum(block.num_params for block in self._blocks.values())

    def reference_parameter_count(self, call_sites: Iterable[tuple[str, str]]) -> int:
        """Sum parameter elements over every call site; repeated calls count again."""
        total = 0
        for block_id, _caller in call_sites:
            total += self._blocks[block_id].num_params
        return total

    def mark_all_updated(self) -> dict[str, int]:
        out = {}
        for block_id, block in self._blocks.items():
            out[block_id] = block.mark_updated()
        return out

    def identity_report(self) -> dict:
        return {
            "num_blocks": self.num_blocks(),
            "unique_params": self.unique_parameter_count(),
            "blocks": [block.identity_report() for block in self._blocks.values()],
            "owners": dict(self._owners),
            "grants": {k: sorted(v) for k, v in self._grants.items()},
        }
