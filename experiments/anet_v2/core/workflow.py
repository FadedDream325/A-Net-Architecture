"""Workflow call graph, pre-L1, and terminal integration.

The graph is declarative: the model builds call sites and edges, then checks
that the graph is acyclic.  Terminal L1 receives only workflow-level features;
it has no reference to the pre-L1 tensor or to a workflow's internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CallSite:
    site_id: str
    caller: str
    block_id: str
    input_slot: str
    config_key: str = "default"


@dataclass
class Workflow:
    workflow_id: str
    owned_block_ids: list[str] = field(default_factory=list)
    used_block_ids: list[str] = field(default_factory=list)
    call_sites: list[CallSite] = field(default_factory=list)

    def own(self, block_id: str) -> None:
        if block_id not in self.owned_block_ids:
            self.owned_block_ids.append(block_id)
        if block_id not in self.used_block_ids:
            self.used_block_ids.append(block_id)

    def use(self, block_id: str) -> None:
        if block_id not in self.used_block_ids:
            self.used_block_ids.append(block_id)

    def add_call(self, site: CallSite) -> None:
        if site.caller != self.workflow_id:
            raise ValueError(f"call site {site.site_id} caller mismatch")
        self.call_sites.append(site)


class WorkflowGraph:
    def __init__(self) -> None:
        self.sites: dict[str, CallSite] = {}
        self.edges: set[tuple[str, str]] = set()

    def add_site(self, site: CallSite, depends_on: list[str] | None = None) -> None:
        if site.site_id in self.sites:
            raise ValueError(f"duplicate call site {site.site_id}")
        self.sites[site.site_id] = site
        for parent in depends_on or []:
            if parent not in self.sites and parent != "input":
                raise KeyError(f"unknown dependency {parent}")
            self.edges.add((parent, site.site_id))

    def topological_order(self) -> list[str]:
        nodes = set(self.sites) | {"input"}
        indegree = {node: 0 for node in nodes}
        children: dict[str, list[str]] = {node: [] for node in nodes}
        for parent, child in self.edges:
            indegree[child] += 1
            children[parent].append(child)
        queue = [node for node, degree in indegree.items() if degree == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(nodes):
            cycle_nodes = [node for node, degree in indegree.items() if degree > 0]
            raise ValueError(f"workflow graph contains a cycle: {cycle_nodes}")
        return order

    def assert_acyclic(self) -> None:
        self.topological_order()

    def dependencies_of(self, site_id: str) -> set[str]:
        return {parent for parent, child in self.edges if child == site_id}

    def terminal_inputs(self) -> set[str]:
        return {
            parent
            for parent, child in self.edges
            if child.startswith("terminal") and parent != "input"
        }

    def summary(self) -> dict:
        return {
            "num_sites": len(self.sites),
            "num_edges": len(self.edges),
            "topological_order": self.topological_order(),
            "terminal_inputs": sorted(self.terminal_inputs()),
        }


class PreL1(nn.Module):
    """Input processing.  The block itself is registered by the model."""

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.op = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class TerminalL1(nn.Module):
    """Integrate only workflow-level outputs into per-task predictions.

    The forward signature intentionally accepts a list of workflow outputs, not
    a hidden-state dictionary or the pre-L1 tensor.  Tests hook this input to
    verify that no intermediate state is passed in.
    """

    def __init__(self, num_tasks: int, feature_dim: int, out_dims: list[int], num_workflows: int | None = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_workflows = num_workflows or num_tasks
        self.feature_dim = feature_dim
        self.out_dims = list(out_dims)
        self.heads = nn.ModuleList(
            [nn.Linear(feature_dim, out_dims[t]) for t in range(num_tasks)]
        )
        # Per-task soft weights over workflow slots; masked to active workflows.
        self.gates = nn.Parameter(torch.zeros(num_tasks, self.num_workflows))

    def forward(
        self,
        workflow_outputs: list[torch.Tensor],
        task_id: int,
        active_ids: list[int],
    ) -> torch.Tensor:
        if not workflow_outputs:
            raise ValueError("terminal L1 needs at least one workflow output")
        if len(workflow_outputs) != len(active_ids):
            raise ValueError("workflow output count != active id count")
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"invalid task_id {task_id}")
        if len(workflow_outputs) == 1:
            # Fast path: stacking/softmax/pooling over K=1 only adds kernels.
            pooled = workflow_outputs[0]
        else:
            stacked = torch.stack(workflow_outputs, dim=1)  # [B, K, e]
            idx = torch.as_tensor(active_ids, device=stacked.device, dtype=torch.long)
            logits = self.gates[task_id].index_select(0, idx)
            weights = F.softmax(logits, dim=0).to(stacked.dtype)  # [K]
            pooled = torch.einsum("k,bke->be", weights, stacked)
        return self.heads[task_id](pooled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
