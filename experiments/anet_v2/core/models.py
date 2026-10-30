"""A-Net v2 model zoo and the six controlled architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .accounting import ForwardAccounting
from .parameter_block import BlockSpec, ParameterBlock, ParameterRegistry
from .result_cache import (
    CacheKey,
    ResultCache,
    config_fingerprint,
    tensor_fingerprint,
    tensor_identity_key,
)
from .workflow import CallSite, PreL1, TerminalL1, Workflow, WorkflowGraph


MODEL_KEYS = (
    "independent",
    "shared",
    "anet_basic",
    "anet_share",
    "anet_reuse",
    "anet_full",
    "anet_adapter",
    "anet_full_adapter",
)

MODEL_LABELS = {
    "independent": "Independent",
    "shared": "Shared Backbone",
    "anet_basic": "A-Net Basic (no share/cache)",
    "anet_share": "A-Net Parameter Sharing",
    "anet_reuse": "A-Net Result Reuse",
    "anet_full": "A-Net Full (share+cache)",
    "anet_adapter": "A-Net Share + Adapter",
    "anet_full_adapter": "A-Net Full + Adapter",
}


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class IndependentBaselineV2(nn.Module):
    def __init__(self, num_tasks: int, in_dim: int, hidden_dim: int, out_dims: list[int]):
        super().__init__()
        self.num_tasks = num_tasks
        self.mlps = nn.ModuleList(
            [_mlp(in_dim, hidden_dim, out_dims[t]) for t in range(num_tasks)]
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"invalid task_id {task_id}")
        return self.mlps[task_id](x)

    def unique_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SharedBackboneV2(nn.Module):
    def __init__(self, num_tasks: int, in_dim: int, hidden_dim: int, out_dims: list[int]):
        super().__init__()
        self.num_tasks = num_tasks
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, out_dims[t]) for t in range(num_tasks)])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"invalid task_id {task_id}")
        return self.heads[task_id](self.trunk(x))

    def unique_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ANetModel(nn.Module):
    """Fixed-call-graph A-Net with explicit ownership and optional cache.

    Call graph for one active workflow w:
        x -> pre-L1 -> h0
        h0 -> cross_block[w] (shared or owned) -> s_w
        h0 -> own_block[w] -> u_w
        z_w = u_w + s_w (+ adapter_w(s_w) if enabled)
        z_w -> terminal L1 -> task logits
    """

    def __init__(
        self,
        num_tasks: int,
        in_dim: int,
        hidden_dim: int,
        out_dims: list[int],
        *,
        workflow_out_dim: int | None = None,
        share_mode: str = "none",
        cache_enabled: bool = False,
        active_mode: str = "sparse",
        adapter_rank: int = 8,
        cache_max_entries: int = 4096,
        cache_max_bytes: int = 256 * 1024 * 1024,
        cache_key_mode: str = "identity",
    ):
        super().__init__()
        if share_mode not in {"none", "block", "adapter"}:
            raise ValueError(f"bad share_mode {share_mode}")
        if active_mode not in {"sparse", "pair", "dense"}:
            raise ValueError(f"bad active_mode {active_mode}")
        self.num_tasks = num_tasks
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dims = list(out_dims)
        self.workflow_out_dim = workflow_out_dim or max(1, hidden_dim // 2)
        self.share_mode = share_mode
        self.active_mode = active_mode
        self.adapter_rank = adapter_rank
        if cache_key_mode not in {"identity", "content"}:
            raise ValueError(f"bad cache_key_mode {cache_key_mode}")
        self.cache_key_mode = cache_key_mode
        self.registry = ParameterRegistry()
        self.blocks = nn.ModuleDict()
        self.workflow_objs: list[Workflow] = []
        self.own_ids: list[str] = []
        self.cross_ids: list[str] = []
        self.adapter_ids: list[str] = []
        self.cache = ResultCache(
            enabled=cache_enabled,
            max_entries=cache_max_entries,
            max_bytes=cache_max_bytes,
        )
        self.accounting = ForwardAccounting()
        self._step = 0
        self._last_terminal_params = 0
        self._last_terminal_flops = 0
        self.last_accounting: dict = {}
        self.last_active_ids: list[int] = []

        self.prel1_op = PreL1(in_dim, hidden_dim)
        self.prel1_id = "prel1"
        self._add_block(
            BlockSpec(
                block_id=self.prel1_id,
                owner="L1_pre",
                in_features=in_dim,
                out_features=hidden_dim,
                kind="prel1",
            ),
            self.prel1_op,
        )

        e = self.workflow_out_dim
        shared_id = "w0.cross"
        for t in range(num_tasks):
            wf = Workflow(workflow_id=f"w{t}")
            own_id = f"w{t}.own"
            own_op = nn.Sequential(nn.Linear(hidden_dim, e), nn.ReLU())
            self._add_block(
                BlockSpec(own_id, f"w{t}", hidden_dim, e, kind="own"), own_op
            )
            wf.own(own_id)
            self.own_ids.append(own_id)

            if share_mode == "none":
                cross_id = f"w{t}.cross"
                cross_op = nn.Sequential(nn.Linear(hidden_dim, e), nn.ReLU())
                self._add_block(
                    BlockSpec(cross_id, f"w{t}", hidden_dim, e, kind="cross"), cross_op
                )
                wf.own(cross_id)
            else:
                cross_id = shared_id
                if t == 0:
                    cross_op = nn.Sequential(nn.Linear(hidden_dim, e), nn.ReLU())
                    self._add_block(
                        BlockSpec(cross_id, "w0", hidden_dim, e, kind="shared_cross"), cross_op
                    )
                    wf.own(cross_id)
                else:
                    wf.use(cross_id)
                self.registry.grant(cross_id, f"w{t}", note="shared cross-block")
            self.cross_ids.append(cross_id)

            if share_mode == "adapter":
                adapter_id = f"w{t}.adapter"
                r = max(1, min(adapter_rank, e))
                adapter_op = nn.Sequential(
                    nn.Linear(e, r), nn.ReLU(), nn.Linear(r, e)
                )
                self._add_block(
                    BlockSpec(adapter_id, f"w{t}", e, e, kind="adapter"), adapter_op
                )
                wf.own(adapter_id)
                self.adapter_ids.append(adapter_id)
            else:
                self.adapter_ids.append("")

            self.workflow_objs.append(wf)

        self.terminal = TerminalL1(num_tasks, e, out_dims)
        self.graph = self._build_graph()
        self.graph.assert_acyclic()

    @staticmethod
    def _module_key(block_id: str) -> str:
        return block_id.replace(".", "__")

    def _add_block(self, spec: BlockSpec, op: nn.Module) -> ParameterBlock:
        block = ParameterBlock(spec, op)
        self.registry.register(block)
        key = self._module_key(spec.block_id)
        if key in self.blocks:
            raise ValueError(f"duplicate module key {key}")
        self.blocks[key] = block
        return block

    def _build_graph(self) -> WorkflowGraph:
        graph = WorkflowGraph()
        graph.add_site(
            CallSite("prel1", "L1_pre", self.prel1_id, "input"),
            depends_on=["input"],
        )
        for t, wf in enumerate(self.workflow_objs):
            own_site = f"w{t}.own"
            cross_site = f"w{t}.cross"
            graph.add_site(
                CallSite(own_site, wf.workflow_id, self.own_ids[t], "prel1"),
                depends_on=["prel1"],
            )
            graph.add_site(
                CallSite(cross_site, wf.workflow_id, self.cross_ids[t], "prel1"),
                depends_on=["prel1"],
            )
            out_site = f"w{t}.output"
            graph.add_site(
                CallSite(out_site, wf.workflow_id, "workflow_output", "workflow"),
                depends_on=[own_site, cross_site],
            )
        graph.add_site(
            CallSite("terminal", "L1_terminal", "terminal", "workflow_outputs"),
            depends_on=[f"w{t}.output" for t in range(self.num_tasks)],
        )
        return graph

    def resolve_active(self, task_id: int, active_ids: list[int] | None = None) -> list[int]:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"invalid task_id {task_id}")
        if active_ids is None:
            if self.active_mode == "sparse":
                return [task_id]
            if self.active_mode == "pair":
                return [task_id, (task_id + 1) % self.num_tasks]
            return list(range(self.num_tasks))
        active = list(dict.fromkeys(int(i) for i in active_ids))
        if not active:
            raise ValueError("active_ids must be non-empty")
        for i in active:
            if not 0 <= i < self.num_tasks:
                raise ValueError(f"invalid active workflow {i}")
        return active

    def _config(self, block: ParameterBlock) -> dict:
        return {
            "op_version": 1,
            "training": bool(self.training),
            "dtype": str(next(block.parameters()).dtype),
            "device": str(next(block.parameters()).device),
            "kind": block.spec.kind,
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "param_token": block.param_version_token(),
        }

    def _call_block(
        self,
        block_id: str,
        x: torch.Tensor,
        caller: str,
        kind: str = "regular",
        *,
        use_cache: bool | None = None,
    ) -> torch.Tensor:
        block = self.registry.require_call(block_id, caller) if caller.startswith("w") else self.registry.get(block_id)
        cache_on = self.cache.enabled if use_cache is None else use_cache
        if not cache_on:
            out = block(x)
            self.accounting.record(block, caller, executed=True, cached=False, kind=kind)
            return out
        if self.cache_key_mode == "content":
            input_hash, hash_us = tensor_fingerprint(x)
            self.cache.stats["hash_us"] += hash_us
        else:
            input_hash = tensor_identity_key(x)
        key = CacheKey(
            block_id=block.block_id,
            version=block.version,
            input_hash=input_hash,
            config_hash=config_fingerprint(self._config(block)),
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.accounting.record(block, caller, executed=False, cached=True, kind=kind)
            return cached
        out = block(x)
        self.cache.put(key, out)
        self.accounting.record(block, caller, executed=True, cached=False, kind=kind)
        return out

    def forward(
        self,
        x: torch.Tensor,
        task_id: int,
        active_ids: list[int] | None = None,
    ) -> torch.Tensor:
        active = self.resolve_active(task_id, active_ids)
        self.last_active_ids = active
        self.accounting.reset()
        h0 = self._call_block(self.prel1_id, x, "L1_pre", kind="prel1")
        zs: list[torch.Tensor] = []
        for w in active:
            cross = self._call_block(self.cross_ids[w], h0, f"w{w}", kind="cross")
            own = self._call_block(self.own_ids[w], h0, f"w{w}", kind="own")
            s = cross
            if self.share_mode == "adapter":
                s = s + self._call_block(
                    self.adapter_ids[w], s, f"w{w}", kind="adapter"
                )
            zs.append(own + s)
        logits = self.terminal(zs, task_id, active)
        head = self.terminal.heads[str(task_id)]
        self._last_terminal_params = sum(p.numel() for p in head.parameters()) + self.terminal.gates.numel()
        self._last_terminal_flops = 2 * head.in_features * head.out_features
        self.last_accounting = self.accounting.summary(
            extra_params=self._last_terminal_params,
            extra_flops_per_sample=self._last_terminal_flops,
        )
        return logits

    def forward_mixed(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate a mixed-task batch; preserves the input order.

        Tasks have different output widths, so outputs are padded on the right
        to max(out_dims).  The padding is zero and is ignored by per-task
        evaluation.
        """
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("x and task_ids batch sizes differ")
        max_dim = max(self.out_dims)
        ordered: list[torch.Tensor | None] = [None] * x.shape[0]
        for t in range(self.num_tasks):
            mask = task_ids == t
            if not mask.any():
                continue
            val = self.forward(x[mask], t)
            if val.size(1) < max_dim:
                pad = val.new_zeros(val.size(0), max_dim - val.size(1))
                val = torch.cat([val, pad], dim=1)
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            for row, pos in enumerate(idx.tolist()):
                ordered[pos] = val[row]
        return torch.stack(ordered, dim=0)

    def begin_step(self, step: int | None = None) -> None:
        if self.training:
            self._step = step if step is not None else self._step + 1
            self.cache.begin_step(self._step)

    def mark_updated(self) -> None:
        self.registry.mark_all_updated()
        self.cache.clear()

    def train(self, mode: bool = True):
        changed = bool(mode) != bool(self.training)
        out = super().train(mode)
        self.cache.detach_on_store = not mode
        if changed:
            self.cache.clear()
        return out

    def unique_parameter_count(self) -> int:
        return self.registry.unique_parameter_count() + self.terminal.parameter_count()

    def terminal_parameter_count(self) -> int:
        return self.terminal.parameter_count()

    def adapter_parameter_count(self) -> int:
        return sum(
            self.registry.get(i).num_params for i in self.adapter_ids if i
        )

    def cost_summary(self) -> dict:
        terminal = self.terminal_parameter_count()
        unique = self.unique_parameter_count()
        accounting = dict(self.last_accounting)
        if accounting:
            accounting["active_params_per_sample"] = (
                accounting.get("active_params_per_sample", 0) + self._last_terminal_params
            )
            accounting["theoretical_flops_per_sample"] = (
                accounting.get("theoretical_flops_per_sample", 0) + self._last_terminal_flops
            )
            accounting["actual_flops_per_sample"] = (
                accounting.get("actual_flops_per_sample", 0) + self._last_terminal_flops
            )
            accounting["reference_params"] = (
                accounting.get("reference_params", 0) + terminal
            )
        return {
            "unique_params": unique,
            "workflow_unique_params": self.registry.unique_parameter_count(),
            "terminal_params": terminal,
            "adapter_params": self.adapter_parameter_count(),
            "adapter_params_total": self.adapter_parameter_count(),
            "adapter_params_executed": accounting.get("adapter_params", 0),
            "share_mode": self.share_mode,
            "cache_enabled": self.cache.enabled,
            "cache_key_mode": self.cache_key_mode,
            "active_mode": self.active_mode,
            "graph": self.graph.summary(),
            "accounting": accounting,
            "cache": self.cache.summary(),
            "identity": {
                "num_blocks": self.registry.num_blocks(),
                "shared_cross_blocks": sorted(
                    {i for i in self.cross_ids if self.cross_ids.count(i) > 1}
                ),
                "owners": {i: self.registry.owner_of(i) for i in self.registry.block_ids()},
            },
        }


def build_model(
    kind: str,
    *,
    num_tasks: int,
    in_dim: int,
    hidden_dim: int,
    out_dims: list[int],
    workflow_out_dim: int | None = None,
    active_mode: str = "sparse",
    cache_key_mode: str = "identity",
) -> nn.Module:
    if kind == "independent":
        return IndependentBaselineV2(num_tasks, in_dim, hidden_dim, out_dims)
    if kind == "shared":
        return SharedBackboneV2(num_tasks, in_dim, hidden_dim, out_dims)
    if kind == "anet_basic":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="none",
                         cache_enabled=False, active_mode=active_mode, cache_key_mode=cache_key_mode)
    if kind == "anet_share":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="block",
                         cache_enabled=False, active_mode=active_mode, cache_key_mode=cache_key_mode)
    if kind == "anet_reuse":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="none",
                         cache_enabled=True, active_mode=active_mode, cache_key_mode=cache_key_mode)
    if kind == "anet_full":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="block",
                         cache_enabled=True, active_mode=active_mode, cache_key_mode=cache_key_mode)
    if kind == "anet_adapter":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="adapter",
                         cache_enabled=False, active_mode=active_mode, cache_key_mode=cache_key_mode)
    if kind == "anet_full_adapter":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="adapter",
                         cache_enabled=True, active_mode=active_mode, cache_key_mode=cache_key_mode)
    raise KeyError(f"unknown model {kind}")
