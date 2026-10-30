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
        num_workflows: int | None = None,
        workflow_out_dim: int | None = None,
        share_mode: str = "none",
        cache_enabled: bool = False,
        active_mode: str = "sparse",
        adapter_rank: int = 8,
        cache_max_entries: int = 4096,
        cache_max_bytes: int = 256 * 1024 * 1024,
        cache_key_mode: str = "identity",
        gradient_isolation: str = "none",
    ):
        super().__init__()
        if share_mode not in {"none", "block", "adapter"}:
            raise ValueError(f"bad share_mode {share_mode}")
        if active_mode not in {"sparse", "pair", "dense"}:
            raise ValueError(f"bad active_mode {active_mode}")
        if gradient_isolation not in {"none", "detach_aux", "detach_shared", "detach_both"}:
            raise ValueError(f"bad gradient_isolation {gradient_isolation}")
        self.num_tasks = num_tasks
        self.num_workflows = num_workflows or num_tasks
        if self.num_workflows < 1:
            raise ValueError("num_workflows must be >= 1")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dims = list(out_dims)
        self.workflow_out_dim = workflow_out_dim or max(1, hidden_dim // 2)
        self.share_mode = share_mode
        self.active_mode = active_mode
        self.gradient_isolation = gradient_isolation
        self._detach_aux = gradient_isolation in {"detach_aux", "detach_both"}
        self._detach_shared = gradient_isolation in {"detach_shared", "detach_both"}
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
        for t in range(self.num_workflows):
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

        self.terminal = TerminalL1(num_tasks, e, out_dims, num_workflows=self.num_workflows)
        # Hot-path tables: routing and block references are resolved once.
        self._active_table = tuple(self._make_active(t) for t in range(num_tasks))
        self._prel1_block = self.registry.get(self.prel1_id)
        self._own_blocks = [self.registry.get(i) for i in self.own_ids]
        self._cross_blocks = [self.registry.get(i) for i in self.cross_ids]
        self._adapter_blocks = [self.registry.get(i) if i else None for i in self.adapter_ids]
        self._has_adapters = bool(self.adapter_ids and any(self.adapter_ids))
        self._caller_names = [f"w{t}" for t in range(self.num_workflows)]
        self.inference_cache = False
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
            depends_on=[f"w{t}.output" for t in range(self.num_workflows)],
        )
        return graph

    def _make_active(self, task_id: int) -> tuple[int, ...]:
        primary = task_id % self.num_workflows
        if self.active_mode == "sparse":
            return (primary,)
        if self.active_mode == "pair":
            if self.num_workflows == 1:
                return (0,)
            return (primary, (primary + 1) % self.num_workflows)
        return tuple(range(self.num_workflows))

    def resolve_active(self, task_id: int, active_ids: list[int] | None = None) -> list[int]:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"invalid task_id {task_id}")
        if active_ids is None:
            return list(self._active_table[task_id])
        active = list(dict.fromkeys(int(i) for i in active_ids))
        if not active:
            raise ValueError("active_ids must be non-empty")
        for i in active:
            if not 0 <= i < self.num_workflows:
                raise ValueError(f"invalid active workflow {i}")
        return active

    def set_inference_cache(self, enabled: bool) -> None:
        """Switch for single-request inference.

        False is the default: an eval forward then runs the pure matrix-multiply
        path with no cache key construction or lookup.
        """
        self.inference_cache = bool(enabled)

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

    def _record(self, block: ParameterBlock, caller: str, kind: str, executed: bool, cached: bool) -> None:
        self.accounting.record(block, caller, executed=executed, cached=cached, kind=kind)

    def _call_cached(
        self,
        block: ParameterBlock,
        caller: str,
        kind: str,
        x: torch.Tensor,
        track: bool,
    ) -> torch.Tensor:
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
            if track:
                self._record(block, caller, kind, executed=False, cached=True)
            return cached
        out = block(x)
        self.cache.put(key, out)
        if track:
            self._record(block, caller, kind, executed=True, cached=False)
        return out

    def _finish_tracking(self, task_id: int) -> None:
        head = self.terminal.heads[task_id]
        self._last_terminal_params = sum(p.numel() for p in head.parameters()) + self.terminal.gates.numel()
        self._last_terminal_flops = 2 * head.in_features * head.out_features
        self.last_accounting = self.accounting.summary(
            extra_params=self._last_terminal_params,
            extra_flops_per_sample=self._last_terminal_flops,
        )

    def _forward_uncached(
        self,
        x: torch.Tensor,
        task_id: int,
        active: tuple[int, ...] | list[int],
        track: bool,
    ) -> torch.Tensor:
        if track:
            self.accounting.reset()
        h0 = self._prel1_block(x)
        if track:
            self._record(self._prel1_block, "L1_pre", "prel1", executed=True, cached=False)
        zs: list[torch.Tensor] = []
        if self._has_adapters:
            for w in active:
                cross = self._cross_blocks[w](h0)
                if self._detach_shared:
                    cross = cross.detach()
                own = self._own_blocks[w](h0)
                s = cross + self._adapter_blocks[w](cross)
                z = own + s
                if self._detach_aux and w != task_id:
                    z = z.detach()
                zs.append(z)
                if track:
                    self._record(self._cross_blocks[w], self._caller_names[w], "cross", True, False)
                    self._record(self._adapter_blocks[w], self._caller_names[w], "adapter", True, False)
                    self._record(self._own_blocks[w], self._caller_names[w], "own", True, False)
        else:
            for w in active:
                cross = self._cross_blocks[w](h0)
                if self._detach_shared:
                    cross = cross.detach()
                own = self._own_blocks[w](h0)
                z = own + cross
                if self._detach_aux and w != task_id:
                    z = z.detach()
                zs.append(z)
                if track:
                    self._record(self._cross_blocks[w], self._caller_names[w], "cross", True, False)
                    self._record(self._own_blocks[w], self._caller_names[w], "own", True, False)
        logits = self.terminal(zs, task_id, active)
        if track:
            self._finish_tracking(task_id)
        return logits

    def _forward_cached(
        self,
        x: torch.Tensor,
        task_id: int,
        active: tuple[int, ...] | list[int],
        track: bool,
    ) -> torch.Tensor:
        if track:
            self.accounting.reset()
        h0 = self._call_cached(self._prel1_block, "L1_pre", "prel1", x, track)
        zs: list[torch.Tensor] = []
        if self._has_adapters:
            for w in active:
                cross = self._call_cached(self._cross_blocks[w], self._caller_names[w], "cross", h0, track)
                if self._detach_shared:
                    cross = cross.detach()
                own = self._call_cached(self._own_blocks[w], self._caller_names[w], "own", h0, track)
                s = cross + self._call_cached(self._adapter_blocks[w], self._caller_names[w], "adapter", cross, track)
                z = own + s
                if self._detach_aux and w != task_id:
                    z = z.detach()
                zs.append(z)
        else:
            for w in active:
                cross = self._call_cached(self._cross_blocks[w], self._caller_names[w], "cross", h0, track)
                if self._detach_shared:
                    cross = cross.detach()
                own = self._call_cached(self._own_blocks[w], self._caller_names[w], "own", h0, track)
                z = own + cross
                if self._detach_aux and w != task_id:
                    z = z.detach()
                zs.append(z)
        logits = self.terminal(zs, task_id, active)
        if track:
            self._finish_tracking(task_id)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        task_id: int,
        active_ids: list[int] | None = None,
        *,
        track: bool = True,
        use_cache: bool | None = None,
    ) -> torch.Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"invalid task_id {task_id}")
        active = self._active_table[task_id] if active_ids is None else tuple(self.resolve_active(task_id, active_ids))
        self.last_active_ids = list(active)
        param_device = next(self.parameters()).device
        if x.device != param_device:
            raise RuntimeError(
                f"input device {x.device} does not match model device {param_device}; "
                "inference is GPU-only when the model is on CUDA"
            )
        cache_on = self.cache.enabled if use_cache is None else bool(use_cache)
        if cache_on and (self.training or self.inference_cache):
            return self._forward_cached(x, task_id, active, track)
        return self._forward_uncached(x, task_id, active, track)

    def forward_mixed(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Group by Task ID with torch.argsort, compute contiguous groups, then
        invert the permutation.  No per-sample Python mask or warp divergence.
        """
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("x and task_ids batch sizes differ")
        if x.device != task_ids.device:
            raise RuntimeError("x and task_ids must be on the same device")
        order = torch.argsort(task_ids, stable=True)
        inverse = torch.argsort(order)
        sorted_x = x.index_select(0, order)
        sorted_t = task_ids.index_select(0, order)
        unique, counts = torch.unique_consecutive(sorted_t, return_counts=True)
        max_dim = max(self.out_dims)
        outs: list[torch.Tensor] = []
        start = 0
        for i in range(unique.numel()):
            t = int(unique[i].item())
            n = int(counts[i].item())
            end = start + n
            val = self.forward(sorted_x[start:end], t)
            if val.size(1) < max_dim:
                val = torch.cat([val, val.new_zeros(val.size(0), max_dim - val.size(1))], dim=1)
            outs.append(val)
            start = end
        out = torch.cat(outs, dim=0)
        return out.index_select(0, inverse)

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
            "gradient_isolation": self.gradient_isolation,
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
    num_workflows: int | None = None,
    workflow_out_dim: int | None = None,
    active_mode: str = "sparse",
    cache_key_mode: str = "identity",
    gradient_isolation: str = "none",
) -> nn.Module:
    if kind == "independent":
        return IndependentBaselineV2(num_tasks, in_dim, hidden_dim, out_dims)
    if kind == "shared":
        return SharedBackboneV2(num_tasks, in_dim, hidden_dim, out_dims)
    if kind == "anet_basic":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="none",
                         cache_enabled=False, active_mode=active_mode, cache_key_mode=cache_key_mode, gradient_isolation=gradient_isolation, num_workflows=num_workflows)
    if kind == "anet_share":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="block",
                         cache_enabled=False, active_mode=active_mode, cache_key_mode=cache_key_mode, gradient_isolation=gradient_isolation, num_workflows=num_workflows)
    if kind == "anet_reuse":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="none",
                         cache_enabled=True, active_mode=active_mode, cache_key_mode=cache_key_mode, gradient_isolation=gradient_isolation, num_workflows=num_workflows)
    if kind == "anet_full":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="block",
                         cache_enabled=True, active_mode=active_mode, cache_key_mode=cache_key_mode, gradient_isolation=gradient_isolation, num_workflows=num_workflows)
    if kind == "anet_adapter":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="adapter",
                         cache_enabled=False, active_mode=active_mode, cache_key_mode=cache_key_mode, gradient_isolation=gradient_isolation, num_workflows=num_workflows)
    if kind == "anet_full_adapter":
        return ANetModel(num_tasks, in_dim, hidden_dim, out_dims,
                         workflow_out_dim=workflow_out_dim, share_mode="adapter",
                         cache_enabled=True, active_mode=active_mode, cache_key_mode=cache_key_mode, gradient_isolation=gradient_isolation, num_workflows=num_workflows)
    raise KeyError(f"unknown model {kind}")
