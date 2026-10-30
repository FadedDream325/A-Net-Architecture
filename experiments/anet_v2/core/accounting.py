"""Five distinct accounting quantities, never collapsed into one number.

- reference_params: parameter elements referenced by all call sites, with
  repeated calls counted repeatedly.
- unique_params: distinct stored parameter elements.
- active_params: distinct parameter elements that participate in one forward
  pass (a shared block counted once even if called several times).
- theoretical_flops: per-sample analytical count if nothing is cached.
- actual_flops / actual_executions: counts after cache hits are removed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .parameter_block import ParameterBlock


@dataclass
class BlockCall:
    block_id: str
    caller: str
    executed: bool
    cached: bool
    params: int
    flops: int
    kind: str = "regular"


@dataclass
class ForwardAccounting:
    calls: list[BlockCall] = field(default_factory=list)
    block_meta: dict[str, dict] = field(default_factory=dict)

    def record(
        self,
        block: ParameterBlock,
        caller: str,
        *,
        executed: bool,
        cached: bool,
        kind: str = "regular",
    ) -> None:
        self.calls.append(
            BlockCall(
                block_id=block.block_id,
                caller=caller,
                executed=executed,
                cached=cached,
                params=block.num_params,
                flops=block.flops,
                kind=kind,
            )
        )
        self.block_meta[block.block_id] = {
            "owner": block.owner,
            "params": block.num_params,
            "flops": block.flops,
            "kind": kind,
        }

    def summary(self, extra_params: int = 0, extra_flops_per_sample: int = 0) -> dict:
        refs = sum(c.params for c in self.calls)
        unique_ids = {c.block_id for c in self.calls}
        active = sum(self.block_meta[b]["params"] for b in unique_ids)
        theoretical = sum(c.flops for c in self.calls) + extra_flops_per_sample
        executions = Counter()
        for c in self.calls:
            if c.executed:
                executions[c.block_id] += 1
        actual = sum(
            self.block_meta[b]["flops"] * n for b, n in executions.items()
        ) + extra_flops_per_sample
        executed_params = sum(
            self.block_meta[b]["params"] * n for b, n in executions.items()
        )
        adapter_ids = {
            c.block_id for c in self.calls if self.block_meta[c.block_id]["kind"] == "adapter"
        }
        adapter_params = sum(self.block_meta[b]["params"] for b in adapter_ids)
        adapter_execs = sum(executions[b] for b in adapter_ids)
        adapter_flops = sum(self.block_meta[b]["flops"] * executions[b] for b in adapter_ids)
        per_block = defaultdict(lambda: {"calls": 0, "executions": 0, "hits": 0})
        for c in self.calls:
            entry = per_block[c.block_id]
            entry["calls"] += 1
            if c.executed:
                entry["executions"] += 1
            if c.cached:
                entry["hits"] += 1
        return {
            "reference_params": refs,
            "active_params_per_sample": active,
            "unique_params_called": active,
            "theoretical_flops_per_sample": theoretical,
            "theoretical_executions_per_sample": len(self.calls),
            "actual_executions_per_sample": sum(executions.values()),
            "actual_flops_per_sample": actual,
            "executed_parameter_elements_per_sample": executed_params,
            "unique_blocks_called": len(unique_ids),
            "adapter_params": adapter_params,
            "adapter_executions": adapter_execs,
            "adapter_flops": adapter_flops,
            "extra_params": extra_params,
            "extra_flops_per_sample": extra_flops_per_sample,
            "per_block": dict(per_block),
        }

    def reset(self) -> None:
        self.calls.clear()
        self.block_meta.clear()


class AveragedAccounting:
    """Accumulate accounting over batches, then report per-sample values."""

    def __init__(self) -> None:
        self.batches = 0
        self.sums: dict[str, float] = defaultdict(float)
        self.per_block: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, summary: dict) -> None:
        self.batches += 1
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                self.sums[key] += float(value)
            elif key == "per_block":
                for block_id, stats in value.items():
                    for stat, number in stats.items():
                        self.per_block[block_id][stat] += float(number)

    def mean(self) -> dict:
        if self.batches == 0:
            return {}
        out = {key: value / self.batches for key, value in self.sums.items()}
        out["per_block"] = {
            block_id: {k: v / self.batches for k, v in stats.items()}
            for block_id, stats in self.per_block.items()
        }
        out["batches"] = self.batches
        return out
