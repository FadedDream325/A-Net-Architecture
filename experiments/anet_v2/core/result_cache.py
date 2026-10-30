"""Result cache with identity/version/input/config keys.

Correctness rules:
- same block object (id), same parameter version, same input content, same op
  configuration are required for a hit;
- training cache is cleared at every optimizer step and never persists across
  steps;
- cached tensors keep their autograd graph in training, so all consumers can
  backpropagate through the same computation;
- evaluation entries may be detached to save memory.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def tensor_identity_key(x: torch.Tensor) -> str:
    """Cheap same-object key for within-graph reuse.

    This is deliberately not a content hash: two different tensors with equal
    values will miss.  It is safe for the common case where a shared block is
    called several times with the exact same tensor object.  In-place edits are
    caught by the tensor version counter.
    """
    try:
        version = int(x._version)
    except RuntimeError:
        # Inference tensors do not track a version counter; their identity is
        # still safe for within-forward reuse.
        version = -1
    return (
        f"ptr={int(x.data_ptr())};ver={version};"
        f"shape={tuple(x.shape)};stride={tuple(x.stride())};"
        f"offset={int(x.storage_offset())};dtype={x.dtype};device={x.device}"
    )


def tensor_fingerprint(x: torch.Tensor) -> tuple[str, float]:
    """Content hash. Returns (hex digest, elapsed microseconds).

    This is intentionally a full-content hash for v1 correctness.  The caller
    records the hash cost so cache break-even can be measured rather than
    assumed.
    """
    start = time.perf_counter_ns()
    t = x.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    t = t.contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(tuple(t.shape)).encode("ascii"))
    h.update(str(t.dtype).encode("ascii"))
    h.update(t.numpy().tobytes())
    elapsed_us = (time.perf_counter_ns() - start) / 1000.0
    return h.hexdigest(), elapsed_us


@dataclass(frozen=True)
class CacheKey:
    block_id: str
    version: int
    input_hash: str
    config_hash: str


@dataclass
class CacheEntry:
    output: torch.Tensor
    nbytes: int
    step: int


class ResultCache:
    def __init__(
        self,
        enabled: bool = True,
        max_entries: int = 4096,
        max_bytes: int = 256 * 1024 * 1024,
        detach_on_store: bool = False,
    ) -> None:
        self.enabled = enabled
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.detach_on_store = detach_on_store
        self._entries: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._bytes = 0
        self._step = 0
        self.stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "stores": 0,
            "clears": 0,
            "lookup_us": 0.0,
            "hash_us": 0.0,
            "bytes": 0,
        }

    @property
    def size(self) -> int:
        return len(self._entries)

    def begin_step(self, step: int) -> None:
        """Training boundary: no result may survive an optimizer step."""
        self._step = step
        self.clear()

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
        self.stats["clears"] += 1
        self.stats["bytes"] = 0

    def invalidate_block(self, block_id: str) -> int:
        keys = [k for k in self._entries if k.block_id == block_id]
        for key in keys:
            entry = self._entries.pop(key)
            self._bytes -= entry.nbytes
            self.stats["evictions"] += 1
        self.stats["bytes"] = self._bytes
        return len(keys)

    def _evict(self) -> None:
        while self._entries and (
            len(self._entries) > self.max_entries or self._bytes > self.max_bytes
        ):
            _, entry = self._entries.popitem(last=False)
            self._bytes -= entry.nbytes
            self.stats["evictions"] += 1
        self.stats["bytes"] = self._bytes

    def get(self, key: CacheKey) -> torch.Tensor | None:
        if not self.enabled:
            return None
        start = time.perf_counter_ns()
        self.stats["lookups"] += 1
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            self.stats["hits"] += 1
            self.stats["lookup_us"] += (time.perf_counter_ns() - start) / 1000.0
            return entry.output
        self.stats["misses"] += 1
        self.stats["lookup_us"] += (time.perf_counter_ns() - start) / 1000.0
        return None

    def put(self, key: CacheKey, output: torch.Tensor) -> None:
        if not self.enabled:
            return
        value = output.detach() if (self.detach_on_store or not output.requires_grad) else output
        nbytes = value.numel() * value.element_size()
        if key in self._entries:
            old = self._entries.pop(key)
            self._bytes -= old.nbytes
        self._entries[key] = CacheEntry(output=value, nbytes=nbytes, step=self._step)
        self._entries.move_to_end(key)
        self._bytes += nbytes
        self.stats["stores"] += 1
        self.stats["bytes"] = self._bytes
        self._evict()

    def get_or_compute(
        self,
        key: CacheKey,
        compute: Callable[[], torch.Tensor],
        *,
        detach: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        cached = self.get(key)
        if cached is not None:
            return cached, True
        value = compute()
        if detach:
            value = value.detach()
        self.put(key, value)
        return value, False

    def summary(self) -> dict:
        lookups = max(self.stats["lookups"], 1)
        return {
            **self.stats,
            "enabled": self.enabled,
            "size": self.size,
            "bytes": self._bytes,
            "hit_rate": self.stats["hits"] / lookups,
            "miss_rate": self.stats["misses"] / lookups,
            "lookup_us_mean": self.stats["lookup_us"] / lookups,
        }
