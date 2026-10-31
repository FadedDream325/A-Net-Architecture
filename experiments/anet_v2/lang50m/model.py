"""50M char-level pilot: Transformer, Shared Backbone, and A-Net.

This is deliberately a pilot, not a language-capability claim.  It uses
byte-level tokens from three public/local corpora and a fixed small budget so
the whole run fits on the local 8 GB GPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab: int = 256, d: int = 512, layers: int = 16, heads: int = 8, ffn: int = 2000):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.task_emb = nn.Embedding(3, d)
        self.blocks = nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.mask = None

    def _mask(self, T: int, device):
        if self.mask is None or self.mask.size(0) != T or self.mask.device != device:
            self.mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return self.mask

    def forward(self, x: torch.Tensor, task: int) -> torch.Tensor:
        h = self.embed(x) + self.task_emb(torch.full_like(x, task))
        m = self._mask(x.size(1), x.device)
        for block in self.blocks:
            h = block(h, m)
        return self.head(self.ln_f(h))


class SharedBackboneLM(nn.Module):
    def __init__(self, vocab: int = 256, d: int = 512, layers: int = 16, heads: int = 8, ffn: int = 2000, tasks: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.task_emb = nn.Embedding(tasks, d)
        self.blocks = nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, vocab) for _ in range(tasks)])
        self.mask = None

    def _mask(self, T: int, device):
        if self.mask is None or self.mask.size(0) != T or self.mask.device != device:
            self.mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return self.mask

    def forward(self, x: torch.Tensor, task: int) -> torch.Tensor:
        h = self.embed(x) + self.task_emb(torch.full_like(x, task))
        m = self._mask(x.size(1), x.device)
        for block in self.blocks:
            h = block(h, m)
        return self.heads[task](self.ln_f(h))


class ANetLM(nn.Module):
    """Pre-L1 + independent Transformer workflows + shared block + terminal L1."""

    def __init__(self, vocab: int = 256, d: int = 512, heads: int = 8, ffn: int = 2000,
                 pre_blocks: int = 2, workflows: int = 3, blocks_per_workflow: int = 4,
                 shared_blocks: int = 2, tasks: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pre = nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(pre_blocks)])
        self.workflows = nn.ModuleList([
            nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(blocks_per_workflow)])
            for _ in range(workflows)
        ])
        self.shared = nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(shared_blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, vocab) for _ in range(tasks)])
        self.gates = nn.Parameter(torch.zeros(tasks, workflows))
        self.tasks = tasks
        self.workflows_n = workflows
        self.mask = None

    def _mask(self, T: int, device):
        if self.mask is None or self.mask.size(0) != T or self.mask.device != device:
            self.mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return self.mask

    def forward(self, x: torch.Tensor, task: int) -> torch.Tensor:
        h = self.embed(x)
        m = self._mask(x.size(1), x.device)
        for block in self.pre:
            h = block(h, m)
        shared = h
        for block in self.shared:
            shared = block(shared, m)
        primary = task % self.workflows_n
        active = (primary, (primary + 1) % self.workflows_n) if self.workflows_n > 1 else (0,)
        zs = []
        for w in active:
            z = h
            for block in self.workflows[w]:
                z = block(z, m)
            zs.append(z + shared)
        weights = F.softmax(self.gates[task, list(active)], dim=0).to(zs[0].dtype)
        pooled = sum(w * z for w, z in zip(weights, zs))
        return self.heads[task](self.ln_f(pooled))


class SharedDepthANetLM(nn.Module):
    """Shared-depth control C.

    Each task path has 12 sequential blocks: 8 shared blocks + 4 blocks owned
    by one of two workflows.  The unique budget is 16 blocks (~50M), while the
    active path is 12 blocks (~37.5M), matching Transformer-12L in depth and
    active parameters/FLOPs.
    """

    def __init__(self, vocab: int = 256, d: int = 512, heads: int = 8, ffn: int = 2000,
                 shared_blocks: int = 8, workflows: int = 2, own_blocks: int = 4, tasks: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.shared = nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(shared_blocks)])
        self.workflows = nn.ModuleList([
            nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(own_blocks)])
            for _ in range(workflows)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, vocab) for _ in range(tasks)])
        self.workflows_n = workflows
        self.mask = None

    def _mask(self, T: int, device):
        if self.mask is None or self.mask.size(0) != T or self.mask.device != device:
            self.mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return self.mask

    def forward(self, x: torch.Tensor, task: int) -> torch.Tensor:
        h = self.embed(x)
        m = self._mask(x.size(1), x.device)
        for block in self.shared:
            h = block(h, m)
        w = task % self.workflows_n
        for block in self.workflows[w]:
            h = block(h, m)
        return self.heads[task](self.ln_f(h))


class NoTerminalANetLM(nn.Module):
    """A-Net without terminal L1 / cross-workflow integration.

    12-block design: pre 2 + one primary workflow 4 + shared 2.  Only the
    primary workflow for the task is used; the shared stack is added as a
    residual feature.  There is no terminal weighted average or auxiliary
    workflow output.
    """

    def __init__(self, vocab: int = 256, d: int = 512, heads: int = 8, ffn: int = 2000,
                 pre_blocks: int = 2, workflows: int = 3, blocks_per_workflow: int = 4,
                 shared_blocks: int = 2, tasks: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pre = nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(pre_blocks)])
        self.workflows = nn.ModuleList([
            nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(blocks_per_workflow)])
            for _ in range(workflows)
        ])
        self.shared = nn.ModuleList([CausalBlock(d, heads, ffn) for _ in range(shared_blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, vocab) for _ in range(tasks)])
        self.workflows_n = workflows
        self.mask = None

    def _mask(self, T: int, device):
        if self.mask is None or self.mask.size(0) != T or self.mask.device != device:
            self.mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return self.mask

    def forward(self, x: torch.Tensor, task: int) -> torch.Tensor:
        h = self.embed(x)
        m = self._mask(x.size(1), x.device)
        for block in self.pre:
            h = block(h, m)
        shared = h
        for block in self.shared:
            shared = block(shared, m)
        w = task % self.workflows_n
        z = h
        for block in self.workflows[w]:
            z = block(z, m)
        z = z + shared
        return self.heads[task](self.ln_f(z))


def parameter_counts(model: nn.Module) -> dict:
    unique = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shared_params = 0
    active = unique
    reference = unique
    if isinstance(model, ANetLM):
        shared_params = sum(p.numel() for b in model.shared for p in b.parameters())
        # active: pre + 2 workflows + one shared stack + head
        primary = 0
        active = sum(p.numel() for p in model.embed.parameters())
        active += sum(p.numel() for b in model.pre for p in b.parameters())
        for w in (0, 1):
            active += sum(p.numel() for b in model.workflows[w] for p in b.parameters())
        active += shared_params
        active += sum(p.numel() for p in model.heads[0].parameters()) + model.ln_f.weight.numel() * 2
        reference = unique + shared_params  # shared stack referenced by both active workflows
    elif isinstance(model, NoTerminalANetLM):
        shared_params = sum(p.numel() for b in model.shared for p in b.parameters())
        active = sum(p.numel() for p in model.embed.parameters())
        active += sum(p.numel() for b in model.pre for p in b.parameters())
        active += sum(p.numel() for b in model.shared for p in b.parameters())
        active += sum(p.numel() for b in model.workflows[0] for p in b.parameters())
        active += sum(p.numel() for p in model.heads[0].parameters()) + model.ln_f.weight.numel() * 2
        reference = unique
    elif isinstance(model, SharedDepthANetLM):
        shared_params = sum(p.numel() for b in model.shared for p in b.parameters())
        active = sum(p.numel() for p in model.embed.parameters())
        active += shared_params
        active += sum(p.numel() for b in model.workflows[0] for p in b.parameters())
        active += sum(p.numel() for p in model.heads[0].parameters()) + model.ln_f.weight.numel() * 2
        reference = unique + shared_params  # shared stack referenced by both workflows
    elif isinstance(model, SharedBackboneLM):
        reference = unique
    return {
        "unique_trainable": unique,
        "reference_parameters": reference,
        "active_parameters_one_forward": active,
        "shared_parameter_elements": shared_params,
        "sharing_ratio": shared_params / unique if unique else 0.0,
        "storage_bytes_fp32": unique * 4,
        "storage_mb_fp32": unique * 4 / 1e6,
    }
