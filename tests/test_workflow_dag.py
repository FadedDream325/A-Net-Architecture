"""Workflow DAG: acyclicity, topological order, terminal dependencies."""

from __future__ import annotations

import pytest
import torch

from experiments.anet_v2.core.models import ANetModel
from experiments.anet_v2.core.workflow import CallSite, WorkflowGraph


def test_linear_graph_topological_order():
    g = WorkflowGraph()
    g.add_site(CallSite("a", "w0", "b0", "input"), ["input"])
    g.add_site(CallSite("b", "w0", "b1", "a"), ["a"])
    g.add_site(CallSite("c", "w1", "b2", "b"), ["b"])
    assert g.topological_order() == ["input", "a", "b", "c"]
    g.assert_acyclic()


def test_cycle_is_detected():
    g = WorkflowGraph()
    g.add_site(CallSite("a", "w0", "b0", "input"), ["input"])
    g.add_site(CallSite("b", "w0", "b1", "a"), ["a"])
    g.edges.add(("b", "a"))  # a -> b and b -> a
    with pytest.raises(ValueError, match="cycle"):
        g.topological_order()


def test_duplicate_site_is_rejected():
    g = WorkflowGraph()
    g.add_site(CallSite("a", "w0", "b0", "input"), ["input"])
    with pytest.raises(ValueError):
        g.add_site(CallSite("a", "w0", "b1", "input"), ["input"])


def test_model_graph_is_acyclic_and_terminal_only_depends_on_workflow_outputs():
    model = ANetModel(3, 6, 16, [2, 1, 5], share_mode="block", cache_enabled=False, active_mode="sparse")
    order = model.graph.topological_order()
    assert order[-1] == "terminal"
    assert model.graph.terminal_inputs() == {f"w{t}.output" for t in range(3)}
    assert all(not x.startswith("prel1") for x in model.graph.terminal_inputs())


def test_all_model_variants_have_acyclic_graphs():
    for share_mode in ("none", "block", "adapter"):
        model = ANetModel(3, 6, 16, [2, 1, 5], share_mode=share_mode, cache_enabled=True, active_mode="dense")
        model.graph.assert_acyclic()
        assert model.graph.terminal_inputs() == {f"w{t}.output" for t in range(3)}
