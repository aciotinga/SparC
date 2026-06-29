"""Tests for CircuitNode API: clone, save/load, normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    CircuitSerializer,
    ProductNode,
    SumNode,
)
from tests.sparc_helpers import (
    assignment_array,
    exact_total_mass,
    walk_pc_invariants,
)

pytestmark = pytest.mark.node


def _build_tree():
    l0a = CategoricalInputNode(scope_var=0, probabilities=[0.8, 0.2])
    l1a = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    l0b = CategoricalInputNode(scope_var=0, probabilities=[0.3, 0.7])
    l1b = CategoricalInputNode(scope_var=1, probabilities=[0.25, 0.75])
    p0 = ProductNode(children=[l0a, l1a])
    p1 = ProductNode(children=[l0b, l1b])
    return SumNode(children=[p0, p1], parameters=[0.6, 0.4])


class TestNodeClone:
    def test_clone_preserves_likelihood(self):
        root = _build_tree()
        asg = assignment_array({0: 1, 1: 0})
        assert_allclose(
            root.clone().log_likelihood(asg),
            root.log_likelihood(asg),
            rtol=0,
            atol=1e-12,
        )

    def test_clone_breaks_parameter_aliasing(self):
        root = _build_tree()
        clone = root.clone()
        clone.set_parameters_list([0.1, 0.9])
        assert root.parameters_list() == pytest.approx([0.6, 0.4])

    def test_clone_is_deep_copy_of_nodes(self):
        root = _build_tree()
        clone = root.clone()
        assert root is not clone
        assert root.children()[0] is not clone.children()[0]


class TestNodeSaveLoad:
    def test_save_load_roundtrip(self):
        root = _build_tree()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "circuit.json"
            root.save(path)
            loaded = CircuitNode.load(path)
        row = assignment_array({0: 0, 1: 1})
        assert_allclose(
            loaded.log_likelihood(row),
            root.log_likelihood(row),
            rtol=0,
            atol=1e-12,
        )

    def test_save_load_preserves_dag_sharing(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.8])
        root = SumNode(children=[leaf, leaf], parameters=[0.4, 0.6])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dag.json"
            root.save(path)
            loaded = CircuitNode.load(path)
        assert loaded.children()[0] is loaded.children()[1]

    def test_bernoulli_save_load_roundtrip(self):
        leaf = BernoulliInputNode(scope_var=0, p=0.4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bern.json"
            leaf.save(path)
            loaded = CircuitNode.load(path)
        assert loaded.likelihood(assignment_array({0: 1})) == pytest.approx(
            leaf.likelihood(assignment_array({0: 1}))
        )


class TestNodeAutoScope:
    def test_query_propagates_empty_scope(self):
        l0 = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        l1 = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
        prod = ProductNode(children=[l0, l1])
        assert set(prod.scope_as_list()) == {0, 1}


class TestNormalization:
    def test_mixture_normalizes(self):
        root = _build_tree()
        assert_allclose(exact_total_mass(root), 1.0, rtol=0, atol=1e-10)
        walk_pc_invariants(root)

    @pytest.mark.parametrize("p", [[0.5, 0.5], [0.1, 0.9], [0.33, 0.67]])
    def test_single_leaf_normalizes(self, p):
        root = CategoricalInputNode(scope_var=0, probabilities=p)
        assert_allclose(exact_total_mass(root), 1.0, rtol=0, atol=1e-10)
