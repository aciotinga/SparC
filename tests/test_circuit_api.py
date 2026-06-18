"""Tests for Circuit wrapper API: clone, save/load, normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    BernoulliInputNode,
    CategoricalInputNode,
    Circuit,
    CircuitSerializer,
    ProductNode,
    SumNode,
)
from tests.sparc_helpers import (
    exact_total_mass,
    walk_pc_invariants,
)

pytestmark = pytest.mark.circuit


def _build_tree():
    l0a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.8, 0.2])
    l1a = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
    l0b = CategoricalInputNode(id=2, scope_var=0, probabilities=[0.3, 0.7])
    l1b = CategoricalInputNode(id=3, scope_var=1, probabilities=[0.25, 0.75])
    p0 = ProductNode(id=4, children=[l0a, l1a])
    p1 = ProductNode(id=5, children=[l0b, l1b])
    return SumNode(id=6, children=[p0, p1], parameters=[0.6, 0.4])


class TestCircuitClone:
    def test_clone_preserves_likelihood(self):
        circuit = Circuit(_build_tree())
        asg = {0: 1, 1: 0}
        assert_allclose(
            circuit.clone().log_likelihood(asg),
            circuit.log_likelihood(asg),
            rtol=0,
            atol=1e-12,
        )

    def test_clone_breaks_parameter_aliasing(self):
        circuit = Circuit(_build_tree())
        clone = circuit.clone()
        clone.root.set_parameters_list([0.1, 0.9])
        assert circuit.root.parameters_list() == pytest.approx([0.6, 0.4])

    def test_clone_is_deep_copy_of_nodes(self):
        circuit = Circuit(_build_tree())
        clone = circuit.clone()
        assert circuit.root is not clone.root
        assert circuit.root.children()[0] is not clone.root.children()[0]


class TestCircuitSaveLoad:
    def test_save_load_roundtrip(self):
        circuit = Circuit(_build_tree())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "circuit.json"
            circuit.save(path)
            loaded = Circuit.load(path)
        row = {0: 0, 1: 1}
        assert_allclose(
            loaded.log_likelihood(row),
            circuit.log_likelihood(row),
            rtol=0,
            atol=1e-12,
        )

    def test_save_load_preserves_dag_sharing(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.2, 0.8])
        root = SumNode(id=1, children=[leaf, leaf], parameters=[0.4, 0.6])
        circuit = Circuit(root)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dag.json"
            circuit.save(path)
            loaded = Circuit.load(path)
        assert loaded.root.children()[0] is loaded.root.children()[1]

    def test_bernoulli_save_load_roundtrip(self):
        leaf = BernoulliInputNode(id=0, scope_var=0, p=0.4)
        circuit = Circuit(leaf)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bern.json"
            circuit.save(path)
            loaded = Circuit.load(path)
        assert loaded.likelihood({0: 1}) == pytest.approx(circuit.likelihood({0: 1}))


class TestCircuitAutoScope:
    def test_constructor_propagates_empty_scope(self):
        l0 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        l1 = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
        prod = ProductNode(id=2, children=[l0, l1])
        circuit = Circuit(prod)
        assert set(circuit.root.scope_as_list()) == {0, 1}


class TestNormalization:
    def test_mixture_normalizes(self):
        circuit = Circuit(_build_tree())
        assert_allclose(exact_total_mass(circuit), 1.0, rtol=0, atol=1e-10)
        walk_pc_invariants(circuit.root)

    @pytest.mark.parametrize("p", [[0.5, 0.5], [0.1, 0.9], [0.33, 0.67]])
    def test_single_leaf_normalizes(self, p):
        circuit = Circuit(CategoricalInputNode(id=0, scope_var=0, probabilities=p))
        assert_allclose(exact_total_mass(circuit), 1.0, rtol=0, atol=1e-10)
