"""Compatibility and error-path tests for OT-based queries."""

from __future__ import annotations

import numpy as np
import pytest

from sparc import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
    cw_distance,
    cw_distance_and_grad,
    exp_query,
    gcw_crossterm,
    gcw_coupling_circuit,
    log_exp_query,
)

pytestmark = pytest.mark.queries


class TestExpectationCompatibility:
    def test_mismatched_scopes_raise(self):
        p1 = ProductNode(
            id=2,
            children=[
                CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5]),
                CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5]),
            ],
        )
        p2 = ProductNode(
            id=4,
            children=[
                CategoricalInputNode(id=3, scope_var=0, probabilities=[0.5, 0.5]),
            ],
        )
        with pytest.raises(ValueError, match="expectation incompatible"):
            exp_query(p1, p2)

    def test_mismatched_cardinality_raises(self):
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        l2 = CategoricalInputNode(
            id=1, scope_var=0, probabilities=[0.3, 0.3, 0.4]
        )
        with pytest.raises(ValueError, match="expectation incompatible"):
            exp_query(l1, l2)


class TestCWCompatibility:
    def test_incompatible_structures_raise(self):
        # product vs sum on same scope without matching decomposition
        prod = ProductNode(
            id=2,
            children=[
                CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5]),
                CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5]),
            ],
        )
        circ = SumNode(
            id=5,
            children=[
                CategoricalInputNode(id=3, scope_var=0, probabilities=[0.5, 0.5]),
            ],
            parameters=[1.0],
        )
        with pytest.raises((ValueError, NotImplementedError)):
            cw_distance(prod, circ)

    def test_grad_matches_forward_on_valid_pair(self):
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.3, 0.7])
        l2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.55, 0.45])
        v = cw_distance(l1, l2)
        vg, grads = cw_distance_and_grad(l1, l2)
        assert vg == pytest.approx(v, rel=0, abs=1e-12)
        assert isinstance(grads.sum_grads, dict)
        assert isinstance(grads.cat_grads, dict)


class TestGCWCompatibility:
    def test_coupling_materializes_for_valid_pair(self):
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        l2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.5, 0.5])
        coupling = gcw_coupling_circuit(l1, l2)
        assert coupling.likelihood({0: 0, 1: 0}) > 0.0

    def test_crossterm_finite_on_valid_pair(self):
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.4, 0.6])
        l2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.2, 0.8])
        cross = gcw_crossterm(l1, l2)
        assert np.isfinite(cross)
        assert cross >= -1e-8

    def test_log_exp_identical_leaves_is_log_half(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        # Inner product = 0.25 + 0.25 = 0.5
        assert log_exp_query(leaf, leaf) == pytest.approx(np.log(0.5), rel=0, abs=1e-12)

    def test_disjoint_support_near_zero(self):
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[1.0, 0.0])
        l2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.0, 1.0])
        val = log_exp_query(l1, l2)
        assert val < -1e10 or val == pytest.approx(float("-inf"), abs=1.0)
