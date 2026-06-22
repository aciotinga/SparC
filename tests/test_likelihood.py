import math

import numpy as np
import pytest

from sparc import (
    CategoricalInputNode,
    Circuit,
    ProductNode,
    SumNode,
    likelihood,
    log_likelihood,
)
from tests.sparc_helpers import assignment_array


def test_leaf_likelihood():
    leaf = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    row = assignment_array({3: 0})
    assert likelihood(leaf, row) == pytest.approx(0.7)
    row = assignment_array({3: 1})
    assert likelihood(leaf, row) == pytest.approx(0.3)


def test_product_likelihood():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    prod.propagate_scope()
    assert likelihood(prod, assignment_array({3: 0, 17: 1})) == pytest.approx(0.35)


def test_sum_likelihood():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    root = SumNode(id=3, children=[prod], parameters=[1.0])
    root.propagate_scope()
    assert likelihood(root, assignment_array({3: 0, 17: 1})) == pytest.approx(0.35)


def test_circuit_wrapper():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    root = SumNode(id=3, children=[prod], parameters=[1.0])
    circuit = Circuit(root)
    assert circuit.likelihood(assignment_array({3: 0, 17: 1})) == pytest.approx(0.35)


def test_dag_shared_subtree():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    shared = ProductNode(id=2, children=[x0, x1])
    sum_a = SumNode(id=3, children=[shared], parameters=[1.0])
    sum_b = SumNode(id=4, children=[shared], parameters=[1.0])
    sum_a.propagate_scope()
    sum_b.propagate_scope()
    assignment = assignment_array({3: 0, 17: 0})
    expected = 0.7 * 0.5
    assert likelihood(sum_a, assignment) == pytest.approx(expected)
    assert likelihood(sum_b, assignment) == pytest.approx(expected)


def test_missing_evidence_raises():
    leaf = CategoricalInputNode(id=0, scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="shorter than required"):
        likelihood(leaf, np.array([0], dtype=np.int32))


def test_out_of_range_evidence_raises():
    leaf = CategoricalInputNode(id=0, scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="out of range"):
        likelihood(leaf, assignment_array({1: 2}))


def test_empty_scope_raises():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    with pytest.raises(ValueError, match="scope is empty"):
        likelihood(prod, assignment_array({3: 0, 17: 0}))


def test_leaf_log_likelihood():
    leaf = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    assert log_likelihood(leaf, assignment_array({3: 0})) == pytest.approx(math.log(0.7))
    assert log_likelihood(leaf, assignment_array({3: 1})) == pytest.approx(math.log(0.3))


def test_log_likelihood_matches_log_of_likelihood():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    root = SumNode(id=3, children=[prod], parameters=[1.0])
    root.propagate_scope()
    assignment = assignment_array({3: 0, 17: 1})
    assert log_likelihood(root, assignment) == pytest.approx(
        math.log(likelihood(root, assignment))
    )


def test_weighted_sum_log_likelihood():
    leaf_a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
    leaf_b = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
    root = SumNode(id=2, children=[leaf_a, leaf_b], parameters=[0.6, 0.4])
    root.propagate_scope()
    assignment = assignment_array({0: 0, 1: 0})
    p = likelihood(root, assignment)
    assert log_likelihood(root, assignment) == pytest.approx(math.log(p))
    assert p == pytest.approx(0.5)


def test_circuit_log_likelihood():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    root = SumNode(id=3, children=[prod], parameters=[1.0])
    circuit = Circuit(root)
    assignment = assignment_array({3: 0, 17: 1})
    assert circuit.log_likelihood(assignment) == pytest.approx(
        math.log(circuit.likelihood(assignment))
    )


def test_logsumexp_numerical_stability():
    """Tiny mixture weights should not underflow in log-space."""
    tiny = 1e-300
    big = 1.0 - tiny
    leaf_a = CategoricalInputNode(id=0, scope_var=0, probabilities=[big, tiny])
    leaf_b = CategoricalInputNode(id=1, scope_var=1, probabilities=[tiny, big])
    root = SumNode(id=2, children=[leaf_a, leaf_b], parameters=[0.5, 0.5])
    root.propagate_scope()
    assignment = assignment_array({0: 0, 1: 1})
    ll = log_likelihood(root, assignment)
    assert ll > -math.inf
    assert ll == pytest.approx(math.log(likelihood(root, assignment)), rel=1e-9)
