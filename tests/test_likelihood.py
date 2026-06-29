import math

import numpy as np
import pytest

from sparc import (
    CategoricalInputNode,
        ProductNode,
    SumNode,
    likelihood,
    log_likelihood,
)
from tests.sparc_helpers import assignment_array, exact_partial_likelihood


def test_leaf_likelihood():
    leaf = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    row = assignment_array({3: 0})
    assert likelihood(leaf, row) == pytest.approx(0.7)
    row = assignment_array({3: 1})
    assert likelihood(leaf, row) == pytest.approx(0.3)


def test_product_likelihood():
    x0 = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    prod.propagate_scope()
    assert likelihood(prod, assignment_array({3: 0, 17: 1})) == pytest.approx(0.35)


def test_sum_likelihood():
    x0 = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    root = SumNode(children=[prod], parameters=[1.0])
    root.propagate_scope()
    assert likelihood(root, assignment_array({3: 0, 17: 1})) == pytest.approx(0.35)


def test_circuit_wrapper():
    x0 = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    root = SumNode(children=[prod], parameters=[1.0])
    circuit = root
    assert circuit.likelihood(assignment_array({3: 0, 17: 1})) == pytest.approx(0.35)


def test_dag_shared_subtree():
    x0 = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=17, probabilities=[0.5, 0.5])
    shared = ProductNode(children=[x0, x1])
    sum_a = SumNode(children=[shared], parameters=[1.0])
    sum_b = SumNode(children=[shared], parameters=[1.0])
    sum_a.propagate_scope()
    sum_b.propagate_scope()
    assignment = assignment_array({3: 0, 17: 0})
    expected = 0.7 * 0.5
    assert likelihood(sum_a, assignment) == pytest.approx(expected)
    assert likelihood(sum_b, assignment) == pytest.approx(expected)


def test_missing_evidence_raises():
    leaf = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="shorter than required"):
        likelihood(leaf, np.array([0], dtype=np.int32))


def test_out_of_range_evidence_raises():
    leaf = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="out of range"):
        likelihood(leaf, assignment_array({1: 2}))


def test_likelihood_auto_propagates_scope():
    x0 = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    assert likelihood(prod, assignment_array({3: 0, 17: 0})) > 0.0


def test_leaf_log_likelihood():
    leaf = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    assert log_likelihood(leaf, assignment_array({3: 0})) == pytest.approx(math.log(0.7))
    assert log_likelihood(leaf, assignment_array({3: 1})) == pytest.approx(math.log(0.3))


def test_log_likelihood_matches_log_of_likelihood():
    x0 = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    root = SumNode(children=[prod], parameters=[1.0])
    root.propagate_scope()
    assignment = assignment_array({3: 0, 17: 1})
    assert log_likelihood(root, assignment) == pytest.approx(
        math.log(likelihood(root, assignment))
    )


def test_weighted_sum_log_likelihood():
    leaf_a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
    leaf_b = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    root = SumNode(children=[leaf_a, leaf_b], parameters=[0.6, 0.4])
    root.propagate_scope()
    assignment = assignment_array({0: 0, 1: 0})
    p = likelihood(root, assignment)
    assert log_likelihood(root, assignment) == pytest.approx(math.log(p))
    assert p == pytest.approx(0.5)


def test_circuit_log_likelihood():
    x0 = CategoricalInputNode(scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    root = SumNode(children=[prod], parameters=[1.0])
    circuit = root
    assignment = assignment_array({3: 0, 17: 1})
    assert circuit.log_likelihood(assignment) == pytest.approx(
        math.log(circuit.likelihood(assignment))
    )


def test_logsumexp_numerical_stability():
    """Tiny mixture weights should not underflow in log-space."""
    tiny = 1e-300
    big = 1.0 - tiny
    leaf_a = CategoricalInputNode(scope_var=0, probabilities=[big, tiny])
    leaf_b = CategoricalInputNode(scope_var=1, probabilities=[tiny, big])
    root = SumNode(children=[leaf_a, leaf_b], parameters=[0.5, 0.5])
    root.propagate_scope()
    assignment = assignment_array({0: 0, 1: 1})
    ll = log_likelihood(root, assignment)
    assert ll > -math.inf
    assert ll == pytest.approx(math.log(likelihood(root, assignment)), rel=1e-9)


def test_leaf_marginal_nan():
    leaf = CategoricalInputNode(scope_var=1, probabilities=[0.7, 0.3])
    row = np.array([np.nan, np.nan], dtype=np.float64)
    assert likelihood(leaf, row) == pytest.approx(1.0)
    assert log_likelihood(leaf, row) == pytest.approx(0.0)


def test_product_partial_evidence_matches_exact():
    x0 = CategoricalInputNode(scope_var=0, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    prod.propagate_scope()
    circuit = prod
    row = np.array([0.0, np.nan], dtype=np.float64)
    expected = exact_partial_likelihood(circuit, {0: 0})
    assert likelihood(prod, row) == pytest.approx(expected)
    assert log_likelihood(prod, row) == pytest.approx(math.log(expected))


def test_sum_partial_evidence_matches_exact():
    x0 = CategoricalInputNode(scope_var=0, probabilities=[0.8, 0.2])
    x1 = CategoricalInputNode(scope_var=0, probabilities=[0.3, 0.7])
    mix = SumNode(children=[x0, x1], parameters=[0.6, 0.4])
    x2 = CategoricalInputNode(scope_var=1, probabilities=[0.25, 0.75])
    root = ProductNode(children=[mix, x2])
    root.propagate_scope()
    circuit = root
    row = np.array([np.nan, 1.0], dtype=np.float64)
    expected = exact_partial_likelihood(circuit, {1: 1})
    assert likelihood(root, row) == pytest.approx(expected)


def test_all_nan_normalized_circuit():
    x0 = CategoricalInputNode(scope_var=0, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    prod = ProductNode(children=[x0, x1])
    root = SumNode(children=[prod], parameters=[1.0])
    root.propagate_scope()
    row = np.array([np.nan, np.nan], dtype=np.float64)
    assert likelihood(root, row) == pytest.approx(1.0)


def test_integer_minus_one_still_raises():
    leaf = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        likelihood(leaf, np.array([0, -1], dtype=np.int32))


def test_non_integer_float_raises():
    leaf = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="integer"):
        likelihood(leaf, np.array([0.5], dtype=np.float64))
