import math

import numpy as np
import pytest

from sparc import (
    CategoricalInputNode,
    Circuit,
    ProductNode,
    SumNode,
    likelihood,
    sample,
)
from tests.sparc_helpers import assignment_array


def _simple_product_circuit():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    prod.propagate_scope()
    return prod


def _weighted_sum_circuit():
    leaf_a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.8, 0.2])
    leaf_b = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.2, 0.8])
    root = SumNode(id=2, children=[leaf_a, leaf_b], parameters=[0.6, 0.4])
    root.propagate_scope()
    return root


def test_sample_deterministic_with_seed():
    root = _simple_product_circuit()
    a = sample(root, 20, seed=42)
    b = sample(root, 20, seed=42)
    assert (a == b).all()


def test_sample_differs_across_seeds():
    root = _simple_product_circuit()
    a = sample(root, 50, seed=1)
    b = sample(root, 50, seed=2)
    assert not (a == b).all()


def test_sample_returns_2d_ndarray():
    root = _simple_product_circuit()
    draws = sample(root, 5, seed=0)
    assert isinstance(draws, np.ndarray)
    assert draws.shape == (5, 18)
    for r in range(5):
        assert draws[r, 3] in (0, 1)
        assert draws[r, 17] in (0, 1)


def test_circuit_wrapper_sample():
    circuit = Circuit(_simple_product_circuit())
    draws = circuit.sample(3, seed=7)
    assert draws.shape == (3, 18)
    for r in range(3):
        assert draws[r, 3] in (0, 1)
        assert draws[r, 17] in (0, 1)


def test_sample_zero_returns_empty_array():
    root = _simple_product_circuit()
    out = sample(root, 0, seed=0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 18)


def test_negative_n_samples_raises():
    root = _simple_product_circuit()
    with pytest.raises(ValueError, match="n_samples must be non-negative"):
        sample(root, -1, seed=0)


def test_empty_scope_raises():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    with pytest.raises(ValueError, match="scope is empty"):
        sample(prod, 1, seed=0)


def test_sampled_assignments_have_positive_likelihood():
    circuit = Circuit(_simple_product_circuit())
    draws = circuit.sample(30, seed=99)
    for r in range(draws.shape[0]):
        p = circuit.likelihood(draws[r])
        assert p > 0.0
        assert math.isfinite(p)


def test_empirical_marginals_match_mixture():
    root = _weighted_sum_circuit()
    n = 20_000
    draws = sample(root, n, seed=123)
    count0 = int((draws[:, 0] == 0).sum())
    p0 = count0 / n
    expected = 0.6 * 0.8 + 0.4 * 0.2
    assert p0 == pytest.approx(expected, abs=0.03)


def test_dag_shared_subtree_samples_valid_assignments():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    shared = ProductNode(id=2, children=[x0, x1])
    root = SumNode(id=3, children=[shared], parameters=[1.0])
    root.propagate_scope()
    draws = sample(root, 10, seed=5)
    for r in range(draws.shape[0]):
        assert draws[r, 3] in (0, 1)
        assert draws[r, 17] in (0, 1)
        assert likelihood(root, draws[r]) > 0.0
