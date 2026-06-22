"""Tests for the differentiable mean log-likelihood query.

``mean_log_likelihood_and_grad`` returns ``(mean_ll, grads)`` where ``grads``
carries the gradient of the mean log-likelihood w.r.t. the circuit's linear
parameters (sum weights and categorical probabilities), keyed by ``node.id``.

The value is checked against averaging ``Circuit.log_likelihood`` over the
dataset, and every gradient vector is checked against a symmetric
finite-difference reference on the simplex tangent.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    Circuit,
    ProductNode,
    SumNode,
    mean_log_likelihood_and_grad,
)


def _structured_circuit():
    cat_a0 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.3, 0.2])
    cat_b0 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.1, 0.6, 0.3])
    sum_x0 = SumNode(id=2, children=[cat_a0, cat_b0], parameters=[0.7, 0.3])

    cat_c1 = CategoricalInputNode(id=3, scope_var=1, probabilities=[0.25, 0.75])
    cat_d1 = CategoricalInputNode(id=4, scope_var=1, probabilities=[0.8, 0.2])
    sum_x1 = SumNode(id=5, children=[cat_c1, cat_d1], parameters=[0.4, 0.6])

    root = ProductNode(id=6, children=[sum_x0, sum_x1])
    root.propagate_scope()
    return root


def _dataset(seed=0, n=25):
    rng = np.random.default_rng(seed)
    data = np.zeros((n, 2), dtype=np.int32)
    data[:, 0] = rng.integers(0, 3, size=n)
    data[:, 1] = rng.integers(0, 2, size=n)
    return data


def _node_by_id(root, target):
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if int(node.id) == target:
            return node
        if isinstance(node, (SumNode, ProductNode)):
            stack.extend(node.children())
    raise KeyError(target)


def _fd_gradient_simplex(f, params, *, eps=1e-6):
    """Symmetric finite difference along ``e_i - e_0`` directions.

    Returns ``g`` with ``sum(g) == 0`` so it matches the simplex-tangent
    projection of the analytic linear-parameter gradient.
    """
    n = len(params)
    dirs = np.zeros(n)
    for i in range(1, n):
        p_plus = np.array(params, dtype=np.float64)
        p_minus = np.array(params, dtype=np.float64)
        p_plus[i] += eps
        p_plus[0] -= eps
        p_minus[i] -= eps
        p_minus[0] += eps
        dirs[i] = (f(p_plus) - f(p_minus)) / (2.0 * eps)
    g = dirs.copy()
    g[0] = 0.0
    g -= g.mean()
    return g


def _tangent(grad):
    g = np.asarray(grad, dtype=np.float64)
    return g - g.mean()


def test_value_matches_average_log_likelihood():
    root = _structured_circuit()
    pc = Circuit(root)
    data = _dataset()

    mean_ll, grads = mean_log_likelihood_and_grad(root, data)

    ref = float(pc.compile().log_likelihood(data).mean())
    assert_allclose(mean_ll, ref, rtol=1e-12, atol=1e-12)
    assert_allclose(grads.value, ref, rtol=1e-12, atol=1e-12)


def test_circuit_method_matches_function():
    root = _structured_circuit()
    pc = Circuit(root)
    data = _dataset(seed=3)

    m1, g1 = pc.mean_log_likelihood_and_grad(data)
    m2, g2 = mean_log_likelihood_and_grad(root, data)

    assert m1 == m2
    for nid in g1.sum_grads:
        assert_allclose(g1.sum_grads[nid], g2.sum_grads[nid])
    for nid in g1.cat_grads:
        assert_allclose(g1.cat_grads[nid], g2.cat_grads[nid])


@pytest.mark.parametrize("node_id", [2, 5])
def test_sum_gradients_match_finite_difference(node_id):
    root = _structured_circuit()
    data = _dataset(seed=1)
    _, grads = mean_log_likelihood_and_grad(root, data)

    node = _node_by_id(root, node_id)
    original = node.parameters_list()

    def f(params):
        node.set_parameters_list((np.asarray(params) / np.sum(params)).tolist())
        val, _ = mean_log_likelihood_and_grad(root, data)
        node.set_parameters_list(original)
        return val

    fd = _fd_gradient_simplex(f, original)
    analytic = _tangent(grads.sum_grads[node_id])
    assert_allclose(analytic, fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("node_id", [0, 1, 3, 4])
def test_categorical_gradients_match_finite_difference(node_id):
    root = _structured_circuit()
    data = _dataset(seed=2)
    _, grads = mean_log_likelihood_and_grad(root, data)

    node = _node_by_id(root, node_id)
    original = node.probabilities_list()

    def f(params):
        node.set_probabilities_list((np.asarray(params) / np.sum(params)).tolist())
        val, _ = mean_log_likelihood_and_grad(root, data)
        node.set_probabilities_list(original)
        return val

    fd = _fd_gradient_simplex(f, original)
    analytic = _tangent(grads.cat_grads[node_id])
    assert_allclose(analytic, fd, rtol=1e-5, atol=1e-6)


def test_empty_dataset_raises():
    root = _structured_circuit()
    with pytest.raises(ValueError):
        mean_log_likelihood_and_grad(root, np.zeros((0, 2), dtype=np.int32))


def test_gradient_ascent_increases_likelihood():
    """One projected ascent step should not decrease the mean log-likelihood."""
    root = _structured_circuit()
    data = _dataset(seed=7)
    lr = 0.05

    mean_ll_before, grads = mean_log_likelihood_and_grad(root, data)

    for node in (_node_by_id(root, i) for i in (2, 5)):
        nid = int(node.id)
        p = np.asarray(node.parameters_list(), dtype=np.float64)
        g = _tangent(grads.sum_grads[nid])
        x = np.clip(p + lr * g, 1e-12, None)
        node.set_parameters_list((x / x.sum()).tolist())
    for node in (_node_by_id(root, i) for i in (0, 1, 3, 4)):
        nid = int(node.id)
        p = np.asarray(node.probabilities_list(), dtype=np.float64)
        g = _tangent(grads.cat_grads[nid])
        x = np.clip(p + lr * g, 1e-12, None)
        node.set_probabilities_list((x / x.sum()).tolist())

    mean_ll_after, _ = mean_log_likelihood_and_grad(root, data)
    assert mean_ll_after >= mean_ll_before - 1e-9
