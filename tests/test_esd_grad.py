"""Tests for expected squared distance (``expected_squared_distance_and_grad``).

Compares analytic gradients to symmetric finite-difference references on the
simplex tangent, mirroring ``tests/test_gcw_grad.py``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
    expected_squared_distance,
    expected_squared_distance_and_grad,
    gcw_crossterm,
    gcw_crossterm_and_grad,
)


def _fd_gradient_simplex(f, params, *, eps=1e-5):
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


def _project_to_simplex_tangent(g):
    g = np.asarray(g, dtype=np.float64)
    return g - g.mean()


class TestForwardValueMatch:
    def test_leaf_value_match(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.3, 0.7])
        v_ref = expected_squared_distance(leaf)
        v_grad, grads = expected_squared_distance_and_grad(leaf)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-12)
        assert_allclose(grads.value, v_ref, rtol=0, atol=1e-12)

    def test_sum_value_match(self):
        c1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.8, 0.15, 0.05])
        c2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.2, 0.3, 0.5])
        circ = SumNode(id=2, children=[c1, c2], parameters=[0.35, 0.65])
        v_ref = expected_squared_distance(circ)
        v_grad, _ = expected_squared_distance_and_grad(circ)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-12)

    def test_product_value_match(self):
        c1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.4, 0.6])
        c2 = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.7, 0.3])
        circ = ProductNode(id=2, children=[c1, c2])
        v_ref = expected_squared_distance(circ)
        v_grad, _ = expected_squared_distance_and_grad(circ)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-12)


class TestLeafGradients:
    def test_leaf_cat_grad_matches_fd(self):
        rng = np.random.default_rng(42)
        p_init = rng.dirichlet([1.5, 1.0, 0.7])

        def f_value(probs):
            leaf = CategoricalInputNode(
                id=0, scope_var=0, probabilities=list(probs)
            )
            return expected_squared_distance(leaf)

        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p_init))
        _, grads = expected_squared_distance_and_grad(leaf)
        assert 0 in grads.cat_grads
        g = _project_to_simplex_tangent(grads.cat_grads[0])
        g_fd = _fd_gradient_simplex(f_value, p_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


class TestSumGradients:
    def test_sum_theta_grad_matches_fd(self):
        rng = np.random.default_rng(7)
        p1 = rng.dirichlet([1.0, 1.0, 1.0])
        p2 = rng.dirichlet([1.0, 1.0, 1.0])
        theta_init = np.array([0.42, 0.58])

        def f_value(theta):
            c1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p1))
            c2 = CategoricalInputNode(id=1, scope_var=0, probabilities=list(p2))
            return expected_squared_distance(
                SumNode(id=2, children=[c1, c2], parameters=list(theta))
            )

        c1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p1))
        c2 = CategoricalInputNode(id=1, scope_var=0, probabilities=list(p2))
        circ = SumNode(id=2, children=[c1, c2], parameters=list(theta_init))
        _, grads = expected_squared_distance_and_grad(circ)
        assert 2 in grads.sum_grads
        g = _project_to_simplex_tangent(grads.sum_grads[2])
        g_fd = _fd_gradient_simplex(f_value, theta_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


class TestProductGradients:
    def test_product_leaf_grad_matches_fd(self):
        rng = np.random.default_rng(99)
        q_init = rng.dirichlet([1.0, 1.0])
        other_q = list(rng.dirichlet([1.0, 1.0]))

        def f_value(probs):
            c1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(probs))
            c2 = CategoricalInputNode(id=1, scope_var=1, probabilities=other_q)
            return expected_squared_distance(ProductNode(id=2, children=[c1, c2]))

        c1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(q_init))
        c2 = CategoricalInputNode(id=1, scope_var=1, probabilities=other_q)
        circ = ProductNode(id=2, children=[c1, c2])
        _, grads = expected_squared_distance_and_grad(circ)
        assert 0 in grads.cat_grads
        g = _project_to_simplex_tangent(grads.cat_grads[0])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


class TestCombinedDescentSmoke:
    """Combined ESD - 2*crossterm descent should decrease total GCW distance."""

    def test_descent_decreases_gcw_distance(self):
        rng = np.random.default_rng(2026)
        n_cats = 3
        p_fixed = rng.dirichlet([1.0] * n_cats)
        leaf_ref = CategoricalInputNode(
            id=0, scope_var=0, probabilities=list(p_fixed)
        )
        esd_c1 = expected_squared_distance(leaf_ref)

        q = rng.dirichlet([1.0] * n_cats)
        lr = 5e-2
        history = []
        for _ in range(8):
            leaf_learn = CategoricalInputNode(
                id=1, scope_var=0, probabilities=list(q)
            )
            cross, cross_grads = gcw_crossterm_and_grad(leaf_ref, leaf_learn)
            esd2, esd_grads = expected_squared_distance_and_grad(leaf_learn)
            history.append(esd_c1 + esd2 - 2.0 * cross)

            g_esd = _project_to_simplex_tangent(esd_grads.cat_grads[1])
            g_cross = _project_to_simplex_tangent(cross_grads.cat_grads[1])
            g_total = g_esd - 2.0 * g_cross
            q = q - lr * g_total
            q = np.clip(q, 1e-6, None)
            q = q / q.sum()

        assert history[-1] <= history[0] + 1e-8
