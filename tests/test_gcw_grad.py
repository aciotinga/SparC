"""Tests for the differentiable GCW cross-term query (``gcw_crossterm_and_grad``).

Each test compares the analytic subgradient returned by
``gcw_crossterm_and_grad`` to a symmetric finite-difference reference. To avoid
non-smooth points of the underlying piecewise-linear/quadratic structure
(LP degeneracy at sum-sum nodes, Hungarian ties, NW mass coincidences) the
fixtures use random non-symmetric weights and probability vectors.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
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
    """The differentiable solver must return the same value as ``gcw_crossterm``."""

    def test_leaf_value_match(self):
        leaf1 = CategoricalInputNode(scope_var=0, probabilities=[0.3, 0.7])
        leaf2 = CategoricalInputNode(scope_var=0, probabilities=[0.55, 0.45])
        v_ref = gcw_crossterm(leaf1, leaf2)
        v_grad, grads = gcw_crossterm_and_grad(leaf1, leaf2)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-12)
        assert_allclose(grads.value, v_ref, rtol=0, atol=1e-12)

    def test_sum_sum_value_match(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.8, 0.15, 0.05])
        c1b = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.3, 0.5])
        circ1 = SumNode(children=[c1a, c1b], parameters=[0.35, 0.65])
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.1, 0.6, 0.3])
        c2b = CategoricalInputNode(scope_var=0, probabilities=[0.7, 0.2, 0.1])
        circ2 = SumNode(children=[c2a, c2b], parameters=[0.4, 0.6])
        v_ref = gcw_crossterm(circ1, circ2)
        v_grad, _ = gcw_crossterm_and_grad(circ1, circ2)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)

    def test_product_product_value_match(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.4, 0.6])
        c1b = CategoricalInputNode(scope_var=1, probabilities=[0.7, 0.3])
        circ1 = ProductNode(children=[c1a, c1b])
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.8])
        c2b = CategoricalInputNode(scope_var=1, probabilities=[0.55, 0.45])
        circ2 = ProductNode(children=[c2a, c2b])
        v_ref = gcw_crossterm(circ1, circ2)
        v_grad, _ = gcw_crossterm_and_grad(circ1, circ2)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)

    def test_mixed_value_match(self):
        rng = np.random.default_rng(0)
        leaves1 = [
            CategoricalInputNode(
                id=10 + i, scope_var=i % 2, probabilities=list(rng.dirichlet([1.0] * 3))
            )
            for i in range(4)
        ]
        prod1 = ProductNode(children=[leaves1[0], leaves1[1]])
        prod2 = ProductNode(children=[leaves1[2], leaves1[3]])
        circ1 = SumNode(children=[prod1, prod2], parameters=[0.45, 0.55])

        leaves2 = [
            CategoricalInputNode(
                id=30 + i, scope_var=i % 2, probabilities=list(rng.dirichlet([1.0] * 3))
            )
            for i in range(4)
        ]
        prod3 = ProductNode(children=[leaves2[0], leaves2[1]])
        prod4 = ProductNode(children=[leaves2[2], leaves2[3]])
        circ2 = SumNode(children=[prod3, prod4], parameters=[0.3, 0.7])

        v_ref = gcw_crossterm(circ1, circ2)
        v_grad, _ = gcw_crossterm_and_grad(circ1, circ2)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)


class TestLeafGradients:
    def test_leaf_cat_grad_matches_fd(self):
        rng = np.random.default_rng(42)
        p_fixed = rng.dirichlet([1.5, 1.0, 0.7])
        q_init = rng.dirichlet([1.0, 1.2, 0.9])
        leaf1 = CategoricalInputNode(scope_var=0, probabilities=list(p_fixed))

        def f_value(probs):
            leaf = CategoricalInputNode(scope_var=0, probabilities=list(probs))
            return gcw_crossterm(leaf1, leaf)

        leaf2 = CategoricalInputNode(scope_var=0, probabilities=list(q_init))
        _, grads = gcw_crossterm_and_grad(leaf1, leaf2)
        assert 1 in grads.cat_grads, "circuit2 leaf should receive a gradient"
        g = _project_to_simplex_tangent(grads.cat_grads[1])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


class TestSumSumGradients:
    def test_sum_sum_phi_grad_matches_fd(self):
        rng = np.random.default_rng(7)
        p1 = rng.dirichlet([1.0, 1.0, 1.0])
        p2 = rng.dirichlet([1.0, 1.0, 1.0])
        c1a = CategoricalInputNode(scope_var=0, probabilities=list(p1))
        c1b = CategoricalInputNode(scope_var=0, probabilities=list(p2))
        circ1 = SumNode(children=[c1a, c1b], parameters=[0.3, 0.7])

        q1 = rng.dirichlet([1.0, 1.0, 1.0])
        q2 = rng.dirichlet([1.0, 1.0, 1.0])
        phi_init = np.array([0.42, 0.58])

        def f_value(phi):
            c2a = CategoricalInputNode(scope_var=0, probabilities=list(q1))
            c2b = CategoricalInputNode(scope_var=0, probabilities=list(q2))
            return gcw_crossterm(
                circ1, SumNode(children=[c2a, c2b], parameters=list(phi))
            )

        c2a = CategoricalInputNode(scope_var=0, probabilities=list(q1))
        c2b = CategoricalInputNode(scope_var=0, probabilities=list(q2))
        circ2 = SumNode(children=[c2a, c2b], parameters=list(phi_init))
        _, grads = gcw_crossterm_and_grad(circ1, circ2)
        assert 5 in grads.sum_grads
        g = _project_to_simplex_tangent(grads.sum_grads[5])
        g_fd = _fd_gradient_simplex(f_value, phi_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


class TestProductGradients:
    def test_product_leaf_grad_matches_fd(self):
        rng = np.random.default_rng(99)
        c1a = CategoricalInputNode(
            scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0]))
        )
        c1b = CategoricalInputNode(
            scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0]))
        )
        circ1 = ProductNode(children=[c1a, c1b])

        q_init = rng.dirichlet([1.0, 1.0])
        other_q = list(rng.dirichlet([1.0, 1.0]))

        def f_value(probs):
            c2a = CategoricalInputNode(scope_var=0, probabilities=list(probs))
            c2b = CategoricalInputNode(scope_var=1, probabilities=other_q)
            return gcw_crossterm(circ1, ProductNode(children=[c2a, c2b]))

        c2a = CategoricalInputNode(scope_var=0, probabilities=list(q_init))
        c2b = CategoricalInputNode(scope_var=1, probabilities=other_q)
        circ2 = ProductNode(children=[c2a, c2b])
        _, grads = gcw_crossterm_and_grad(circ1, circ2)
        assert int(c2a.id) in grads.cat_grads
        g = _project_to_simplex_tangent(grads.cat_grads[int(c2a.id)])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


class TestMixedGradients:
    """End-to-end finite-difference check on a small sum-of-products circuit."""

    def _build_circuit1(self, rng):
        l1 = CategoricalInputNode(
            scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        l2 = CategoricalInputNode(
            scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        l3 = CategoricalInputNode(
            scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        l4 = CategoricalInputNode(
            scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        prodA = ProductNode(children=[l1, l2])
        prodB = ProductNode(children=[l3, l4])
        return SumNode(children=[prodA, prodB], parameters=[0.4, 0.6])

    def test_sum_grad_in_sum_of_products(self):
        rng = np.random.default_rng(13)
        circ1 = self._build_circuit1(rng)

        def build_circ2(theta):
            l1 = CategoricalInputNode(scope_var=0, probabilities=[0.55, 0.25, 0.2])
            l2 = CategoricalInputNode(scope_var=1, probabilities=[0.3, 0.5, 0.2])
            l3 = CategoricalInputNode(scope_var=0, probabilities=[0.15, 0.6, 0.25])
            l4 = CategoricalInputNode(scope_var=1, probabilities=[0.7, 0.1, 0.2])
            prodA = ProductNode(children=[l1, l2])
            prodB = ProductNode(children=[l3, l4])
            return SumNode(children=[prodA, prodB], parameters=list(theta))

        theta_init = np.array([0.45, 0.55])
        circ2 = build_circ2(theta_init)
        _, grads = gcw_crossterm_and_grad(circ1, circ2)
        assert int(circ2.id) in grads.sum_grads
        g_an = _project_to_simplex_tangent(grads.sum_grads[int(circ2.id)])
        g_fd = _fd_gradient_simplex(
            lambda th: gcw_crossterm(circ1, build_circ2(th)), theta_init, eps=1e-6
        )
        assert_allclose(g_an, g_fd, atol=1e-5)


class TestGradientAscentSmoke:
    """A few projected-gradient steps should improve the crossterm monotonically."""

    def test_ascent_increases_crossterm(self):
        rng = np.random.default_rng(2026)
        n_cats = 3
        p_fixed = rng.dirichlet([1.0] * n_cats)
        leaf_ref = CategoricalInputNode(scope_var=0, probabilities=list(p_fixed))

        q = rng.dirichlet([1.0] * n_cats)
        lr = 5e-2
        history = []
        for _ in range(8):
            leaf_learn = CategoricalInputNode(scope_var=0, probabilities=list(q))
            value, grads = gcw_crossterm_and_grad(leaf_ref, leaf_learn)
            history.append(value)
            g = _project_to_simplex_tangent(grads.cat_grads[int(leaf_learn.id)])
            q = q + lr * g
            q = np.clip(q, 1e-6, None)
            q = q / q.sum()

        diffs = np.diff(history)
        assert history[-1] >= history[0] - 1e-12
        assert (diffs >= -1e-8).all(), f"saw decrease in ascent: {history}"
