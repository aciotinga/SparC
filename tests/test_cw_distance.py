"""Tests for Circuit-Wasserstein distance (``cw_distance`` / ``cw_distance_and_grad``).

Brute-force NW-corner references on leaf couplings validate the built-in
transport solver.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
    cw_distance,
    cw_distance_and_grad,
)


def _nw_coupling_dense(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n, m = len(p), len(q)
    T = np.zeros((n, m), dtype=np.float64)
    i = j = 0
    p_rem, q_rem = p[0], q[0]
    eps = 1e-8
    while i < n and j < m:
        flow = min(p_rem, q_rem)
        T[i, j] = flow
        p_rem -= flow
        q_rem -= flow
        if p_rem < eps:
            i += 1
            if i < n:
                p_rem = p[i]
        if q_rem < eps:
            j += 1
            if j < m:
                q_rem = q[j]
    return T


def _brute_force_cw_leaf(p, q, metric_p=1.0, scale_factor=1.0):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n, m = len(p), len(q)
    d = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            d[i, j] = abs(i - j) ** metric_p / scale_factor
    T = _nw_coupling_dense(p, q)
    return float(np.sum(T * d))


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


class TestCWDistanceSmoke:
    def test_identical_leaves_zero(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        assert cw_distance(leaf, leaf) == pytest.approx(0.0, abs=1e-12)

    def test_leaf_matches_brute_force(self):
        p = [0.3, 0.7]
        q = [0.55, 0.45]
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=q)
        expected = _brute_force_cw_leaf(p, q)
        assert_allclose(cw_distance(leaf1, leaf2), expected, rtol=0, atol=1e-12)

    def test_leaf_unequal_cardinality(self):
        p = [0.2, 0.3, 0.5]
        q = [0.6, 0.4]
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=q)
        expected = _brute_force_cw_leaf(p, q)
        assert_allclose(cw_distance(leaf1, leaf2), expected, rtol=0, atol=1e-12)

    def test_symmetry(self):
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.25, 0.75])
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.6, 0.4])
        assert_allclose(
            cw_distance(leaf1, leaf2), cw_distance(leaf2, leaf1), rtol=0, atol=1e-12
        )

    def test_sum_sum_forward_parity(self):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.8, 0.15, 0.05])
        c1b = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.2, 0.3, 0.5])
        circ1 = SumNode(id=2, children=[c1a, c1b], parameters=[0.35, 0.65])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.1, 0.6, 0.3])
        c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=[0.7, 0.2, 0.1])
        circ2 = SumNode(id=5, children=[c2a, c2b], parameters=[0.4, 0.6])
        v_ref = cw_distance(circ1, circ2)
        v_grad, _ = cw_distance_and_grad(circ1, circ2)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)

    def test_product_product(self):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.4, 0.6])
        c1b = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.7, 0.3])
        circ1 = ProductNode(id=2, children=[c1a, c1b])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.2, 0.8])
        c2b = CategoricalInputNode(id=4, scope_var=1, probabilities=[0.55, 0.45])
        circ2 = ProductNode(id=5, children=[c2a, c2b])
        v_ref = cw_distance(circ1, circ2)
        v_grad, _ = cw_distance_and_grad(circ1, circ2)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)

    def test_metric_scale_factor(self):
        p = [0.3, 0.7]
        q = [0.55, 0.45]
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=q)
        base = cw_distance(leaf1, leaf2, metric_p=1.0, scale_factor=1.0)
        scaled = cw_distance(leaf1, leaf2, metric_p=1.0, scale_factor=2.0)
        assert_allclose(scaled, base / 2.0, rtol=0, atol=1e-12)


class TestCWCompatibilityErrors:
    def test_type_mismatch_sum_product(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        prod = ProductNode(id=1, children=[leaf])
        summ = SumNode(id=2, children=[leaf], parameters=[1.0])
        with pytest.raises(ValueError, match="CW incompatible"):
            cw_distance(summ, prod)

    def test_type_mismatch_leaf_product(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        prod = ProductNode(id=1, children=[leaf])
        with pytest.raises(ValueError, match="CW incompatible"):
            cw_distance(leaf, prod)

    def test_product_scope_mismatch(self):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        c1b = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
        p1 = ProductNode(id=2, children=[c1a, c1b])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.5, 0.5])
        c2b = CategoricalInputNode(id=4, scope_var=2, probabilities=[0.5, 0.5])
        p2 = ProductNode(id=5, children=[c2a, c2b])
        with pytest.raises(ValueError, match="CW incompatible"):
            cw_distance(p1, p2)

    def test_product_child_count_mismatch(self):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        c1b = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
        p1 = ProductNode(id=2, children=[c1a, c1b])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.5, 0.5])
        p2 = ProductNode(id=4, children=[c2a])
        with pytest.raises(ValueError, match="CW incompatible"):
            cw_distance(p1, p2)


class TestCWGradients:
    def test_leaf_cat_grad_matches_fd(self):
        rng = np.random.default_rng(42)
        p_fixed = rng.dirichlet([1.5, 1.0, 0.7])
        q_init = rng.dirichlet([1.0, 1.2, 0.9])
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p_fixed))

        def f_value(probs):
            leaf = CategoricalInputNode(id=1, scope_var=0, probabilities=list(probs))
            return cw_distance(leaf1, leaf)

        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=list(q_init))
        _, grads = cw_distance_and_grad(leaf1, leaf2)
        assert 1 in grads.cat_grads
        g = _project_to_simplex_tangent(grads.cat_grads[1])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)

    def test_sum_sum_phi_grad_matches_fd(self):
        rng = np.random.default_rng(7)
        q1 = rng.dirichlet([1.0, 1.0, 1.0])
        q2 = rng.dirichlet([1.0, 1.0, 1.0])
        p1 = rng.dirichlet([1.0, 1.0, 1.0])
        p2 = rng.dirichlet([1.0, 1.0, 1.0])
        circ1 = SumNode(
            id=2,
            children=[
                CategoricalInputNode(id=0, scope_var=0, probabilities=list(p1)),
                CategoricalInputNode(id=1, scope_var=0, probabilities=list(p2)),
            ],
            parameters=[0.3, 0.7],
        )
        phi_init = np.array([0.42, 0.58])

        def f_value(phi):
            return cw_distance(
                circ1,
                SumNode(
                    id=5,
                    children=[
                        CategoricalInputNode(id=3, scope_var=0, probabilities=list(q1)),
                        CategoricalInputNode(id=4, scope_var=0, probabilities=list(q2)),
                    ],
                    parameters=list(phi),
                ),
            )

        circ2 = SumNode(
            id=5,
            children=[
                CategoricalInputNode(id=3, scope_var=0, probabilities=list(q1)),
                CategoricalInputNode(id=4, scope_var=0, probabilities=list(q2)),
            ],
            parameters=list(phi_init),
        )
        _, grads = cw_distance_and_grad(circ1, circ2)
        assert 5 in grads.sum_grads
        g = _project_to_simplex_tangent(grads.sum_grads[5])
        g_fd = _fd_gradient_simplex(f_value, phi_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)

    def test_product_leaf_grad_matches_fd(self):
        rng = np.random.default_rng(99)
        circ1 = ProductNode(
            id=2,
            children=[
                CategoricalInputNode(
                    id=0, scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0]))
                ),
                CategoricalInputNode(
                    id=1, scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0]))
                ),
            ],
        )
        q_init = rng.dirichlet([1.0, 1.0])
        other_q = list(rng.dirichlet([1.0, 1.0]))

        def f_value(probs):
            return cw_distance(
                circ1,
                ProductNode(
                    id=5,
                    children=[
                        CategoricalInputNode(id=3, scope_var=0, probabilities=list(probs)),
                        CategoricalInputNode(id=4, scope_var=1, probabilities=other_q),
                    ],
                ),
            )

        circ2 = ProductNode(
            id=5,
            children=[
                CategoricalInputNode(id=3, scope_var=0, probabilities=list(q_init)),
                CategoricalInputNode(id=4, scope_var=1, probabilities=other_q),
            ],
        )
        _, grads = cw_distance_and_grad(circ1, circ2)
        assert 3 in grads.cat_grads
        g = _project_to_simplex_tangent(grads.cat_grads[3])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)

    def test_descent_decreases_cw_distance(self):
        rng = np.random.default_rng(2026)
        leaf_ref = CategoricalInputNode(
            id=0, scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        q = rng.dirichlet([1.0, 1.0, 1.0])
        lr = 0.05
        history = []
        for _ in range(8):
            leaf_learn = CategoricalInputNode(id=1, scope_var=0, probabilities=list(q))
            value, grads = cw_distance_and_grad(leaf_ref, leaf_learn)
            history.append(value)
            g = _project_to_simplex_tangent(grads.cat_grads[1])
            q = q - lr * g
            q = np.clip(q, 1e-6, None)
            q = q / q.sum()
        assert history[-1] <= history[0] + 1e-12
