"""Tests for expectation query (``exp_query`` / ``exp_query_and_grad``)."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
    exp_query,
    exp_query_and_grad,
    likelihood,
)


def _collect_scopes(root):
    scopes = set()
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        if isinstance(node, CategoricalInputNode):
            scopes.update(node.scope_as_list())
        elif isinstance(node, (SumNode, ProductNode)):
            stack.extend(node.children())
    return sorted(scopes)


def _cardinality_for_var(root, var):
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        if isinstance(node, CategoricalInputNode) and var in node.scope_as_list():
            return node.cardinality()
        elif isinstance(node, (SumNode, ProductNode)):
            stack.extend(node.children())
    raise KeyError(f"variable {var} not found")


def _propagate_scope(root):
    if not isinstance(root, CategoricalInputNode):
        root.propagate_scope()


def _brute_force_inner_product(circ1, circ2):
    _propagate_scope(circ1)
    _propagate_scope(circ2)
    vars_ = _collect_scopes(circ1)
    assert vars_ == _collect_scopes(circ2)
    total = 0.0
    ranges = [
        range(_cardinality_for_var(circ1, v))
        for v in vars_
    ]
    for assignment_tuple in itertools.product(*ranges):
        row = np.zeros(max(vars_) + 1, dtype=np.int32)
        for v, val in zip(vars_, assignment_tuple):
            row[v] = val
        total += likelihood(circ1, row) * likelihood(circ2, row)
    return total


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


class TestExpQuerySmoke:
    def test_identical_leaves(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        assert exp_query(leaf, leaf) == pytest.approx(0.5, abs=1e-12)

    def test_leaf_dot_product(self):
        p = [0.3, 0.7]
        q = [0.55, 0.45]
        leaf1 = CategoricalInputNode(scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(scope_var=0, probabilities=q)
        expected = float(np.dot(p, q))
        got = exp_query(leaf1, leaf2)
        assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_leaf_matches_brute_force(self):
        p = [0.2, 0.3, 0.5]
        q = [0.6, 0.4, 0.0]
        leaf1 = CategoricalInputNode(scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(scope_var=0, probabilities=q)
        expected = _brute_force_inner_product(leaf1, leaf2)
        got = exp_query(leaf1, leaf2)
        assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_sum_sum_forward_parity(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.8, 0.15, 0.05])
        c1b = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.3, 0.5])
        circ1 = SumNode(children=[c1a, c1b], parameters=[0.35, 0.65])
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.1, 0.6, 0.3])
        c2b = CategoricalInputNode(scope_var=0, probabilities=[0.7, 0.2, 0.1])
        circ2 = SumNode(children=[c2a, c2b], parameters=[0.4, 0.6])
        expected = _brute_force_inner_product(circ1, circ2)
        got = exp_query(circ1, circ2)
        assert_allclose(got, expected, rtol=0, atol=1e-10)

    def test_product_product(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.4, 0.6])
        c1b = CategoricalInputNode(scope_var=1, probabilities=[0.7, 0.3])
        circ1 = ProductNode(children=[c1a, c1b])
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.8])
        c2b = CategoricalInputNode(scope_var=1, probabilities=[0.55, 0.45])
        circ2 = ProductNode(children=[c2a, c2b])
        expected = _brute_force_inner_product(circ1, circ2)
        got = exp_query(circ1, circ2)
        assert_allclose(got, expected, rtol=0, atol=1e-10)

    def test_nested_matches_brute_force(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        c1b = CategoricalInputNode(scope_var=1, probabilities=[0.3, 0.7])
        c1c = CategoricalInputNode(scope_var=2, probabilities=[0.2, 0.8])
        circ1 = ProductNode(
            children=[
                SumNode(children=[c1a], parameters=[1.0]),
                ProductNode(children=[c1b, c1c]),
            ],
        )
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.6, 0.4])
        c2b = CategoricalInputNode(scope_var=1, probabilities=[0.25, 0.75])
        c2c = CategoricalInputNode(scope_var=2, probabilities=[0.9, 0.1])
        circ2 = ProductNode(
            children=[
                SumNode(children=[c2a], parameters=[1.0]),
                ProductNode(children=[c2b, c2c]),
            ],
        )
        expected = _brute_force_inner_product(circ1, circ2)
        v_ref = exp_query(circ1, circ2)
        v_grad, _, _ = exp_query_and_grad(circ1, circ2)
        assert_allclose(v_ref, expected, rtol=0, atol=1e-10)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)


class TestExpCompatibilityErrors:
    def test_type_mismatch_sum_product(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        prod = ProductNode(children=[leaf])
        summ = SumNode(children=[leaf], parameters=[1.0])
        with pytest.raises(ValueError, match="expectation incompatible"):
            exp_query(summ, prod)

    def test_type_mismatch_leaf_product(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        prod = ProductNode(children=[leaf])
        with pytest.raises(ValueError, match="expectation incompatible"):
            exp_query(leaf, prod)

    def test_leaf_cardinality_mismatch(self):
        leaf1 = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        leaf2 = CategoricalInputNode(scope_var=0, probabilities=[0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="expectation incompatible"):
            exp_query(leaf1, leaf2)

    def test_product_scope_mismatch(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        c1b = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
        p1 = ProductNode(children=[c1a, c1b])
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        c2b = CategoricalInputNode(scope_var=2, probabilities=[0.5, 0.5])
        p2 = ProductNode(children=[c2a, c2b])
        with pytest.raises(ValueError, match="expectation incompatible"):
            exp_query(p1, p2)

    def test_product_child_count_mismatch(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        c1b = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
        p1 = ProductNode(children=[c1a, c1b])
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        p2 = ProductNode(children=[c2a])
        with pytest.raises(ValueError, match="expectation incompatible"):
            exp_query(p1, p2)


class TestExpGradients:
    def test_leaf_cat_grad_c1_matches_fd(self):
        rng = np.random.default_rng(42)
        p_init = rng.dirichlet([1.5, 1.0, 0.7])
        q_fixed = rng.dirichlet([1.0, 1.2, 0.9])
        leaf2 = CategoricalInputNode(scope_var=0, probabilities=list(q_fixed))

        def f_value(probs):
            leaf = CategoricalInputNode(
                scope_var=0, probabilities=list(probs)
            )
            return exp_query(leaf, leaf2)

        leaf1 = CategoricalInputNode(scope_var=0, probabilities=list(p_init))
        _, grads1, _ = exp_query_and_grad(leaf1, leaf2)
        assert int(leaf1.id) in grads1.cat_grads
        g = _project_to_simplex_tangent(grads1.cat_grads[int(leaf1.id)])
        g_fd = _fd_gradient_simplex(f_value, p_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)

    def test_leaf_cat_grad_c2_matches_fd(self):
        rng = np.random.default_rng(43)
        p_fixed = rng.dirichlet([1.5, 1.0, 0.7])
        q_init = rng.dirichlet([1.0, 1.2, 0.9])
        leaf1 = CategoricalInputNode(scope_var=0, probabilities=list(p_fixed))

        def f_value(probs):
            leaf = CategoricalInputNode(
                scope_var=0, probabilities=list(probs)
            )
            return exp_query(leaf1, leaf)

        leaf2 = CategoricalInputNode(scope_var=0, probabilities=list(q_init))
        _, _, grads2 = exp_query_and_grad(leaf1, leaf2)
        assert int(leaf2.id) in grads2.cat_grads
        g = _project_to_simplex_tangent(grads2.cat_grads[int(leaf2.id)])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)

    def test_sum_sum_theta_grad_matches_fd(self):
        rng = np.random.default_rng(7)
        p1 = rng.dirichlet([1.0, 1.0, 1.0])
        p2 = rng.dirichlet([1.0, 1.0, 1.0])
        theta_init = np.array([0.42, 0.58])
        q1 = rng.dirichlet([1.0, 1.0, 1.0])
        q2 = rng.dirichlet([1.0, 1.0, 1.0])
        circ2 = SumNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=list(q1)),
                CategoricalInputNode(scope_var=0, probabilities=list(q2)),
            ],
            parameters=[0.4, 0.6],
        )

        def f_value(theta):
            return exp_query(
                SumNode(
                    children=[
                        CategoricalInputNode(scope_var=0, probabilities=list(p1)),
                        CategoricalInputNode(scope_var=0, probabilities=list(p2)),
                    ],
                    parameters=list(theta),
                ),
                circ2,
            )

        circ1 = SumNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=list(p1)),
                CategoricalInputNode(scope_var=0, probabilities=list(p2)),
            ],
            parameters=list(theta_init),
        )
        _, grads1, _ = exp_query_and_grad(circ1, circ2)
        assert int(circ1.id) in grads1.sum_grads
        g = _project_to_simplex_tangent(grads1.sum_grads[int(circ1.id)])
        g_fd = _fd_gradient_simplex(f_value, theta_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)

    def test_sum_sum_phi_grad_matches_fd(self):
        rng = np.random.default_rng(8)
        p1 = rng.dirichlet([1.0, 1.0, 1.0])
        p2 = rng.dirichlet([1.0, 1.0, 1.0])
        circ1 = SumNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=list(p1)),
                CategoricalInputNode(scope_var=0, probabilities=list(p2)),
            ],
            parameters=[0.3, 0.7],
        )
        q1 = rng.dirichlet([1.0, 1.0, 1.0])
        q2 = rng.dirichlet([1.0, 1.0, 1.0])
        phi_init = np.array([0.42, 0.58])

        def f_value(phi):
            return exp_query(
                circ1,
                SumNode(
                    children=[
                        CategoricalInputNode(scope_var=0, probabilities=list(q1)),
                        CategoricalInputNode(scope_var=0, probabilities=list(q2)),
                    ],
                    parameters=list(phi),
                ),
            )

        circ2 = SumNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=list(q1)),
                CategoricalInputNode(scope_var=0, probabilities=list(q2)),
            ],
            parameters=list(phi_init),
        )
        _, _, grads2 = exp_query_and_grad(circ1, circ2)
        assert int(circ2.id) in grads2.sum_grads
        g = _project_to_simplex_tangent(grads2.sum_grads[int(circ2.id)])
        g_fd = _fd_gradient_simplex(f_value, phi_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)

    def test_product_leaf_grad_both_sides(self):
        rng = np.random.default_rng(99)
        p_init = rng.dirichlet([1.0, 1.0])
        q_init = rng.dirichlet([1.0, 1.0])
        other_p = list(rng.dirichlet([1.0, 1.0]))
        other_q = list(rng.dirichlet([1.0, 1.0]))

        def f_c1(probs):
            return exp_query(
                ProductNode(
                    children=[
                        CategoricalInputNode(scope_var=0, probabilities=list(probs)),
                        CategoricalInputNode(scope_var=1, probabilities=other_p),
                    ],
                ),
                ProductNode(
                    children=[
                        CategoricalInputNode(scope_var=0, probabilities=list(q_init)),
                        CategoricalInputNode(scope_var=1, probabilities=other_q),
                    ],
                ),
            )

        def f_c2(probs):
            return exp_query(
                ProductNode(
                    children=[
                        CategoricalInputNode(scope_var=0, probabilities=list(p_init)),
                        CategoricalInputNode(scope_var=1, probabilities=other_p),
                    ],
                ),
                ProductNode(
                    children=[
                        CategoricalInputNode(scope_var=0, probabilities=list(probs)),
                        CategoricalInputNode(scope_var=1, probabilities=other_q),
                    ],
                ),
            )

        circ1 = ProductNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=list(p_init)),
                CategoricalInputNode(scope_var=1, probabilities=other_p),
            ],
        )
        circ2 = ProductNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=list(q_init)),
                CategoricalInputNode(scope_var=1, probabilities=other_q),
            ],
        )
        _, grads1, grads2 = exp_query_and_grad(circ1, circ2)
        g1 = _project_to_simplex_tangent(grads1.cat_grads[0])
        g1_fd = _fd_gradient_simplex(f_c1, p_init, eps=1e-6)
        assert_allclose(g1, g1_fd, atol=1e-5)
        g2 = _project_to_simplex_tangent(grads2.cat_grads[3])
        g2_fd = _fd_gradient_simplex(f_c2, q_init, eps=1e-6)
        assert_allclose(g2, g2_fd, atol=1e-5)
