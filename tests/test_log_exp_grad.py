"""Gradient tests for log_exp_query_and_grad (previously forward-only)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
    log_exp_query,
    log_exp_query_and_grad,
)
from tests.sparc_helpers import (
    fd_gradient_simplex,
    project_to_simplex_tangent,
)

pytestmark = pytest.mark.queries


class TestLogExpForwardParity:
    def test_matches_log_of_exp(self):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.7, 0.3])
        c1b = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.2, 0.8])
        circ1 = SumNode(id=2, children=[c1a, c1b], parameters=[0.4, 0.6])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.5, 0.5])
        c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=[0.1, 0.9])
        circ2 = SumNode(id=5, children=[c2a, c2b], parameters=[0.55, 0.45])
        from sparc import exp_query

        assert_allclose(
            log_exp_query(circ1, circ2),
            np.log(exp_query(circ1, circ2)),
            rtol=0,
            atol=1e-12,
        )

    def test_and_grad_value_matches_forward(self):
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.3, 0.7])
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.55, 0.45])
        val, g1, g2 = log_exp_query_and_grad(leaf1, leaf2)
        assert_allclose(val, log_exp_query(leaf1, leaf2), rtol=0, atol=1e-12)
        assert g1.value == pytest.approx(val, rel=0, abs=1e-12)


class TestLogExpGradients:
    def test_leaf_grad_circuit2_matches_fd(self):
        rng = np.random.default_rng(11)
        p_fixed = rng.dirichlet([1.0, 1.0])
        q_init = rng.dirichlet([1.0, 1.0])
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p_fixed))

        def f(q):
            leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=list(q))
            return log_exp_query(leaf1, leaf2)

        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=list(q_init))
        _, _, g2 = log_exp_query_and_grad(leaf1, leaf2)
        assert 1 in g2.cat_grads
        g_an = project_to_simplex_tangent(g2.cat_grads[1])
        g_fd = fd_gradient_simplex(f, q_init, eps=1e-6)
        assert_allclose(g_an, g_fd, atol=1e-5)

    def test_leaf_grad_circuit1_matches_fd(self):
        rng = np.random.default_rng(12)
        p_init = rng.dirichlet([1.0, 1.0, 1.0])
        q_fixed = rng.dirichlet([1.0, 1.0, 1.0])
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=list(q_fixed))

        def f(p):
            leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p))
            return log_exp_query(leaf1, leaf2)

        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p_init))
        _, g1, _ = log_exp_query_and_grad(leaf1, leaf2)
        g_an = project_to_simplex_tangent(g1.cat_grads[0])
        g_fd = fd_gradient_simplex(f, p_init, eps=1e-6)
        assert_allclose(g_an, g_fd, atol=1e-5)

    def test_sum_sum_grad_matches_fd(self):
        rng = np.random.default_rng(3)
        p1 = rng.dirichlet([1.0, 1.0])
        p2 = rng.dirichlet([1.0, 1.0])
        circ1 = SumNode(
            id=2,
            children=[
                CategoricalInputNode(id=0, scope_var=0, probabilities=list(p1)),
                CategoricalInputNode(id=1, scope_var=0, probabilities=list(p2)),
            ],
            parameters=[0.35, 0.65],
        )
        q_init = np.array([0.42, 0.58])

        def f(phi):
            c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.5, 0.5])
            c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=[0.1, 0.9])
            circ2 = SumNode(id=5, children=[c2a, c2b], parameters=list(phi))
            return log_exp_query(circ1, circ2)

        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.5, 0.5])
        c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=[0.1, 0.9])
        circ2 = SumNode(id=5, children=[c2a, c2b], parameters=list(q_init))
        _, _, g2 = log_exp_query_and_grad(circ1, circ2)
        g_an = project_to_simplex_tangent(g2.sum_grads[5])
        g_fd = fd_gradient_simplex(f, q_init, eps=1e-6)
        assert_allclose(g_an, g_fd, atol=1e-5)

    def test_product_grad_matches_fd(self):
        rng = np.random.default_rng(8)
        c1a = CategoricalInputNode(
            id=0, scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0]))
        )
        c1b = CategoricalInputNode(
            id=1, scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0]))
        )
        circ1 = ProductNode(id=2, children=[c1a, c1b])
        q_init = rng.dirichlet([1.0, 1.0])

        def f(probs):
            c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=list(probs))
            c2b = CategoricalInputNode(id=4, scope_var=1, probabilities=[0.55, 0.45])
            return log_exp_query(circ1, ProductNode(id=5, children=[c2a, c2b]))

        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=list(q_init))
        c2b = CategoricalInputNode(id=4, scope_var=1, probabilities=[0.55, 0.45])
        circ2 = ProductNode(id=5, children=[c2a, c2b])
        _, _, g2 = log_exp_query_and_grad(circ1, circ2)
        g_an = project_to_simplex_tangent(g2.cat_grads[3])
        g_fd = fd_gradient_simplex(f, q_init, eps=1e-6)
        assert_allclose(g_an, g_fd, atol=1e-5)
