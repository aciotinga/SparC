"""Bernoulli integration across likelihood, gradients, queries, and sampling."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    BernoulliInputNode,
        ProductNode,
    SumNode,
    cw_distance,
    expected_squared_distance,
    gcw_crossterm,
    mean_log_likelihood_and_grad,
)
from tests.sparc_helpers import (
    assignment_array,
    empirical_marginal,
    exact_marginal,
    fd_gradient_simplex,
    project_to_simplex_tangent,
)

pytestmark = pytest.mark.integration


class TestBernoulliLikelihoodGrad:
    def test_bernoulli_ll_grad_matches_fd(self):
        p_init = 0.35
        dataset = np.array([[0], [1], [1], [0]], dtype=np.int32)

        def f(probs):
            node = BernoulliInputNode(scope_var=0, p=float(probs[1]))
            mean_ll, _ = mean_log_likelihood_and_grad(node, dataset)
            return mean_ll

        node = BernoulliInputNode(scope_var=0, p=p_init)
        _, grads = mean_log_likelihood_and_grad(node, dataset)
        g_an = project_to_simplex_tangent(grads.cat_grads[0])
        g_fd = fd_gradient_simplex(f, [1.0 - p_init, p_init], eps=1e-6)
        assert_allclose(g_an, g_fd, atol=1e-5)

    def test_bernoulli_in_product_circuit(self):
        b = BernoulliInputNode(scope_var=0, p=0.4)
        c = BernoulliInputNode(scope_var=1, p=0.7)
        circuit = ProductNode(children=[b, c])
        assert circuit.likelihood(assignment_array({0: 1, 1: 1})) == pytest.approx(0.4 * 0.7)


class TestBernoulliQueries:
    def test_cw_leaf_bernoulli(self):
        b1 = BernoulliInputNode(scope_var=0, p=0.3)
        b2 = BernoulliInputNode(scope_var=0, p=0.8)
        d = cw_distance(b1, b2)
        assert np.isfinite(d)
        assert d >= 0.0

    def test_esd_bernoulli(self):
        b = BernoulliInputNode(scope_var=0, p=0.4)
        val = expected_squared_distance(b)
        assert np.isfinite(val)
        assert val >= 0.0

    def test_gcw_bernoulli(self):
        b1 = BernoulliInputNode(scope_var=0, p=0.25)
        b2 = BernoulliInputNode(scope_var=0, p=0.75)
        cross = gcw_crossterm(b1, b2)
        assert np.isfinite(cross)
        assert cross >= -1e-8


class TestBernoulliSampling:
    def test_empirical_mean(self):
        node = BernoulliInputNode(scope_var=0, p=0.35)
        draws = node.sample(30_000, seed=0)
        assert_allclose(empirical_marginal(draws, 0, 2)[1], 0.35, atol=0.02)

    def test_exact_marginal(self):
        node = BernoulliInputNode(scope_var=0, p=0.6)
        circuit = node
        assert_allclose(exact_marginal(circuit, 0), [0.4, 0.6], atol=1e-10)
