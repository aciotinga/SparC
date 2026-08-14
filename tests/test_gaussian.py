"""Univariate Gaussian leaves and the all-continuous circuit domain."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    CircuitSerializer,
    GaussianInputNode,
    ProductNode,
    SumNode,
    cw_distance,
    cw_distance_and_grad,
    exp_query,
    expected_squared_distance,
    expected_squared_distance_and_grad,
    gcw_crossterm,
    log_exp_query,
    log_likelihood,
    likelihood,
    mean_log_likelihood_and_grad,
    sample,
)
from sparc.metrics import PNormMetric
from sparc.optim import apply_grads
from sparc.structures import Gaussian, GeneralizedHMM


def _n_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def _n_pdf(x, mu, sigma):
    return math.exp(_n_logpdf(x, mu, sigma))


def test_domain_mismatch_raises():
    cat = CategoricalInputNode(0, [0.5, 0.5])
    g = GaussianInputNode(1, 0.0, 1.0)
    with pytest.raises(ValueError, match="cannot mix discrete and continuous"):
        ProductNode([cat, g])


def test_density_log_density_and_missing():
    g = GaussianInputNode(0, 1.5, 0.5)
    x = np.array([1.5], dtype=np.float64)
    assert likelihood(g, x) == pytest.approx(_n_pdf(1.5, 1.5, 0.5))
    assert log_likelihood(g, x) == pytest.approx(_n_logpdf(1.5, 1.5, 0.5))
    missing = np.array([np.nan], dtype=np.float64)
    assert likelihood(g, missing) == pytest.approx(1.0)
    assert log_likelihood(g, missing) == pytest.approx(0.0)


def test_sampling_moments():
    g = GaussianInputNode(0, 2.0, 0.7)
    xs = g.sample(20_000, seed=1)[:, 0]
    assert xs.dtype == np.float64
    assert xs.mean() == pytest.approx(2.0, abs=0.05)
    assert xs.std(ddof=0) == pytest.approx(0.7, abs=0.05)


def test_cw_w2sq_closed_form_and_fd():
    p = GaussianInputNode(0, 0.0, 1.0)
    q = GaussianInputNode(0, 1.5, 0.5)
    metric = PNormMetric(p=2.0)
    want = (0.0 - 1.5) ** 2 + (1.0 - 0.5) ** 2
    val = cw_distance(p, q, metric=metric)
    assert val == pytest.approx(want)
    val2, grads = cw_distance_and_grad(p, q, metric=metric)
    assert val2 == pytest.approx(want)
    g = grads.cont_grads[q.id]
    eps = 1e-6
    q.set_mu(1.5 + eps)
    plus = cw_distance(p, q, metric=metric)
    q.set_mu(1.5 - eps)
    minus = cw_distance(p, q, metric=metric)
    q.set_mu(1.5)
    assert g[0] == pytest.approx((plus - minus) / (2 * eps), rel=1e-4, abs=1e-6)
    q.set_sigma(0.5 + eps)
    plus = cw_distance(p, q, metric=metric)
    q.set_sigma(0.5 - eps)
    minus = cw_distance(p, q, metric=metric)
    q.set_sigma(0.5)
    assert g[1] == pytest.approx((plus - minus) / (2 * eps), rel=1e-4, abs=1e-6)


def test_cw_p1_raises():
    p = GaussianInputNode(0, 0.0, 1.0)
    q = GaussianInputNode(0, 0.0, 1.0)
    with pytest.raises(ValueError, match="p=2|W_2"):
        cw_distance(p, q, metric=PNormMetric(p=1.0))


def test_exp_query_inner_product():
    p = GaussianInputNode(0, 0.0, 1.0)
    q = GaussianInputNode(0, 1.0, 2.0)
    var = 1.0 + 4.0
    want = _n_pdf(0.0, 1.0, math.sqrt(var))
    assert exp_query(p, q) == pytest.approx(want)
    assert log_exp_query(p, q) == pytest.approx(math.log(want))


def test_esd_two_sigma_sq():
    g = GaussianInputNode(0, 3.0, 1.5)
    metric = PNormMetric(p=2.0)
    want = 2.0 * 1.5 ** 2
    assert expected_squared_distance(g, metric=metric) == pytest.approx(want)
    val, grads = expected_squared_distance_and_grad(g, metric=metric)
    assert val == pytest.approx(want)
    # d/dσ (2σ²) = 4σ
    assert grads.cont_grads[g.id][1] == pytest.approx(4.0 * 1.5)


def test_mixture_compile_and_likelihood():
    mix = SumNode(
        [GaussianInputNode(0, 0.0, 1.0), GaussianInputNode(0, 4.0, 1.0)],
        [0.4, 0.6],
    )
    x = np.array([[0.0], [4.0], [2.0]], dtype=np.float64)
    compiled = mix.compile()
    assert_allclose(mix.log_likelihood(x), compiled.log_likelihood(x))
    p0 = _n_pdf(0.0, 0.0, 1.0)
    p1 = _n_pdf(0.0, 4.0, 1.0)
    assert likelihood(mix, x[0]) == pytest.approx(0.4 * p0 + 0.6 * p1)


def test_object_graph_compile_parity():
    leaf0 = GaussianInputNode(0, -0.2, 0.8)
    leaf1 = GaussianInputNode(1, 1.1, 1.3)
    root = ProductNode([leaf0, leaf1])
    compiled = root.compile()
    x = np.array([[-0.2, 1.1], [0.0, 0.0]], dtype=np.float64)
    assert_allclose(root.log_likelihood(x), compiled.log_likelihood(x))
    metric = PNormMetric(p=2.0)
    other = ProductNode(
        [GaussianInputNode(0, 0.0, 1.0), GaussianInputNode(1, 0.0, 1.0)]
    )
    other_c = other.compile()
    assert cw_distance(root, other, metric=metric) == pytest.approx(
        cw_distance(compiled, other_c, metric=metric)
    )
    assert exp_query(root, other) == pytest.approx(exp_query(compiled, other_c))
    assert expected_squared_distance(root, metric=metric) == pytest.approx(
        expected_squared_distance(compiled, metric=metric)
    )


def test_serializer_and_clone():
    g = GaussianInputNode(0, -1.0, 2.5)
    cloned = g.clone()
    assert cloned is not g
    assert cloned.mu_value() == pytest.approx(-1.0)
    assert cloned.sigma_value() == pytest.approx(2.5)
    restored = CircuitSerializer.loads(CircuitSerializer.dumps(g))
    assert restored.mu_value() == pytest.approx(-1.0)
    assert restored.sigma_value() == pytest.approx(2.5)


def test_gcw_rejects_continuous():
    g = GaussianInputNode(0, 0.0, 1.0)
    with pytest.raises(ValueError, match="continuous"):
        gcw_crossterm(g, g)


def test_apply_grads_and_cw_minimization():
    p = GaussianInputNode(0, 0.0, 1.0)
    q = GaussianInputNode(0, 3.0, 2.0)
    metric = PNormMetric(p=2.0)
    before = cw_distance(p, q, metric=metric)
    for _ in range(40):
        _, grads = cw_distance_and_grad(p, q, metric=metric)
        apply_grads(q, grads, lr=0.1, ascent=False)
    after = cw_distance(p, q, metric=metric)
    assert after < before
    assert q.mu_value() == pytest.approx(0.0, abs=0.2)
    assert q.sigma_value() == pytest.approx(1.0, abs=0.2)


def test_mle_grad_matches_fd():
    g = GaussianInputNode(0, 0.2, 1.1)
    x = np.array([[0.5]], dtype=np.float64)
    ll, grads = mean_log_likelihood_and_grad(g, x)
    assert ll == pytest.approx(_n_logpdf(0.5, 0.2, 1.1))
    cg = grads.cont_grads[g.id]
    eps = 1e-6
    g.set_mu(0.2 + eps)
    plus = float(np.asarray(log_likelihood(g, x)).reshape(-1)[0])
    g.set_mu(0.2 - eps)
    minus = float(np.asarray(log_likelihood(g, x)).reshape(-1)[0])
    g.set_mu(0.2)
    assert cg[0] == pytest.approx((plus - minus) / (2 * eps), rel=1e-4, abs=1e-6)


def test_differentiable_reparam_vjp():
    g = GaussianInputNode(0, 1.0, 2.0)
    draws = sample(g, 8, seed=0, differentiable=True)
    assert draws.one_hot.shape[1] == 0
    assert draws.assignments.dtype == np.float64
    up = np.ones_like(draws.assignments)
    grads = draws.vjp(up)
    assert g.id in grads.cont_grads
    assert grads.cont_grads[g.id].shape == (2,)
    compiled = g.compile().sample(8, seed=0, differentiable=True)
    assert compiled.assignments.dtype == np.float64


def test_structure_gaussian_hmm():
    circuit = GeneralizedHMM(
        seq_length=3, num_latents=2, input_dist=Gaussian(mu=0.0, sigma=1.0), seed=0
    )
    assert circuit.circuit_domain == 2
    xs = circuit.sample(4, seed=1)
    assert xs.dtype == np.float64
    assert xs.shape[1] >= 3
