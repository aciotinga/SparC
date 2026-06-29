"""Integration tests: log_exp parity, batched-vs-dict eval, clone independence,
simplex_step projections, MLE ascent, and a DRO loop smoke test.
"""

import math
import random

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
        ProductNode,
    SumNode,
    cw_distance,
    exp_query,
    gcw_coupling_circuit,
    gcw_crossterm,
    log_exp_query,
    log_exp_query_and_grad,
)
from sparc.builders import EmbeddingBuilder, RandomRegionGraph, RegionEmbeddingBuilder
from sparc.optim import MLETrainer, apply_grads, simplex_step
from tests.sparc_helpers import assignment_array


def _single_var_sum(ids, scope_var, probs_list, params):
    leaves = [
        CategoricalInputNode(id=i, scope_var=scope_var, probabilities=p)
        for i, p in zip(ids, probs_list)
    ]
    return SumNode(id=ids[-1] + 1, children=leaves, parameters=params)


class TestLogExpParity:
    def test_log_exp_equals_log_of_exp(self):
        circ1 = _single_var_sum([0, 1], 0, [[0.7, 0.3], [0.2, 0.8]], [0.4, 0.6])
        circ2 = _single_var_sum([3, 4], 0, [[0.5, 0.5], [0.1, 0.9]], [0.55, 0.45])
        e = exp_query(circ1, circ2)
        le = log_exp_query(circ1, circ2)
        assert_allclose(le, math.log(e), rtol=0, atol=1e-12)

    def test_log_exp_grad_value_matches_forward(self):
        circ1 = _single_var_sum([0, 1], 0, [[0.7, 0.3], [0.2, 0.8]], [0.4, 0.6])
        circ2 = _single_var_sum([3, 4], 0, [[0.5, 0.5], [0.1, 0.9]], [0.55, 0.45])
        val, _, _ = log_exp_query_and_grad(circ1, circ2)
        assert_allclose(val, log_exp_query(circ1, circ2), rtol=0, atol=1e-12)


class TestBatchedVsDict:
    def test_batched_matches_per_row(self):
        np.random.seed(0)
        random.seed(0)
        circuit = EmbeddingBuilder(
            num_vars=6, num_categories=4, sum_arity=2, prod_arity=2,
            sum_concentration=1.0, sum_reuse_probability=0.0,
            prod_reuse_probability=0.0, input_distribution="categorical", alpha=1.0,
        ).build()
        nvars = len(circuit.scope_as_list())
        data = np.random.randint(0, 4, size=(32, nvars)).astype(np.int32)

        batched = circuit.compile().log_likelihood(data)
        per_row = np.array([circuit.log_likelihood(data[i]) for i in range(data.shape[0])])
        assert_allclose(batched, per_row, rtol=0, atol=1e-10)


class TestClone:
    def test_clone_is_independent(self):
        leaf_a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        leaf_b = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.8])
        root = SumNode(children=[leaf_a, leaf_b], parameters=[0.6, 0.4])
        circuit = root
        clone = circuit.clone()

        before = circuit.likelihood(assignment_array({0: 0}))
        # mutate the clone's sum parameters
        clone.set_parameters_list([0.1, 0.9])
        after = circuit.likelihood(assignment_array({0: 0}))
        assert before == after  # original untouched
        assert clone.likelihood(assignment_array({0: 0})) != after

    def test_clone_preserves_values(self):
        np.random.seed(1)
        random.seed(1)
        circuit = EmbeddingBuilder(
            num_vars=5, num_categories=3, sum_arity=2, prod_arity=2,
            sum_concentration=1.0, sum_reuse_probability=0.3,
            prod_reuse_probability=0.3, input_distribution="categorical", alpha=1.0,
        ).build()
        width = max(circuit.scope_as_list()) + 1
        asg = np.zeros(width, dtype=np.int32)
        assert_allclose(circuit.clone().log_likelihood(asg),
                        circuit.log_likelihood(asg), rtol=0, atol=1e-12)


class TestSimplexStep:
    @pytest.mark.parametrize("method", ["tangent", "euclidean"])
    def test_step_stays_on_simplex(self, method):
        out = simplex_step([0.2, 0.3, 0.5], [1.0, -0.5, 0.2], 0.1,
                           ascent=True, method=method)
        assert abs(sum(out) - 1.0) < 1e-12
        assert all(x >= 0.0 for x in out)

    def test_ascent_descent_opposite(self):
        up = np.array(simplex_step([0.5, 0.5], [1.0, -1.0], 0.1, ascent=True))
        dn = np.array(simplex_step([0.5, 0.5], [1.0, -1.0], 0.1, ascent=False))
        # ascent raises component 0; descent lowers it
        assert up[0] > 0.5 > dn[0]

    def test_euclidean_projection_sparse(self):
        # a step that pushes mass negative should project to the simplex boundary
        out = simplex_step([0.9, 0.05, 0.05], [-5.0, 1.0, 1.0], 1.0,
                           ascent=True, method="euclidean")
        assert abs(sum(out) - 1.0) < 1e-12
        assert min(out) >= 0.0


class TestMLE:
    def test_mle_increases_likelihood(self):
        np.random.seed(2)
        random.seed(2)

        def make():
            return EmbeddingBuilder(
                num_vars=6, num_categories=3, sum_arity=2, prod_arity=2,
                sum_concentration=1.0, sum_reuse_probability=0.0,
                prod_reuse_probability=0.0, input_distribution="categorical", alpha=1.0,
            ).build()

        truth = make()
        data = truth.sample(300, seed=11)
        model = make()
        trainer = MLETrainer(model, lr=0.5)
        history = trainer.fit(data, epochs=30)
        assert history[-1] > history[0]


class TestGCWCouplingCircuit:
    def _empirical_marginal(self, draws, var, n_outcomes):
        counts = np.zeros(n_outcomes)
        for r in range(draws.shape[0]):
            counts[draws[r, var]] += 1
        return counts / draws.shape[0]

    def test_leaf_coupling_marginals(self):
        p = [0.3, 0.7]
        q = [0.55, 0.45]
        leaf1 = CategoricalInputNode(scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(scope_var=0, probabilities=q)
        coupling = gcw_coupling_circuit(leaf1, leaf2)

        # disjoint variable namespaces: P keeps var 0, Q is shifted to var 1
        assert set(coupling.scope_as_list()) == {0, 1}

        draws = coupling.sample(40000, seed=0)
        mp = self._empirical_marginal(draws, 0, 2)
        mq = self._empirical_marginal(draws, 1, 2)
        assert_allclose(mp, p, atol=0.02)
        assert_allclose(mq, q, atol=0.02)

    def test_sum_coupling_is_valid_pc_and_marginals(self):
        c1a = CategoricalInputNode(scope_var=0, probabilities=[0.8, 0.2])
        c1b = CategoricalInputNode(scope_var=0, probabilities=[0.3, 0.7])
        circ1 = SumNode(children=[c1a, c1b], parameters=[0.5, 0.5])
        c2a = CategoricalInputNode(scope_var=0, probabilities=[0.6, 0.4])
        c2b = CategoricalInputNode(scope_var=0, probabilities=[0.1, 0.9])
        circ2 = SumNode(children=[c2a, c2b], parameters=[0.4, 0.6])

        coupling = gcw_coupling_circuit(circ1, circ2)
        # P marginal over var 0 = 0.5*[0.8,0.2] + 0.5*[0.3,0.7]
        p_marg = [0.5 * 0.8 + 0.5 * 0.3, 0.5 * 0.2 + 0.5 * 0.7]
        q_marg = [0.4 * 0.6 + 0.6 * 0.1, 0.4 * 0.4 + 0.6 * 0.9]
        draws = coupling.sample(40000, seed=1)
        assert_allclose(self._empirical_marginal(draws, 0, 2), p_marg, atol=0.02)
        assert_allclose(self._empirical_marginal(draws, 1, 2), q_marg, atol=0.02)
        for r in range(50):
            assert coupling.likelihood(draws[r]) > 0.0

    def test_product_coupling_disjoint_vars(self):
        circ1 = ProductNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=[0.4, 0.6]),
                CategoricalInputNode(scope_var=1, probabilities=[0.7, 0.3]),
            ],
        )
        circ2 = ProductNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.8]),
                CategoricalInputNode(scope_var=1, probabilities=[0.55, 0.45]),
            ],
        )
        coupling = gcw_coupling_circuit(circ1, circ2)
        # circ1 has vars {0,1}; circ2 shifted by 2 -> {2,3}
        assert set(coupling.scope_as_list()) == {0, 1, 2, 3}
        draws = coupling.sample(2000, seed=2)
        for r in range(50):
            row = draws[r]
            assert (row[[0, 1, 2, 3]] >= 0).all()


class TestDROSmoke:
    def test_dro_loop_runs(self):
        random.seed(0)
        np.random.seed(0)
        rg = RandomRegionGraph(
            frozenset(range(5)), partitions_per_region=1, sub_regions_per_partition=2
        )
        region = rg.generate(frozenset(range(5)))
        p_hat = RegionEmbeddingBuilder(
            region, num_categories=3, block_size=2,
            sum_concentration=1.0, input_distribution="categorical", alpha=1.0,
        ).build()
        p_theta = p_hat.clone()
        q_phi = p_hat.clone()

        for _ in range(3):
            _, _, grad_phi = log_exp_query_and_grad(p_theta, q_phi)
            apply_grads(q_phi, grad_phi, 1e-2, ascent=False)
            _, grad_theta, _ = log_exp_query_and_grad(p_theta, q_phi)
            apply_grads(p_theta, grad_theta, 1e-2, ascent=True)

        assert math.isfinite(log_exp_query(p_theta, q_phi))
        assert math.isfinite(cw_distance(p_hat, q_phi))

    def test_sample_based_dro_theta_step_runs(self):
        random.seed(0)
        np.random.seed(0)
        rg = RandomRegionGraph(
            frozenset(range(5)), partitions_per_region=1, sub_regions_per_partition=2
        )
        region = rg.generate(frozenset(range(5)))
        p_hat = RegionEmbeddingBuilder(
            region, num_categories=3, block_size=2,
            sum_concentration=1.0, input_distribution="categorical", alpha=1.0,
        ).build()
        p_theta = p_hat.clone()
        q_phi = p_hat.clone()

        for _ in range(3):
            _, _, grad_phi = log_exp_query_and_grad(p_theta, q_phi)
            apply_grads(q_phi, grad_phi, 1e-2, ascent=False)
            q_samples = q_phi.sample(50, seed=0)
            _, grad_theta = p_theta.mean_log_likelihood_and_grad(q_samples)
            apply_grads(p_theta, grad_theta, 1e-2, ascent=True)

        assert math.isfinite(log_exp_query(p_theta, q_phi))
        assert math.isfinite(cw_distance(p_hat, q_phi))
