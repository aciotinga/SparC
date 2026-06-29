"""Cross-cutting property tests: normalization, idempotence, determinism."""

from __future__ import annotations

import gc

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
        ProductNode,
    SumNode,
    gcw_coupling_circuit,
    likelihood,
    log_likelihood,
    sample,
)
from sparc.builders import EmbeddingBuilder, RegionEmbeddingBuilder, RandomRegionGraph
from tests.gcw_helpers import pollute_heap_with_couplings
from tests.sparc_helpers import (
    assignment_array,
    exact_marginal,
    exact_total_mass,
    make_sum,
    sum_mixture_marginal,
    walk_pc_invariants,
)

pytestmark = pytest.mark.property


class TestNormalizationProperties:
    @pytest.mark.parametrize("seed", range(6))
    def test_embedding_builder_normalizes(self, seed):
        rng = np.random.default_rng(seed)
        circuit = EmbeddingBuilder(
            num_vars=5,
            num_categories=3,
            sum_arity=2,
            prod_arity=2,
            sum_concentration=1.0,
            sum_reuse_probability=0.2,
            prod_reuse_probability=0.2,
            input_distribution="categorical",
            alpha=1.0,
        ).build()
        walk_pc_invariants(circuit)
        assert_allclose(exact_total_mass(circuit), 1.0, rtol=0, atol=1e-8)

    def test_region_embedding_normalizes(self):
        rg = RandomRegionGraph(
            frozenset(range(4)), partitions_per_region=1, sub_regions_per_partition=2
        )
        region = rg.generate(frozenset(range(4)))
        circuit = RegionEmbeddingBuilder(
            region,
            num_categories=3,
            block_size=2,
            sum_concentration=1.0,
            input_distribution="categorical",
            alpha=1.0,
        ).build()
        walk_pc_invariants(circuit)
        assert_allclose(exact_total_mass(circuit), 1.0, rtol=0, atol=1e-8)


class TestDeterminismProperties:
    def test_sample_seed_stable_under_heap_noise(self):
        circ1, circ2 = make_sum(
            2, 0, [[0.8, 0.2], [0.3, 0.7]], [0.5, 0.5], id_base=0
        ), make_sum(
            5, 0, [[0.6, 0.4], [0.1, 0.9]], [0.4, 0.6], id_base=10
        )
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=8, seed=0)
        gc.collect()
        coupling = gcw_coupling_circuit(circ1, circ2)
        a = coupling.sample(500, seed=42)
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=8, seed=1)
        b = coupling.sample(500, seed=42)
        assert (a == b).all()

    def test_likelihood_independent_of_build_order(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.3, 0.7])
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=5, seed=2)
        assert likelihood(leaf, assignment_array({0: 1})) == pytest.approx(0.7)


class TestIdempotenceProperties:
    def test_double_propagate_scope_unchanged(self):
        root = ProductNode(
            children=[
                CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5]),
                CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5]),
            ],
        )
        root.propagate_scope()
        scope1 = root.scope_as_list()
        root.propagate_scope()
        assert root.scope_as_list() == scope1

    def test_log_likelihood_is_log_of_likelihood(self):
        circuit = make_sum(1, 0, [[0.6, 0.4], [0.2, 0.8]], [0.5, 0.5], id_base=10)
        row = assignment_array({0: 0})
        assert circuit.log_likelihood(row) == pytest.approx(
            np.log(circuit.likelihood(row))
        )


class TestMarginalPreservation:
    def test_mixture_marginal_matches_exact(self):
        probs = [[0.9, 0.1], [0.2, 0.8]]
        weights = [0.35, 0.65]
        circuit = make_sum(10, 0, probs, weights, id_base=100)
        assert_allclose(
            exact_marginal(circuit, 0),
            sum_mixture_marginal(probs, weights),
            atol=1e-10,
        )
