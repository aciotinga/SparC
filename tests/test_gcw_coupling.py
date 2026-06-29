"""Rigorous tests for ``gcw_coupling_circuit`` materialization and sampling.

These tests target properties that the existing GCW *value* tests (crossterm,
gradients) do not exercise:

* Marginal preservation of the returned coupling PC w.r.t. both input circuits
* Valid-PC structure (normalized sum weights / leaf probabilities)
* Sampling correctness and determinism
* Order- and heap-sensitive stress (memoization isolation across builds)

If a memoization key is lossy (e.g. packing 64-bit object addresses into 32-bit
slots), several tests here should fail intermittently or consistently on 64-bit
Linux while value-only GCW tests still pass.
"""

from __future__ import annotations

import gc
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
        ProductNode,
    SumNode,
    gcw_coupling_circuit,
)
from tests.sparc_helpers import assignment_array
from tests.gcw_helpers import (
    empirical_marginal,
    exact_marginal,
    make_categorical,
    make_product,
    make_sum,
    nw_coupling_dense,
    pollute_heap_with_couplings,
    q_var_offset,
    sum_mixture_marginal,
    var_cardinalities,
    walk_pc_invariants,
)

pytestmark = pytest.mark.gcw


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _canonical_sum_sum_pair():
    """The regression fixture from ``test_sparc_extras`` (sum x sum, 1 shared var)."""
    c1a = make_categorical(0, 0, [0.8, 0.2])
    c1b = make_categorical(1, 0, [0.3, 0.7])
    circ1 = SumNode(children=[c1a, c1b], parameters=[0.5, 0.5])
    c2a = make_categorical(3, 0, [0.6, 0.4])
    c2b = make_categorical(4, 0, [0.1, 0.9])
    circ2 = SumNode(children=[c2a, c2b], parameters=[0.4, 0.6])
    return circ1, circ2


def _leaf_pair(p=None, q=None):
    p = p or [0.3, 0.7]
    q = q or [0.55, 0.45]
    return (
        make_categorical(0, 0, p),
        make_categorical(1, 0, q),
    )


def _product_pair():
    circ1 = make_product(
        2,
        [(0, [0.4, 0.6]), (1, [0.7, 0.3])],
    )
    circ2 = make_product(
        5,
        [(0, [0.2, 0.8]), (1, [0.55, 0.45])],
    )
    return circ1, circ2


def _nested_sum_of_products(rng: np.random.Generator):
    leaves1 = [
        make_categorical(
            10 + i, i % 2, list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        for i in range(4)
    ]
    prod_a = ProductNode(children=[leaves1[0], leaves1[1]])
    prod_b = ProductNode(children=[leaves1[2], leaves1[3]])
    circ1 = SumNode(children=[prod_a, prod_b], parameters=[0.45, 0.55])

    leaves2 = [
        make_categorical(
            30 + i, i % 2, list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        for i in range(4)
    ]
    prod_c = ProductNode(children=[leaves2[0], leaves2[1]])
    prod_d = ProductNode(children=[leaves2[2], leaves2[3]])
    circ2 = SumNode(children=[prod_c, prod_d], parameters=[0.3, 0.7])
    return circ1, circ2


# ---------------------------------------------------------------------------
# Structure / validity
# ---------------------------------------------------------------------------


class TestCouplingStructure:
    def test_leaf_coupling_scope_disjoint(self):
        leaf1, leaf2 = _leaf_pair()
        coupling = gcw_coupling_circuit(leaf1, leaf2)
        assert set(coupling.scope_as_list()) == {0, 1}

    def test_product_coupling_scope_shift(self):
        circ1, circ2 = _product_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        assert set(coupling.scope_as_list()) == {0, 1, 2, 3}

    def test_coupling_is_valid_pc_tree(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        walk_pc_invariants(coupling)

    @pytest.mark.parametrize(
        "builder",
        [
            _leaf_pair,
            _canonical_sum_sum_pair,
            _product_pair,
            lambda: _nested_sum_of_products(np.random.default_rng(0)),
        ],
    )
    def test_coupling_pc_invariants_across_topologies(self, builder):
        c1, c2 = builder()
        walk_pc_invariants(gcw_coupling_circuit(c1, c2))

    def test_coupling_total_mass_is_one(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        scope = sorted(coupling.scope_as_list())
        cards = var_cardinalities(coupling)
        total = 0.0
        import itertools

        for values in itertools.product(*[range(cards[v]) for v in scope]):
            row = assignment_array({scope[i]: values[i] for i in range(len(scope))})
            total += coupling.likelihood(row)
        assert_allclose(total, 1.0, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Exact marginals (enumeration, no sampling noise)
# ---------------------------------------------------------------------------


class TestCouplingExactMarginals:
    def test_leaf_coupling_exact_marginals_match_inputs(self):
        p = [0.3, 0.7]
        q = [0.55, 0.45]
        leaf1, leaf2 = _leaf_pair(p, q)
        coupling = gcw_coupling_circuit(leaf1, leaf2)
        assert_allclose(exact_marginal(coupling, 0), p, atol=1e-10)
        assert_allclose(exact_marginal(coupling, 1), q, atol=1e-10)

    def test_sum_sum_exact_marginals_match_mixtures(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        p_marg = sum_mixture_marginal([[0.8, 0.2], [0.3, 0.7]], [0.5, 0.5])
        q_marg = sum_mixture_marginal([[0.6, 0.4], [0.1, 0.9]], [0.4, 0.6])
        off = q_var_offset(circ1)
        assert_allclose(exact_marginal(coupling, 0), p_marg, atol=1e-10)
        assert_allclose(exact_marginal(coupling, off + 0), q_marg, atol=1e-10)

    def test_product_coupling_exact_marginals_match_factors(self):
        circ1, circ2 = _product_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        assert_allclose(exact_marginal(coupling, 0), [0.4, 0.6], atol=1e-10)
        assert_allclose(exact_marginal(coupling, 1), [0.7, 0.3], atol=1e-10)
        assert_allclose(exact_marginal(coupling, 2), [0.2, 0.8], atol=1e-10)
        assert_allclose(exact_marginal(coupling, 3), [0.55, 0.45], atol=1e-10)

    def test_leaf_nw_plan_matches_builtin_solver(self):
        p = [0.2, 0.3, 0.5]
        q = [0.4, 0.1, 0.5]
        leaf1, leaf2 = _leaf_pair(p, q)
        coupling = gcw_coupling_circuit(leaf1, leaf2)
        plan = nw_coupling_dense(p, q)
        # Joint probabilities for (x=i, y=j) should equal the NW plan entries.
        for i in range(len(p)):
            for j in range(len(q)):
                joint = coupling.likelihood(assignment_array({0: i, 1: j}))
                assert_allclose(joint, plan[i, j], rtol=0, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_random_nested_exact_marginals(self, seed):
        rng = np.random.default_rng(seed)
        circ1, circ2 = _nested_sum_of_products(rng)
        coupling = gcw_coupling_circuit(circ1, circ2)
        c1 = circ1
        c2 = circ2
        off = q_var_offset(circ1)
        for var in sorted(circ1.scope_as_list()):
            assert_allclose(
                exact_marginal(coupling, var),
                exact_marginal(c1, var),
                atol=1e-9,
            )
        for var in sorted(circ2.scope_as_list()):
            assert_allclose(
                exact_marginal(coupling, off + var),
                exact_marginal(c2, var),
                atol=1e-9,
            )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestCouplingSampling:
    N_SAMPLES = 40_000
    ATOL = 0.02

    def test_sum_sum_empirical_marginals(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        p_marg = sum_mixture_marginal([[0.8, 0.2], [0.3, 0.7]], [0.5, 0.5])
        q_marg = sum_mixture_marginal([[0.6, 0.4], [0.1, 0.9]], [0.4, 0.6])
        off = q_var_offset(circ1)
        draws = coupling.sample(self.N_SAMPLES, seed=1)
        assert_allclose(
            empirical_marginal(draws, 0, 2), p_marg, atol=self.ATOL
        )
        assert_allclose(
            empirical_marginal(draws, off + 0, 2), q_marg, atol=self.ATOL
        )

    def test_leaf_empirical_marginals(self):
        p = [0.3, 0.7]
        q = [0.55, 0.45]
        leaf1, leaf2 = _leaf_pair(p, q)
        coupling = gcw_coupling_circuit(leaf1, leaf2)
        draws = coupling.sample(self.N_SAMPLES, seed=0)
        assert_allclose(empirical_marginal(draws, 0, 2), p, atol=self.ATOL)
        assert_allclose(empirical_marginal(draws, 1, 2), q, atol=self.ATOL)

    def test_sampled_assignments_have_positive_likelihood(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        draws = coupling.sample(100, seed=99)
        for r in range(draws.shape[0]):
            mass = coupling.likelihood(draws[r])
            assert mass > 0.0
            assert math.isfinite(mass)

    def test_sampling_is_deterministic_with_seed(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        a = coupling.sample(200, seed=123)
        b = coupling.sample(200, seed=123)
        assert (a == b).all()

    def test_sampling_differs_across_seeds(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        a = coupling.sample(500, seed=1)
        b = coupling.sample(500, seed=2)
        assert not (a == b).all()


# ---------------------------------------------------------------------------
# Memoization / build-order stress (would catch pointer-key collisions)
# ---------------------------------------------------------------------------


@pytest.mark.gcw_stress
class TestCouplingMemoizationStress:
    """Stress patterns that expose cross-build cache poisoning."""

    N_SAMPLES = 40_000
    ATOL = 0.02

    def _assert_sum_sum_marginals(self, coupling, circ1, circ2, *, seed: int):
        p_marg = sum_mixture_marginal([[0.8, 0.2], [0.3, 0.7]], [0.5, 0.5])
        q_marg = sum_mixture_marginal([[0.6, 0.4], [0.1, 0.9]], [0.4, 0.6])
        off = q_var_offset(circ1)
        draws = coupling.sample(self.N_SAMPLES, seed=seed)
        assert_allclose(
            empirical_marginal(draws, 0, 2), p_marg, atol=self.ATOL
        )
        assert_allclose(
            empirical_marginal(draws, off + 0, 2), q_marg, atol=self.ATOL
        )

    def test_prior_leaf_coupling_does_not_poison_sum_sum(self):
        """Mirrors ``test_sparc_extras`` ordering: leaf test then sum-sum test."""
        leaf1, leaf2 = _leaf_pair([0.3, 0.7], [0.55, 0.45])
        gcw_coupling_circuit(leaf1, leaf2).sample(self.N_SAMPLES, seed=0)

        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        self._assert_sum_sum_marginals(coupling, circ1, circ2, seed=1)

    def test_heap_pollution_before_sum_sum(self):
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=24, seed=11)
        gc.collect()
        circ1, circ2 = _canonical_sum_sum_pair()
        coupling = gcw_coupling_circuit(circ1, circ2)
        self._assert_sum_sum_marginals(coupling, circ1, circ2, seed=1)

    def test_many_sequential_sum_sum_builds(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        p_marg = sum_mixture_marginal([[0.8, 0.2], [0.3, 0.7]], [0.5, 0.5])
        for r in range(30):
            # Fresh leaf allocations between builds perturb object addresses.
            pollute_heap_with_couplings(gcw_coupling_circuit, rounds=2, seed=100 + r)
            coupling = gcw_coupling_circuit(circ1, circ2)
            draws = coupling.sample(8000, seed=r)
            got = empirical_marginal(draws, 0, 2)
            assert_allclose(got, p_marg, atol=self.ATOL)

    def test_fresh_nodes_same_distribution_same_marginals(self):
        """Rebuild inputs at new addresses; marginals must not change."""
        target_p = sum_mixture_marginal([[0.8, 0.2], [0.3, 0.7]], [0.5, 0.5])
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=16, seed=5)
        for base in (0, 10_000, 100_000):
            c1a = make_categorical(base + 0, 0, [0.8, 0.2])
            c1b = make_categorical(base + 1, 0, [0.3, 0.7])
            circ1 = SumNode(id=base + 2, children=[c1a, c1b], parameters=[0.5, 0.5])
            c2a = make_categorical(base + 3, 0, [0.6, 0.4])
            c2b = make_categorical(base + 4, 0, [0.1, 0.9])
            circ2 = SumNode(id=base + 5, children=[c2a, c2b], parameters=[0.4, 0.6])
            coupling = gcw_coupling_circuit(circ1, circ2)
            draws = coupling.sample(12000, seed=1)
            assert_allclose(
                empirical_marginal(draws, 0, 2), target_p, atol=self.ATOL
            )

    def test_exact_marginals_stable_after_heap_pollution(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        p_marg = sum_mixture_marginal([[0.8, 0.2], [0.3, 0.7]], [0.5, 0.5])
        q_marg = sum_mixture_marginal([[0.6, 0.4], [0.1, 0.9]], [0.4, 0.6])
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=20, seed=99)
        coupling = gcw_coupling_circuit(circ1, circ2)
        off = q_var_offset(circ1)
        assert_allclose(exact_marginal(coupling, 0), p_marg, atol=1e-10)
        assert_allclose(exact_marginal(coupling, off + 0), q_marg, atol=1e-10)


class TestCouplingCrossBuildConsistency:
    """Independent builds must agree on exact joint probabilities."""

    def test_two_builds_same_joint_likelihood_grid(self):
        circ1, circ2 = _canonical_sum_sum_pair()
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=10, seed=3)
        c_a = gcw_coupling_circuit(circ1, circ2)
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=10, seed=4)
        c_b = gcw_coupling_circuit(circ1, circ2)
        scope = sorted(c_a.scope_as_list())
        cards = var_cardinalities(c_a)
        import itertools

        for values in itertools.product(*[range(cards[v]) for v in scope]):
            row = assignment_array({scope[i]: values[i] for i in range(len(scope))})
            assert_allclose(
                c_a.likelihood(row),
                c_b.likelihood(row),
                rtol=0,
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# Topology coverage beyond the extras smoke tests
# ---------------------------------------------------------------------------


class TestCouplingTopologies:
    def test_sum_over_product_coupling_marginals(self):
        """Sum x product and product x sum are both supported."""
        prod = make_product(10, [(0, [0.6, 0.4]), (1, [0.25, 0.75])])
        circ_sum = make_sum(
            20,
            0,
            [[0.9, 0.1], [0.2, 0.8]],
            [0.35, 0.65],
        )
        for c1, c2 in ((circ_sum, prod), (prod, circ_sum)):
            coupling = gcw_coupling_circuit(c1, c2)
            walk_pc_invariants(coupling)
            c1_circ = c1
            c2_circ = c2
            q_off = q_var_offset(c1)
            for var in sorted(c1.scope_as_list()):
                assert_allclose(
                    exact_marginal(coupling, var),
                    exact_marginal(c1_circ, var),
                    atol=1e-9,
                )
            for var in sorted(c2.scope_as_list()):
                assert_allclose(
                    exact_marginal(coupling, q_off + var),
                    exact_marginal(c2_circ, var),
                    atol=1e-9,
                )

    @pytest.mark.parametrize("seed", range(8))
    def test_random_sum_sum_exact_marginals(self, seed):
        rng = np.random.default_rng(seed)
        n = 3
        probs1 = [list(rng.dirichlet([1.0] * n)) for _ in range(2)]
        probs2 = [list(rng.dirichlet([1.0] * n)) for _ in range(2)]
        w1 = list(rng.dirichlet([1.0, 1.0]))
        w2 = list(rng.dirichlet([1.0, 1.0]))
        circ1 = make_sum(2, 0, probs1, w1, id_base=seed * 100)
        circ2 = make_sum(5, 0, probs2, w2, id_base=seed * 100 + 50)
        pollute_heap_with_couplings(gcw_coupling_circuit, rounds=4, seed=seed)
        coupling = gcw_coupling_circuit(circ1, circ2)
        assert_allclose(
            exact_marginal(coupling, 0),
            sum_mixture_marginal(probs1, w1),
            atol=1e-9,
        )
        assert_allclose(
            exact_marginal(coupling, q_var_offset(circ1) + 0),
            sum_mixture_marginal(probs2, w2),
            atol=1e-9,
        )
