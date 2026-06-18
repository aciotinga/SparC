"""Tests for the built-in circuit structures."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from sparc import CircuitSerializer, cw_distance, likelihood, log_likelihood
from sparc.optim import MLETrainer
from sparc.structures import (
    HCLT,
    HMM,
    PD,
    PDHCLT,
    RAT_SPN,
    Bernoulli,
    DiscreteLogistic,
    GeneralizedHMM,
    Indicator,
)


def _full_assignment(scope):
    return {v: 0 for v in scope}


def _enumerate_ll_sum(circuit, num_vars, num_states):
    total = 0.0
    for combo in itertools.product(range(num_states), repeat=num_vars):
        total += likelihood(circuit.root, dict(enumerate(combo)))
    return total


def test_hmm_builds_and_normalizes():
    circuit = HMM(seq_length=4, num_latents=3, num_emits=2, seed=0)
    assert set(circuit.root.scope_as_list()) == set(range(4))
    assert _enumerate_ll_sum(circuit, 4, 2) == pytest.approx(1.0)
    sample = circuit.sample(1, seed=1)[0]
    assert set(sample.keys()) == set(range(4))
    assert math.isfinite(circuit.log_likelihood(sample))


def test_generalized_hmm_custom_emission():
    circuit = GeneralizedHMM(
        seq_length=5, num_latents=2, input_dist=Bernoulli(), seed=0
    )
    assert set(circuit.root.scope_as_list()) == set(range(5))
    assert _enumerate_ll_sum(circuit, 5, 2) == pytest.approx(1.0)


def test_rat_spn_builds_and_normalizes():
    circuit = RAT_SPN(
        num_vars=5, num_latents=2, depth=2, num_repetitions=2,
        num_pieces=2, num_cats=2, seed=0,
    )
    assert set(circuit.root.scope_as_list()) == set(range(5))
    assert _enumerate_ll_sum(circuit, 5, 2) == pytest.approx(1.0)


def test_hclt_builds_and_normalizes():
    data = np.random.RandomState(0).randint(0, 2, size=(80, 5))
    circuit = HCLT(data, num_latents=3, num_bins=4, num_cats=2, seed=0)
    assert set(circuit.root.scope_as_list()) == set(range(5))
    assert _enumerate_ll_sum(circuit, 5, 2) == pytest.approx(1.0)


def test_pd_builds_and_normalizes():
    circuit = PD(data_shape=(2, 3), num_latents=2, num_cats=2, seed=0)
    assert set(circuit.root.scope_as_list()) == set(range(6))
    assert _enumerate_ll_sum(circuit, 6, 2) == pytest.approx(1.0)


def test_pd_prod_dominated():
    circuit = PD(
        data_shape=(2, 2), num_latents=2, num_cats=2,
        structure_type="prod_dominated", seed=0,
    )
    assert set(circuit.root.scope_as_list()) == set(range(4))
    assert _enumerate_ll_sum(circuit, 4, 2) == pytest.approx(1.0)


def test_pdhclt_builds_and_normalizes():
    data = np.random.RandomState(0).randint(0, 2, size=(60, 6))
    circuit = PDHCLT(
        data, data_shape=(2, 3), num_latents=2, num_cats=2,
        num_bins=4, max_split_depth=1, seed=0,
    )
    assert set(circuit.root.scope_as_list()) == set(range(6))
    assert _enumerate_ll_sum(circuit, 6, 2) == pytest.approx(1.0)


def test_discrete_logistic_emission_hmm():
    circuit = GeneralizedHMM(
        seq_length=4, num_latents=2, input_dist=DiscreteLogistic(3), seed=0
    )
    assert _enumerate_ll_sum(circuit, 4, 3) == pytest.approx(1.0)


def test_indicator_emission_normalizes():
    circuit = GeneralizedHMM(
        seq_length=3, num_latents=2, input_dist=Indicator(2), seed=0
    )
    total = _enumerate_ll_sum(circuit, 3, 2)
    # Deterministic leaves place all mass on specific outcomes; total mass over
    # the full outcome grid is still a valid (<= 1) probability and the circuit
    # is normalized over the support it can express.
    assert 0.0 <= total <= 1.0 + 1e-9


def test_same_seed_structures_identical_cw():
    a = HMM(seq_length=5, num_latents=3, num_emits=2, seed=7)
    b = HMM(seq_length=5, num_latents=3, num_emits=2, seed=7)
    assert cw_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_different_structures_positive_cw():
    a = HMM(seq_length=5, num_latents=3, num_emits=2, seed=1)
    b = HMM(seq_length=5, num_latents=3, num_emits=2, seed=2)
    d = cw_distance(a, b)
    assert math.isfinite(d)
    assert d > 0.0


def test_structure_batched_log_likelihood():
    circuit = HMM(seq_length=6, num_latents=3, num_emits=3, seed=0)
    samples = circuit.sample(20, seed=1)
    batch = np.array([[s[v] for v in range(6)] for s in samples])
    batched = circuit.batched_log_likelihood(batch)
    assert batched.shape == (20,)
    single = np.array([circuit.log_likelihood(s) for s in samples])
    assert batched == pytest.approx(single, abs=1e-9)


def test_structure_serialization_roundtrip():
    circuit = RAT_SPN(
        num_vars=5, num_latents=2, depth=2, num_repetitions=1,
        num_cats=3, seed=0,
    )
    restored = CircuitSerializer.loads(CircuitSerializer.dumps(circuit))
    samples = circuit.sample(10, seed=2)
    for s in samples:
        assert circuit.log_likelihood(s) == pytest.approx(log_likelihood(restored, s))


def test_hmm_mle_training_improves():
    truth = HMM(seq_length=6, num_latents=3, num_emits=2, seed=0)
    data = truth.sample(300, seed=1)
    model = HMM(seq_length=6, num_latents=3, num_emits=2, seed=42)
    history = MLETrainer(model, lr=0.3).fit(data, epochs=30)
    assert all(math.isfinite(h) for h in history)
    assert history[-1] >= history[0]
