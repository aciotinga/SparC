"""Tests for CompiledCircuit.refresh_parameters after parameter updates."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import CategoricalInputNode, Circuit, GradBundle, ProductNode
from sparc.optim import apply_grads

pytestmark = pytest.mark.eval


def test_refresh_parameters_updates_batched_ll():
    leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.7, 0.3])
    root = ProductNode(id=1, children=[leaf])
    circuit = Circuit(root)
    compiled = circuit.compile()
    data = np.array([[0], [1]], dtype=np.int32)
    ll_before = compiled.log_likelihood(data).copy()

    grads = GradBundle()
    grads.cat_grads = {0: np.array([0.1, -0.1])}
    apply_grads(circuit.root, grads, lr=0.5)
    compiled.refresh_parameters()
    ll_after = compiled.log_likelihood(data)
    assert not np.allclose(ll_before, ll_after)
    assert_allclose(
        ll_after,
        circuit.compile().log_likelihood(data),
        rtol=0,
        atol=1e-10,
    )
