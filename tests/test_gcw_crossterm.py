"""Tests for the GCW cross-term query (``sparc.gcw_crossterm``).

Runs against SparC's built-in solvers (no Gurobi / SciPy). The leaf brute-force
check validates the built-in NW coupling.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import CategoricalInputNode, ProductNode, SumNode, gcw_crossterm


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


def _brute_force_crossterm(p, q, d_p, d_q):
    T = _nw_coupling_dense(p, q)
    n, m = T.shape
    expected = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    expected += T[i, j] * T[k, l] * d_p[i, k] * d_q[j, l]
    return expected


class TestGCWCrosstermSmoke:
    def test_identical_leaves(self):
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.5, 0.5])
        cross = gcw_crossterm(leaf1, leaf2)
        assert np.isfinite(cross)
        assert cross >= -1e-8

    def test_sum_to_sum(self):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.8, 0.2])
        c1b = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.3, 0.7])
        circ1 = SumNode(id=2, children=[c1a, c1b], parameters=[0.5, 0.5])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.6, 0.4])
        c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=[0.1, 0.9])
        circ2 = SumNode(id=5, children=[c2a, c2b], parameters=[0.4, 0.6])
        cross = gcw_crossterm(circ1, circ2)
        assert np.isfinite(cross)
        assert cross >= -1e-8

    def test_product_coupling(self):
        p1 = ProductNode(
            id=6,
            children=[
                CategoricalInputNode(id=7, scope_var=0, probabilities=[0.5, 0.5]),
                CategoricalInputNode(id=8, scope_var=1, probabilities=[0.5, 0.5]),
            ],
        )
        p2 = ProductNode(
            id=9,
            children=[
                CategoricalInputNode(id=10, scope_var=0, probabilities=[0.25, 0.75]),
                CategoricalInputNode(id=11, scope_var=1, probabilities=[0.75, 0.25]),
            ],
        )
        cross = gcw_crossterm(p1, p2)
        assert np.isfinite(cross)
        assert cross >= -1e-8


class TestLeafCrossterm:
    def test_crossterm_brute_force(self):
        """Leaf cross-term matches explicit quadruple sum over NW plan."""
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        d_p = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        d_q = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        expected = _brute_force_crossterm(p, q, d_p, d_q)

        leaf1 = CategoricalInputNode(id=20, scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(id=21, scope_var=0, probabilities=q)
        cross = gcw_crossterm(leaf1, leaf2)
        assert_allclose(cross, expected, atol=1e-10)
