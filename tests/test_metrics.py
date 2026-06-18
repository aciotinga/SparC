"""Tests for ground metrics and their effect on CW / GCW queries."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    PNormMetric,
    cw_distance,
    gcw_crossterm,
)
from tests.sparc_helpers import brute_force_cw_leaf

pytestmark = pytest.mark.metrics


class TestPNormMetric:
    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            PNormMetric(scale=0.0)

    @pytest.mark.parametrize("metric_p", [1.0, 2.0])
    def test_cw_leaf_matches_brute_force(self, metric_p):
        p = [0.3, 0.7]
        q = [0.55, 0.45]
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=q)
        expected = brute_force_cw_leaf(p, q, metric_p=metric_p, scale_factor=1.0)
        got = cw_distance(leaf1, leaf2, metric_p=metric_p)
        assert_allclose(got, expected, atol=1e-10)

    def test_scale_factor_divides_cost(self):
        p = [0.4, 0.6]
        q = [0.1, 0.9]
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=p)
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=q)
        d1 = cw_distance(leaf1, leaf2, scale_factor=1.0)
        d2 = cw_distance(leaf1, leaf2, scale_factor=2.0)
        assert_allclose(d2, d1 / 2.0, rtol=0, atol=1e-12)

    def test_identical_leaves_zero_distance(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.2, 0.8])
        assert cw_distance(leaf, leaf) == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self):
        p = [0.25, 0.75]
        q = [0.6, 0.4]
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=p)
        l2 = CategoricalInputNode(id=1, scope_var=0, probabilities=q)
        assert_allclose(cw_distance(l1, l2), cw_distance(l2, l1), atol=1e-12)


class TestDualMetricsGCW:
    """GCW accepts independent metrics per circuit."""

    def test_asymmetric_scale_factors(self):
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=p)
        l2 = CategoricalInputNode(id=1, scope_var=0, probabilities=q)
        c_default = gcw_crossterm(l1, l2)
        c_scaled = gcw_crossterm(
            l1, l2, scale_factor_1=2.0, scale_factor_2=4.0
        )
        assert np.isfinite(c_default)
        assert np.isfinite(c_scaled)

    def test_custom_metric_objects(self):
        m1 = PNormMetric(p=2.0, scale=1.0)
        m2 = PNormMetric(p=1.0, scale=3.0)
        l1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.3, 0.7])
        l2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.6, 0.4])
        val = gcw_crossterm(l1, l2, metric1=m1, metric2=m2)
        assert np.isfinite(val)
        assert val >= -1e-8
