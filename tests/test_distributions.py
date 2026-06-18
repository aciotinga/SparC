"""Tests for the finite-discrete leaf distributions and their integration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sparc import (
    BernoulliInputNode,
    Circuit,
    CircuitSerializer,
    DiscreteLogisticInputNode,
    IndicatorInputNode,
    LiteralInputNode,
    ProductNode,
    SumNode,
    cw_distance,
    likelihood,
)


def _leaf_pmf(node, card):
    node.propagate_scope()
    return [likelihood(node, {0: k}) for k in range(card)]


def test_bernoulli_pmf_and_sampling():
    node = BernoulliInputNode(0, 0, 0.3)
    assert _leaf_pmf(node, 2) == pytest.approx([0.7, 0.3])
    assert node.p() == pytest.approx(0.3)
    samples = Circuit(node).sample(2000, seed=0)
    mean = np.mean([s[0] for s in samples])
    assert mean == pytest.approx(0.3, abs=0.05)


def test_indicator_is_one_hot():
    node = IndicatorInputNode(0, 0, 2, 5)
    assert _leaf_pmf(node, 5) == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0])
    assert node.value_at() == 2
    assert node.num_categories() == 5


def test_literal_pmf():
    node = LiteralInputNode(0, 0, True)
    assert _leaf_pmf(node, 2) == pytest.approx([0.0, 1.0])
    assert node.value_at() == 1


def test_discrete_logistic_normalized():
    node = DiscreteLogisticInputNode(0, 0, 3.0, 1.5, 8)
    pmf = _leaf_pmf(node, 8)
    assert sum(pmf) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in pmf)
    assert node.num_categories() == 8


@pytest.mark.parametrize(
    "node",
    [
        BernoulliInputNode(0, 0, 0.4),
        IndicatorInputNode(0, 0, 1, 4),
        LiteralInputNode(0, 0, 0),
        DiscreteLogisticInputNode(0, 0, 1.0, 0.8, 6),
    ],
)
def test_leaf_clone_roundtrip(node):
    node.propagate_scope()
    clone = Circuit(node).clone().root
    assert type(clone) is type(node)
    card = node.cardinality()
    a = [likelihood(node, {0: k}) for k in range(card)]
    b = [likelihood(clone, {0: k}) for k in range(card)]
    assert a == pytest.approx(b)
    assert clone is not node


@pytest.mark.parametrize(
    "leaf_a, leaf_b",
    [
        (BernoulliInputNode(0, 0, 0.4), BernoulliInputNode(1, 1, 0.6)),
        (IndicatorInputNode(0, 0, 1, 4), IndicatorInputNode(1, 1, 2, 4)),
        (LiteralInputNode(0, 0, 0), LiteralInputNode(1, 1, 1)),
        (
            DiscreteLogisticInputNode(0, 0, 1.0, 0.8, 6),
            DiscreteLogisticInputNode(1, 1, 2.0, 1.0, 6),
        ),
    ],
)
def test_leaf_serializer_roundtrip(leaf_a, leaf_b):
    root = SumNode(2, [ProductNode(3, [leaf_a, leaf_b])], [1.0])
    root.propagate_scope()
    restored = CircuitSerializer.loads(CircuitSerializer.dumps(root))
    prod = restored.children()[0]
    for original, copy in zip([leaf_a, leaf_b], prod.children()):
        assert type(copy) is type(original)
        card = original.cardinality()
        a = [likelihood(original, {original_var: k})
             for original_var, k in zip([original.scope_as_list()[0]] * card, range(card))]
        b = [likelihood(copy, {copy.scope_as_list()[0]: k}) for k in range(card)]
        assert a == pytest.approx(b)


def test_new_leaves_participate_in_cw():
    # Two structurally identical circuits over bernoulli leaves.
    def build(p0, p1):
        a = BernoulliInputNode(0, 0, p0)
        b = BernoulliInputNode(1, 1, p1)
        root = ProductNode(2, [a, b])
        root.propagate_scope()
        return Circuit(root)

    c1 = build(0.2, 0.8)
    c2 = build(0.5, 0.5)
    d = cw_distance(c1, c2)
    assert math.isfinite(d)
    assert d >= 0.0
    assert cw_distance(c1, c1) == pytest.approx(0.0, abs=1e-9)
