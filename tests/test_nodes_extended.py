"""Extended node-level tests beyond test_nodes.py."""

from __future__ import annotations

import pytest

from sparc import (
    BernoulliInputNode,
    CategoricalInputNode,
    IndicatorInputNode,
    LiteralInputNode,
    ProductNode,
    SumNode,
)

pytestmark = pytest.mark.nodes


class TestBernoulliNode:
    def test_p_accessor(self):
        node = BernoulliInputNode(id=0, scope_var=0, p=0.25)
        assert node.p() == pytest.approx(0.25)

    def test_probabilities_list_format(self):
        node = BernoulliInputNode(id=0, scope_var=0, p=0.4)
        assert node.probabilities_list() == pytest.approx([0.6, 0.4])

    def test_set_probabilities(self):
        node = BernoulliInputNode(id=0, scope_var=0, p=0.4)
        node.set_probabilities_list([0.7, 0.3])
        assert node.p() == pytest.approx(0.3)


class TestIndicatorLiteral:
    def test_indicator_cardinality(self):
        node = IndicatorInputNode(id=0, scope_var=2, value=3, num_cats=7)
        assert node.num_categories() == 7
        assert node.value_at() == 3
        assert node.scope_as_list() == [2]

    def test_literal_boolean_encoding(self):
        node = LiteralInputNode(id=0, scope_var=0, value=False)
        assert node.value_at() == 0
        node2 = LiteralInputNode(id=1, scope_var=0, value=True)
        assert node2.value_at() == 1


class TestScopePropagation:
    def test_dag_shared_child_scope_once(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        shared = ProductNode(
            id=1,
            children=[
                leaf,
                CategoricalInputNode(id=2, scope_var=1, probabilities=[0.5, 0.5]),
            ],
        )
        root = SumNode(id=3, children=[shared, shared], parameters=[0.5, 0.5])
        root.propagate_scope()
        assert set(root.scope_as_list()) == {0, 1}

    def test_sum_mismatched_child_scopes_allowed_at_build(self):
        """Scope union still works when children share variables differently."""
        l0 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        l1 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.5, 0.5])
        root = SumNode(id=2, children=[l0, l1], parameters=[0.5, 0.5])
        root.propagate_scope()
        assert root.scope_as_list() == [0]


class TestParameterValidation:
    def test_sum_negative_weight_rejected(self):
        leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
        with pytest.raises(ValueError):
            SumNode(id=1, children=[leaf], parameters=[-0.1])

    def test_categorical_too_few_outcomes(self):
        with pytest.raises(ValueError, match="at least 2"):
            CategoricalInputNode(id=0, scope_var=0, probabilities=[1.0])

    def test_negative_scope_var(self):
        with pytest.raises(ValueError, match="scope_var"):
            CategoricalInputNode(id=0, scope_var=-1, probabilities=[0.5, 0.5])
