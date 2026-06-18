import pytest

from sparc import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
)


def test_categorical_scope_size_one():
    node = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    node.propagate_scope()
    assert node.scope_as_list() == [3]


def test_categorical_rejects_short_distribution():
    with pytest.raises(ValueError, match="at least 2 outcomes"):
        CategoricalInputNode(id=0, scope_var=0, probabilities=[1.0])


def test_categorical_rejects_bad_probabilities():
    with pytest.raises(ValueError, match="sum to 1"):
        CategoricalInputNode(id=0, scope_var=0, probabilities=[0.2, 0.2])


def test_product_scope_union():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    prod.propagate_scope()
    assert set(prod.scope_as_list()) == {3, 17}


def test_sum_scope_union():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    prod = ProductNode(id=2, children=[x0, x1])
    root = SumNode(id=3, children=[prod], parameters=[1.0])
    root.propagate_scope()
    assert set(root.scope_as_list()) == {3, 17}


def test_dag_shared_child():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
    shared = ProductNode(id=2, children=[x0, x1])
    sum_a = SumNode(id=3, children=[shared], parameters=[1.0])
    sum_b = SumNode(id=4, children=[shared], parameters=[1.0])
    sum_a.propagate_scope()
    sum_b.propagate_scope()
    assert set(sum_a.scope_as_list()) == {3, 17}
    assert set(sum_b.scope_as_list()) == {3, 17}


def test_sum_parameter_length_mismatch():
    leaf = CategoricalInputNode(id=0, scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="length mismatch"):
        SumNode(id=1, children=[leaf], parameters=[0.5, 0.5])


def test_sum_parameters_must_sum_to_one():
    leaf = CategoricalInputNode(id=0, scope_var=1, probabilities=[0.5, 0.5])
    with pytest.raises(ValueError, match="sum to 1"):
        SumNode(id=1, children=[leaf, leaf], parameters=[0.5, 0.4])


def test_scope_as_list_sorted():
    x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
    assert x0.scope_as_list() == [3]
