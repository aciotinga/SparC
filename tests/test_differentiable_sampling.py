"""Tests for hard-forward SIMPLE sampling and its explicit VJP."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sparc import (
    BernoulliInputNode,
    CategoricalInputNode,
    DifferentiableSample,
    DiscreteLogisticInputNode,
    IndicatorInputNode,
    ProductNode,
    SumNode,
    sample,
)


pytestmark = [pytest.mark.eval, pytest.mark.no_dual_path]


def _packed_product():
    left = CategoricalInputNode(3, [0.7, 0.3])
    right = CategoricalInputNode(17, [0.2, 0.3, 0.5])
    return ProductNode([left, right]), left, right


def _assert_same_grads(left, right):
    assert left.has_value == right.has_value
    assert left.sum_grads.keys() == right.sum_grads.keys()
    assert left.cat_grads.keys() == right.cat_grads.keys()
    for node_id in left.sum_grads:
        assert_allclose(left.sum_grads[node_id], right.sum_grads[node_id])
    for node_id in left.cat_grads:
        assert_allclose(left.cat_grads[node_id], right.cat_grads[node_id])


def test_packed_result_matches_assignments_and_is_immutable():
    root, _, _ = _packed_product()
    result = sample(root, 8, seed=11, differentiable=True)

    assert isinstance(result, DifferentiableSample)
    assert result.assignments.shape == (8, 18)
    assert result.assignments.dtype == np.int32
    assert result.one_hot.shape == (8, 5)
    assert result.one_hot.dtype == np.float64
    assert_array_equal(result.variables, [3, 17])
    assert_array_equal(result.cardinalities, [2, 3])
    assert_array_equal(result.offsets, [0, 2, 5])

    for row in range(8):
        assert result.one_hot[row, result.assignments[row, 3]] == 1.0
        assert (
            result.one_hot[row, 2 + result.assignments[row, 17]] == 1.0
        )
        assert result.one_hot[row, :2].sum() == 1.0
        assert result.one_hot[row, 2:].sum() == 1.0

    assert not result.assignments.flags.writeable
    assert not result.one_hot.flags.writeable
    with pytest.raises(ValueError):
        result.one_hot[0, 0] = 0.0


def test_result_objects_cannot_be_forged_with_unchecked_tapes():
    with pytest.raises(TypeError, match="created by sample"):
        DifferentiableSample()


def test_result_tapes_cannot_be_restored_from_pickle_state():
    result = CategoricalInputNode(0, [0.5, 0.5]).sample(
        1, seed=0, differentiable=True
    )
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(result)


def test_vjp_uses_frozen_dimensions_if_public_array_is_reshaped():
    result = CategoricalInputNode(0, [0.5, 0.5]).sample(
        2, seed=0, differentiable=True
    )
    result.one_hot.shape = (1, 4)

    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        result.vjp(np.ones((1, 4)))
    grads = result.vjp(np.ones((2, 2)))
    assert_allclose(next(iter(grads.cat_grads.values())), [2.0, 2.0])


def test_differentiable_sampling_is_seed_deterministic():
    root, _, _ = _packed_product()
    first = sample(root, 20, seed=42, differentiable=True)
    second = sample(root, 20, seed=42, differentiable=True)
    assert_array_equal(first.assignments, second.assignments)
    assert_array_equal(first.one_hot, second.one_hot)
    upstream = np.arange(first.one_hot.size, dtype=np.float64).reshape(
        first.one_hot.shape
    )
    _assert_same_grads(first.vjp(upstream), second.vjp(upstream))


def test_categorical_leaf_vjp_sums_batch_upstream():
    leaf = CategoricalInputNode(0, [0.2, 0.3, 0.5])
    result = leaf.sample(2, seed=3, differentiable=True)
    upstream = np.array([[1.0, -2.0, 4.0], [0.5, 3.0, -1.0]])

    grads = result.vjp(upstream)

    assert not grads.has_value
    assert math.isnan(grads.value)
    assert grads.sum_grads == {}
    assert_allclose(
        grads.cat_grads[int(leaf.id)],
        upstream.sum(axis=0),
    )


def test_bernoulli_leaf_vjp_uses_two_simplex_coordinates():
    leaf = BernoulliInputNode(0, 0.7)
    result = leaf.sample(3, seed=5, differentiable=True)
    upstream = np.array([[2.0, 5.0], [-1.0, 4.0], [0.0, 3.0]])
    grads = result.vjp(upstream)
    assert_allclose(grads.cat_grads[int(leaf.id)], [1.0, 12.0])


def test_product_routes_packed_slices_to_each_leaf():
    root, left, right = _packed_product()
    result = root.sample(2, seed=8, differentiable=True)
    upstream = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, -1.0, 2.0, 0.0, 7.0],
        ]
    )
    grads = result.vjp(upstream)
    assert_allclose(grads.cat_grads[int(left.id)], [1.5, 1.0])
    assert_allclose(grads.cat_grads[int(right.id)], [5.0, 4.0, 12.0])


def test_sum_vjp_uses_every_counterfactual_child_assignment():
    zero = IndicatorInputNode(0, 0, 2)
    one = IndicatorInputNode(0, 1, 2)
    root = SumNode([zero, one], [0.25, 0.75])
    result = root.sample(2, seed=4, differentiable=True)
    upstream = np.array([[2.0, 7.0], [-1.0, 3.0]])

    grads = result.vjp(upstream)

    assert_allclose(grads.sum_grads[int(root.id)], [1.0, 10.0])
    assert grads.cat_grads == {}


def test_sum_vjp_is_restricted_to_its_scope():
    local_sum = SumNode(
        [IndicatorInputNode(0, 0, 2), IndicatorInputNode(0, 1, 2)],
        [0.5, 0.5],
    )
    sibling = CategoricalInputNode(2, [0.5, 0.5])
    root = ProductNode([local_sum, sibling])
    result = root.sample(1, seed=0, differentiable=True)
    upstream = np.array([[2.0, 7.0, 100.0, 200.0]])

    grads = result.vjp(upstream)

    assert_allclose(grads.sum_grads[int(local_sum.id)], [2.0, 7.0])


def test_unselected_nested_nodes_are_stop_gradient():
    trainable = CategoricalInputNode(0, [0.4, 0.6])
    inner = SumNode(
        [trainable, IndicatorInputNode(0, 0, 2)],
        [0.5, 0.5],
    )
    selected = IndicatorInputNode(0, 1, 2)
    root = SumNode([inner, selected], [0.0, 1.0])
    result = root.sample(5, seed=9, differentiable=True)

    grads = result.vjp(np.ones_like(result.one_hot))

    assert int(root.id) in grads.sum_grads
    assert int(inner.id) not in grads.sum_grads
    assert int(trainable.id) not in grads.cat_grads


def test_shared_dag_occurrences_reduce_by_node_id():
    shared = CategoricalInputNode(0, [0.6, 0.4])
    root = SumNode([shared, shared], [0.5, 0.5])
    result = root.sample(4, seed=12, differentiable=True)
    upstream = np.array(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0], [2.0, 0.0]]
    )

    grads = result.vjp(upstream)

    assert_allclose(
        grads.cat_grads[int(shared.id)],
        upstream.sum(axis=0),
    )
    assert int(root.id) in grads.sum_grads


def test_vjp_is_repeatable_linear_and_rejects_wrong_shape():
    leaf = CategoricalInputNode(0, [0.3, 0.7])
    result = leaf.sample(3, seed=1, differentiable=True)
    left = np.array([[1.0, 2.0], [0.0, -1.0], [4.0, 3.0]])
    right = np.array([[2.0, 0.5], [1.0, 1.0], [-1.0, 2.0]])

    first = result.vjp(left)
    again = result.vjp(left)
    combined = result.vjp(left + right)
    right_grad = result.vjp(right)

    _assert_same_grads(first, again)
    assert_allclose(
        combined.cat_grads[int(leaf.id)],
        first.cat_grads[int(leaf.id)]
        + right_grad.cat_grads[int(leaf.id)],
    )
    with pytest.raises(ValueError, match="upstream must have shape"):
        result.vjp(np.ones((3, 3)))
    with pytest.raises(ValueError, match="upstream must have shape"):
        result.vjp(np.ones(2))


def test_zero_samples_have_metadata_and_empty_vjp():
    leaf = CategoricalInputNode(3, [0.25, 0.75])
    result = leaf.sample(0, seed=0, differentiable=True)
    assert result.assignments.shape == (0, 4)
    assert result.one_hot.shape == (0, 2)
    grads = result.vjp(np.empty((0, 2)))
    assert grads.sum_grads == {}
    assert grads.cat_grads == {}


def test_deterministic_and_discrete_logistic_leaves_have_no_leaf_grads():
    root = ProductNode(
        [
            IndicatorInputNode(0, 1, 2),
            DiscreteLogisticInputNode(1, mu=1.0, s=0.5, num_cats=3),
        ]
    )
    result = root.sample(4, seed=2, differentiable=True)
    grads = result.vjp(np.ones_like(result.one_hot))
    assert grads.cat_grads == {}


def test_nonsmooth_sum_is_rejected_only_in_differentiable_mode():
    product = ProductNode(
        [
            CategoricalInputNode(0, [0.5, 0.5]),
            CategoricalInputNode(1, [0.5, 0.5]),
        ]
    )
    root = SumNode(
        [product, CategoricalInputNode(0, [0.5, 0.5])],
        [0.5, 0.5],
    )
    assert isinstance(root.sample(1, seed=0), np.ndarray)
    with pytest.raises(ValueError, match="not smooth"):
        root.sample(1, seed=0, differentiable=True)


def test_overlapping_product_scopes_are_rejected():
    root = ProductNode(
        [
            CategoricalInputNode(0, [0.5, 0.5]),
            CategoricalInputNode(0, [0.4, 0.6]),
        ]
    )
    with pytest.raises(ValueError, match="not decomposable"):
        root.sample(1, seed=0, differentiable=True)


def test_inconsistent_cardinalities_are_rejected():
    root = SumNode(
        [
            CategoricalInputNode(0, [0.5, 0.5]),
            CategoricalInputNode(0, [0.2, 0.3, 0.5]),
        ],
        [0.5, 0.5],
    )
    with pytest.raises(ValueError, match="not smooth"):
        root.sample(1, seed=0, differentiable=True)


def test_duplicate_node_ids_are_rejected():
    left = CategoricalInputNode(0, [0.5, 0.5], id=7)
    right = CategoricalInputNode(0, [0.4, 0.6], id=7)
    root = SumNode([left, right], [0.5, 0.5], id=8)
    with pytest.raises(ValueError, match="distinct nodes share id 7"):
        root.sample(1, seed=0, differentiable=True)


def test_object_and_compiled_results_and_vjps_match():
    left = SumNode(
        [
            CategoricalInputNode(0, [0.8, 0.2]),
            CategoricalInputNode(0, [0.3, 0.7]),
        ],
        [0.6, 0.4],
    )
    right = BernoulliInputNode(2, 0.65)
    root = ProductNode([left, right])
    compiled = root.compile()

    object_result = root.sample(25, seed=33, differentiable=True)
    compiled_result = compiled.sample(25, seed=33, differentiable=True)
    assert_array_equal(object_result.assignments, compiled_result.assignments)
    assert_array_equal(object_result.one_hot, compiled_result.one_hot)

    upstream = np.random.default_rng(4).normal(
        size=object_result.one_hot.shape
    )
    _assert_same_grads(
        object_result.vjp(upstream),
        compiled_result.vjp(upstream),
    )


def test_compiled_differentiable_sampling_uses_parameter_snapshot():
    zero = IndicatorInputNode(0, 0, 2)
    one = IndicatorInputNode(0, 1, 2)
    root = SumNode([zero, one], [1.0, 0.0])
    compiled = root.compile()
    root.set_parameters_list([0.0, 1.0])

    stale = compiled.sample(3, seed=0, differentiable=True)
    live = root.sample(3, seed=0, differentiable=True)
    assert_array_equal(stale.assignments[:, 0], [0, 0, 0])
    assert_array_equal(live.assignments[:, 0], [1, 1, 1])

    old_vjp = stale.vjp(np.array([[2.0, 7.0]] * 3))
    compiled.refresh_parameters()
    refreshed = compiled.sample(3, seed=0, differentiable=True)
    assert_array_equal(refreshed.assignments[:, 0], [1, 1, 1])
    delayed_vjp = stale.vjp(np.array([[2.0, 7.0]] * 3))
    _assert_same_grads(old_vjp, delayed_vjp)


def test_compiled_sampling_ignores_invalid_live_probabilities():
    leaf = CategoricalInputNode(0, [0.7, 0.3])
    compiled = leaf.compile()
    with pytest.raises(ValueError, match="sum to 1"):
        leaf.set_probabilities_list([0.8, 0.8])

    result = compiled.sample(4, seed=0, differentiable=True)

    assert result.assignments.shape == (4, 1)
    with pytest.raises(ValueError, match="sum to 1"):
        leaf.sample(1, seed=0, differentiable=True)


def test_compiled_sampling_rejects_stale_scope_metadata():
    leaf = CategoricalInputNode(0, [0.7, 0.3])
    compiled = leaf.compile()
    leaf.set_scope_from_iterable([1])

    with pytest.raises(ValueError, match="stale scope metadata"):
        compiled.sample(1, seed=0, differentiable=True)


def test_monte_carlo_simple_gradient_matches_linear_objective():
    first = CategoricalInputNode(0, [0.8, 0.2])
    second = CategoricalInputNode(0, [0.3, 0.7])
    root = SumNode([first, second], [0.6, 0.4])
    n_samples = 10_000
    result = root.sample(n_samples, seed=123, differentiable=True)
    reward = np.array([0.0, 1.0])
    upstream = np.broadcast_to(
        reward / n_samples, result.one_hot.shape
    ).copy()

    grads = result.vjp(upstream)

    assert_allclose(
        grads.sum_grads[int(root.id)],
        [0.2, 0.7],
        atol=0.025,
    )
    assert_allclose(
        grads.cat_grads[int(first.id)],
        0.6 * reward,
        atol=0.025,
    )
    assert_allclose(
        grads.cat_grads[int(second.id)],
        0.4 * reward,
        atol=0.025,
    )
