"""Tests for sparc.optim: simplex projection, apply_grads, iter_nodes, MLETrainer."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    BernoulliInputNode,
    CategoricalInputNode,
        ProductNode,
    SumNode,
    mean_log_likelihood_and_grad,
)
from sparc.optim import (
    MLETrainer,
    apply_grads,
    global_grad_norm,
    iter_nodes,
    simplex_step,
)
from tests.sparc_helpers import assert_on_simplex, assignment_array, project_to_simplex_tangent

pytestmark = pytest.mark.optim


class TestSimplexStep:
    @pytest.mark.parametrize("method", ["tangent", "euclidean"])
    @pytest.mark.parametrize("ascent", [True, False])
    def test_stays_on_simplex(self, method, ascent):
        out = simplex_step(
            [0.2, 0.3, 0.5], [1.0, -0.5, 0.2], 0.1, ascent=ascent, method=method
        )
        assert_on_simplex(out)

    def test_tangent_step_is_deterministic(self):
        a = simplex_step([0.5, 0.5], [1.0, -1.0], 0.05, ascent=True)
        b = simplex_step([0.5, 0.5], [1.0, -1.0], 0.05, ascent=True)
        assert a == b

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            simplex_step([0.5, 0.5], [0.0, 0.0], 0.1, method="bad")


class TestIterNodes:
    def test_visits_each_node_once(self):
        l0 = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        l1 = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
        prod = ProductNode(children=[l0, l1])
        root = SumNode(children=[prod], parameters=[1.0])
        ids = [node.id for node in iter_nodes(root)]
        assert sorted(ids) == [0, 1, 2, 3]

    def test_dag_shared_subtree_once_per_object(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        root = SumNode(children=[leaf, leaf], parameters=[0.5, 0.5])
        seen = list(iter_nodes(root))
        assert len(seen) == 2  # sum + one leaf object, shared pointer


class TestApplyGrads:
    def test_only_touches_nodes_in_grad_dict(self):
        leaf_a = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        leaf_b = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.8])
        root = SumNode(children=[leaf_a, leaf_b], parameters=[0.6, 0.4])
        circuit = root
        before_b = leaf_b.probabilities_list()
        data = np.array([[0], [1], [0]], dtype=np.int32)
        _, grads = mean_log_likelihood_and_grad(root, data)
        # Drop cat grad for leaf_b
        grads.cat_grads.pop(1, None)
        apply_grads(circuit, grads, lr=0.1, ascent=True)
        assert leaf_b.probabilities_list() == pytest.approx(before_b)
        assert leaf_a.probabilities_list() != pytest.approx([0.5, 0.5])

    def test_matches_manual_simplex_step(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.4, 0.6])
        root = SumNode(children=[leaf], parameters=[1.0])
        circuit = root
        grad_vec = [0.3, -0.3]
        manual = simplex_step(
            leaf.probabilities_list(), grad_vec, 0.05, ascent=True
        )
        apply_grads(circuit, ({}, {0: grad_vec}), lr=0.05, ascent=True)
        assert leaf.probabilities_list() == pytest.approx(manual)

    @pytest.mark.parametrize("method", ["tangent", "euclidean"])
    def test_bernoulli_leaf_updated(self, method):
        leaf = BernoulliInputNode(scope_var=0, p=0.3)
        root = SumNode(children=[leaf], parameters=[1.0])
        circuit = root
        apply_grads(
            circuit, ({}, {0: [0.5, -0.5]}), lr=0.02, ascent=True, method=method
        )
        assert_on_simplex(leaf.probabilities_list())


class TestGlobalGradNorm:
    def test_zero_for_empty_bundle(self):
        assert global_grad_norm(({}, {})) == 0.0

    def test_matches_manual_l2(self):
        bundle = ({1: [0.3, -0.3]}, {0: [0.1, -0.1]})
        manual = np.sqrt(0.3**2 + (-0.3)**2 + 0.1**2 + (-0.1)**2)
        assert global_grad_norm(bundle) == pytest.approx(manual)


class TestMLETrainer:
    def test_increases_mean_ll_on_synthetic_data(self):
        truth_leaf = CategoricalInputNode(
            scope_var=0, probabilities=[0.7, 0.3]
        )
        truth = truth_leaf
        data = truth.sample(200, seed=0)
        model_leaf = CategoricalInputNode(
            scope_var=0, probabilities=[0.5, 0.5]
        )
        model = model_leaf
        trainer = MLETrainer(model, lr=0.3)
        hist = trainer.fit(data, epochs=25)
        assert hist[-1] > hist[0]

    def test_euclidean_method_runs(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.5, 0.5])
        circuit = leaf
        data = np.array([[0], [1], [0]], dtype=np.int32)
        trainer = MLETrainer(circuit, lr=0.1, method="euclidean")
        ll = trainer.step(data)
        assert np.isfinite(ll)
        assert_on_simplex(leaf.probabilities_list())
