"""Tests for CompiledCircuit batched evaluation and path parity."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    CategoricalInputNode,
    CompiledCircuit,
    ProductNode,
    SumNode,
    likelihood,
    log_likelihood,
    sample,
)
from tests.sparc_helpers import assignment_array

pytestmark = pytest.mark.eval


def _mixed_circuit():
    l0 = CategoricalInputNode(scope_var=5, probabilities=[0.6, 0.4])
    l1 = CategoricalInputNode(scope_var=9, probabilities=[0.25, 0.75])
    prod = ProductNode(children=[l0, l1])
    l2 = CategoricalInputNode(scope_var=5, probabilities=[0.1, 0.9])
    root = SumNode(children=[prod, l2], parameters=[0.7, 0.3])
    return root


class TestCompiledCircuit:
    def test_batched_matches_per_row(self):
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        n = 64
        rng = np.random.default_rng(0)
        vars_ = sorted(circuit.scope_as_list())
        cards = {v: 2 for v in vars_}
        width = max(vars_) + 1
        data = np.zeros((n, width), dtype=np.int32)
        for v in vars_:
            data[:, v] = rng.integers(0, cards[v], size=n)
        batched = compiled.log_likelihood(data)
        per_row = np.array([circuit.log_likelihood(data[r]) for r in range(n)])
        assert_allclose(batched, per_row, rtol=0, atol=1e-10)

    def test_large_batch_stress(self):
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        n = 10_000
        rng = np.random.default_rng(42)
        vars_ = sorted(circuit.scope_as_list())
        width = max(vars_) + 1
        data = np.zeros((n, width), dtype=np.int32)
        for v in vars_:
            data[:, v] = rng.integers(0, 2, size=n)
        batched = compiled.log_likelihood(data)
        per_row = np.array([circuit.log_likelihood(data[r]) for r in range(8)])
        assert_allclose(batched[:8], per_row, rtol=0, atol=1e-10)
        assert batched.shape == (n,)
        assert np.all(np.isfinite(batched))

    def test_var_to_col_reordering(self):
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        data = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int32)
        var_to_col = {5: 1, 9: 0}
        batched = compiled.log_likelihood(data, var_to_col=var_to_col)
        expected = np.array(
            [
                circuit.log_likelihood(data[0], var_to_col=var_to_col),
                circuit.log_likelihood(data[1], var_to_col=var_to_col),
                circuit.log_likelihood(data[2], var_to_col=var_to_col),
            ]
        )
        assert_allclose(batched, expected, rtol=0, atol=1e-12)

    def test_invalid_column_mapping_raises(self):
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        data = np.zeros((2, 1), dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            compiled.log_likelihood(data, var_to_col={5: 0, 9: 5})

    def test_out_of_range_value_raises(self):
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        data = np.array([[0, 99]], dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            compiled.log_likelihood(data)

    def test_direct_compiled_matches_object_path(self):
        root = _mixed_circuit()
        compiled = CompiledCircuit(root)
        row = assignment_array({5: 0, 9: 1})
        data = row.reshape(1, -1)
        assert_allclose(
            compiled.log_likelihood(data)[0],
            log_likelihood(root, row),
            rtol=0,
            atol=1e-12,
        )
        assert_allclose(
            compiled.likelihood(row),
            likelihood(root, row),
            rtol=0,
            atol=1e-12,
        )

    def test_1d_unified_api(self):
        circuit = _mixed_circuit()
        row = assignment_array({5: 1, 9: 0})
        assert circuit.log_likelihood(row) == pytest.approx(
            circuit.compile().log_likelihood(row)
        )

    def test_batched_nan_matches_per_row(self):
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        width = max(circuit.scope_as_list()) + 1
        data = np.full((3, width), np.nan, dtype=np.float64)
        data[0, 5] = 0.0
        data[1, 9] = 1.0
        data[2, 5] = 1.0
        data[2, 9] = 0.0
        batched = compiled.log_likelihood(data)
        per_row = np.array([circuit.log_likelihood(data[r]) for r in range(3)])
        assert_allclose(batched, per_row, rtol=0, atol=1e-12)

    def test_var_to_col_with_nan(self):
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        data = np.array([[1.0, np.nan], [np.nan, 0.0]], dtype=np.float64)
        var_to_col = {5: 1, 9: 0}
        batched = compiled.log_likelihood(data, var_to_col=var_to_col)
        expected = np.array(
            [
                circuit.log_likelihood(data[0], var_to_col=var_to_col),
                circuit.log_likelihood(data[1], var_to_col=var_to_col),
            ]
        )
        assert_allclose(batched, expected, rtol=0, atol=1e-12)


class TestOpenMPBatchInvariance:
    """Compiled batch eval must not depend on OpenMP thread count."""

    @pytest.fixture
    def omp(self):
        from sparc._graph import (
            _omp_enabled,
            _omp_max_threads,
            _omp_set_num_threads,
        )

        prev = int(_omp_max_threads())
        yield _omp_set_num_threads, _omp_enabled()
        _omp_set_num_threads(max(prev, 1))

    def test_log_likelihood_invariant_across_threads(self, omp):
        set_n, _enabled = omp
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        rng = np.random.default_rng(1)
        width = max(circuit.scope_as_list()) + 1
        sizes = (1, 2, 3, 63, 64, 65, 1000)
        thread_counts = (1, 2, 3, 8)
        for n_rows in sizes:
            data = rng.integers(0, 2, size=(n_rows, width), dtype=np.int32)
            set_n(1)
            ref = compiled.log_likelihood(data)
            for nt in thread_counts:
                set_n(nt)
                got = compiled.log_likelihood(data)
                assert_allclose(got, ref, rtol=0, atol=1e-12)

    def test_likelihood_linear_space_invariant(self, omp):
        set_n, _enabled = omp
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        rng = np.random.default_rng(2)
        width = max(circuit.scope_as_list()) + 1
        data = rng.integers(0, 2, size=(128, width), dtype=np.int32)
        set_n(1)
        ref = compiled.likelihood(data)
        set_n(8)
        got = compiled.likelihood(data)
        assert_allclose(got, ref, rtol=0, atol=1e-12)

    def test_large_fanin_sum_invariant(self, omp):
        set_n, _enabled = omp
        n_children = 65
        children = [
            CategoricalInputNode(0, [0.35, 0.65]) for _ in range(n_children)
        ]
        w = [1.0 / n_children] * n_children
        root = SumNode(children, w)
        compiled = root.compile()
        data = np.array([[0], [1]] * 64, dtype=np.int32)
        set_n(1)
        ref = compiled.log_likelihood(data)
        set_n(4)
        got = compiled.log_likelihood(data)
        assert_allclose(got, ref, rtol=0, atol=1e-12)
        per_row = np.array([root.log_likelihood(data[r]) for r in range(4)])
        assert_allclose(got[:4], per_row, rtol=0, atol=1e-10)

    def test_compiled_matches_object_graph_multithreaded(self, omp):
        set_n, _enabled = omp
        set_n(8)
        circuit = _mixed_circuit()
        compiled = circuit.compile()
        rng = np.random.default_rng(3)
        width = max(circuit.scope_as_list()) + 1
        data = rng.integers(0, 2, size=(80, width), dtype=np.int32)
        batched = compiled.log_likelihood(data)
        per_row = np.array([circuit.log_likelihood(data[r]) for r in range(80)])
        assert_allclose(batched, per_row, rtol=0, atol=1e-10)


class TestNumThreadsApi:
    def test_assignment_and_setter(self):
        import sparc
        from sparc._graph import _omp_enabled

        prev = sparc.num_threads
        try:
            sparc.num_threads = 2
            if _omp_enabled():
                assert sparc.get_num_threads() == 2
                assert sparc.num_threads == 2
            else:
                assert sparc.get_num_threads() == 1
            sparc.set_num_threads(1)
            assert sparc.num_threads == 1
            with pytest.raises(ValueError):
                sparc.set_num_threads(0)
        finally:
            sparc.num_threads = max(int(prev), 1)


class TestEvalPathParity:
    """Object-path eval must agree with root-node API and compiled path."""

    def test_likelihood_wrapper_vs_function(self):
        circuit = _mixed_circuit()
        row = assignment_array({5: 1, 9: 0})
        assert circuit.likelihood(row) == pytest.approx(
            likelihood(circuit, row)
        )

    def test_log_likelihood_consistency(self):
        circuit = _mixed_circuit()
        row = assignment_array({5: 0, 9: 0})
        assert circuit.log_likelihood(row) == pytest.approx(
            np.log(circuit.likelihood(row))
        )

    def test_sample_wrapper_vs_function(self):
        root = _mixed_circuit()
        a = sample(root, 50, seed=7)
        b = root.sample(50, seed=7)
        assert_allclose(a, b)

    def test_shared_subtree_eval_consistent(self):
        leaf = CategoricalInputNode(scope_var=0, probabilities=[0.2, 0.8])
        shared = ProductNode(
            children=[
                leaf,
                CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5]),
            ],
        )
        root = SumNode(children=[shared, shared], parameters=[0.4, 0.6])
        circuit = root
        row = assignment_array({0: 1, 1: 0})
        assert circuit.likelihood(row) == pytest.approx(likelihood(root, row))
