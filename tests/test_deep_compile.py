"""Tests for deep-compiled native inference."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc import (
    BernoulliInputNode,
    CategoricalInputNode,
    Circuit,
    GradBundle,
    IndicatorInputNode,
    LiteralInputNode,
    ProductNode,
    SumNode,
)
from sparc.deep_compile.compiler import smoke_compile
from sparc.optim import apply_grads

pytestmark = pytest.mark.eval

requires_compiler = pytest.mark.skipif(
    not smoke_compile(),
    reason="no working C compiler for deep_compile",
)


def _mixed_circuit():
    l0 = CategoricalInputNode(id=0, scope_var=5, probabilities=[0.6, 0.4])
    l1 = CategoricalInputNode(id=1, scope_var=9, probabilities=[0.25, 0.75])
    prod = ProductNode(id=2, children=[l0, l1])
    l2 = CategoricalInputNode(id=3, scope_var=5, probabilities=[0.1, 0.9])
    root = SumNode(id=4, children=[prod, l2], parameters=[0.7, 0.3])
    return Circuit(root)


def _bernoulli_sum_circuit():
    l0 = BernoulliInputNode(id=0, scope_var=0, p=0.9)
    l1 = BernoulliInputNode(id=1, scope_var=1, p=0.3)
    root = SumNode(id=2, children=[l0, l1], parameters=[0.8, 0.2])
    return Circuit(root)


def _deterministic_circuit():
    lit = LiteralInputNode(id=0, scope_var=0, value=1)
    ind = IndicatorInputNode(id=1, scope_var=1, value=0, num_cats=3)
    root = ProductNode(id=2, children=[lit, ind])
    return Circuit(root)


@requires_compiler
class TestDeepCompile:
    def test_1d_log_likelihood_parity(self, tmp_path):
        circuit = _mixed_circuit()
        ref = circuit.compile()
        deep = circuit.deep_compile(tmp_path / "mixed")
        row = np.zeros(10, dtype=np.int32)
        row[5] = 1
        row[9] = 0
        assert_allclose(
            deep.log_likelihood(row),
            ref.log_likelihood(row),
            rtol=0,
            atol=1e-12,
        )

    def test_1d_likelihood_parity(self, tmp_path):
        circuit = _bernoulli_sum_circuit()
        ref = circuit.compile()
        deep = circuit.deep_compile(tmp_path / "bern")
        row = np.array([1, 0], dtype=np.int32)
        assert_allclose(
            deep.likelihood(row),
            ref.likelihood(row),
            rtol=0,
            atol=1e-12,
        )

    def test_batched_matches_per_row(self, tmp_path):
        circuit = _mixed_circuit()
        ref = circuit.compile()
        deep = circuit.deep_compile(tmp_path / "mixed_batch")
        n = 32
        rng = np.random.default_rng(0)
        width = 10
        data = np.zeros((n, width), dtype=np.int32)
        data[:, 5] = rng.integers(0, 2, size=n)
        data[:, 9] = rng.integers(0, 2, size=n)
        assert_allclose(
            deep.log_likelihood(data),
            ref.log_likelihood(data),
            rtol=0,
            atol=1e-12,
        )

    def test_var_to_col_reordering(self, tmp_path):
        circuit = _mixed_circuit()
        ref = circuit.compile()
        deep = circuit.deep_compile(tmp_path / "mixed_vtc")
        data = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int32)
        var_to_col = {5: 1, 9: 0}
        assert_allclose(
            deep.log_likelihood(data, var_to_col=var_to_col),
            ref.log_likelihood(data, var_to_col=var_to_col),
            rtol=0,
            atol=1e-12,
        )

    def test_deterministic_leaves(self, tmp_path):
        circuit = _deterministic_circuit()
        ref = circuit.compile()
        deep = circuit.deep_compile(tmp_path / "det")
        row = np.array([1, 0], dtype=np.int32)
        assert_allclose(deep.likelihood(row), ref.likelihood(row), atol=1e-12)
        bad = np.array([0, 0], dtype=np.int32)
        assert_allclose(deep.likelihood(bad), ref.likelihood(bad), atol=1e-12)

    def test_artifacts_exist(self, tmp_path):
        circuit = _bernoulli_sum_circuit()
        stem = tmp_path / "artifacts"
        deep = circuit.deep_compile(stem)
        assert deep.source_path.is_file()
        assert deep.library_path.is_file()
        assert deep.source_path.suffix == ".c"

    def test_recompile_overwrites(self, tmp_path):
        circuit = _bernoulli_sum_circuit()
        stem = tmp_path / "overwrite"
        d1 = circuit.deep_compile(stem)
        d1.close()
        d2 = circuit.deep_compile(stem)
        assert d1.library_path == d2.library_path
        row = np.array([1, 1], dtype=np.int32)
        assert_allclose(
            d2.log_likelihood(row),
            circuit.compile().log_likelihood(row),
            atol=1e-12,
        )

    def test_refresh_parameters_updates_ll(self, tmp_path):
        circuit = _bernoulli_sum_circuit()
        deep = circuit.deep_compile(tmp_path / "refresh")
        data = np.array([[1, 0], [0, 1]], dtype=np.int32)
        ll_before = deep.log_likelihood(data).copy()

        grads = GradBundle()
        grads.cat_grads = {0: np.array([0.05, -0.05])}
        grads.sum_grads = {2: np.array([0.1, -0.1])}
        apply_grads(circuit.root, grads, lr=0.5)
        deep.refresh_parameters()
        ll_after = deep.log_likelihood(data)
        assert not np.allclose(ll_before, ll_after)
        assert_allclose(
            ll_after,
            circuit.compile().log_likelihood(data),
            rtol=0,
            atol=1e-12,
        )
        deep.close()


@requires_compiler
def test_managed_temp_artifacts_cleanup():
    circuit = _bernoulli_sum_circuit()
    deep = circuit.deep_compile()
    artifact_dir = deep.source_path.parent
    assert artifact_dir.is_dir()
    assert deep.library_path.is_file()
    row = np.array([1, 0], dtype=np.int32)
    assert_allclose(
        deep.log_likelihood(row),
        circuit.compile().log_likelihood(row),
        atol=1e-12,
    )
    deep.close()
    assert not artifact_dir.exists()

    with circuit.deep_compile() as managed:
        assert managed.log_likelihood(row) is not None
    assert not managed.source_path.parent.exists()


@requires_compiler
def test_generated_source_uses_sparc_op_table(tmp_path):
    circuit = _bernoulli_sum_circuit()
    deep = circuit.deep_compile(tmp_path / "src")
    text = deep.source_path.read_text(encoding="utf-8")
    assert "sparc_deep_rt.h" in text
    assert "static const SparcOp sparc_ops_log" in text
    assert "static const SparcOp sparc_ops_lin" in text
    assert "sparc_log_likelihood_row" in text
    assert "sparc_log_likelihood_batch" in text
    assert "double* tape" in text
    assert "double* workspace" in text
    assert "sparc_dispatch()" in text
    assert text.count("for (int32_t r = 0; r < n_rows") == 0
    deep.close()


@requires_compiler
@pytest.mark.parametrize("isa", ["scalar", "avx2"])
def test_isa_override_parity(tmp_path, isa):
    circuit = _mixed_circuit()
    ref = circuit.compile()
    deep = circuit.deep_compile(tmp_path / f"mixed_{isa}", simd=isa)
    assert deep.active_isa == isa
    n = 64
    rng = np.random.default_rng(1)
    width = 10
    data = np.zeros((n, width), dtype=np.int32)
    data[:, 5] = rng.integers(0, 2, size=n)
    data[:, 9] = rng.integers(0, 2, size=n)
    assert_allclose(
        deep.log_likelihood(data),
        ref.log_likelihood(data),
        rtol=0,
        atol=1e-11,
    )
    deep.close()


class TestDeepCompileValidation:
    @requires_compiler
    def test_rejects_float_data(self, tmp_path):
        circuit = _bernoulli_sum_circuit()
        deep = circuit.deep_compile(tmp_path / "val")
        with pytest.raises(ValueError, match="fully observed integer"):
            deep.log_likelihood(np.array([1.0, 0.0]))

    @requires_compiler
    def test_rejects_missing_evidence(self, tmp_path):
        circuit = _bernoulli_sum_circuit()
        deep = circuit.deep_compile(tmp_path / "val2")
        with pytest.raises(ValueError, match="fully observed integer"):
            deep.log_likelihood(np.array([1, -1], dtype=np.int32))

    @requires_compiler
    def test_rejects_nan_batch(self, tmp_path):
        circuit = _bernoulli_sum_circuit()
        deep = circuit.deep_compile(tmp_path / "val3")
        data = np.array([[1.0, np.nan]], dtype=np.float64)
        with pytest.raises(ValueError, match="fully observed integer"):
            deep.log_likelihood(data)
