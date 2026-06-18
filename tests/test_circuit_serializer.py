"""Roundtrip tests for CircuitSerializer."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sparc import (
    CategoricalInputNode,
    Circuit,
    CircuitSerializer,
    ProductNode,
    SumNode,
)


def _assert_tree_equal(a, b) -> None:
    assert type(a) is type(b)
    if isinstance(a, CategoricalInputNode):
        assert a.scope_as_list() == b.scope_as_list()
        assert a.probabilities_list() == pytest.approx(b.probabilities_list())
        return
    if isinstance(a, SumNode):
        assert a.parameters_list() == pytest.approx(b.parameters_list())
    elif isinstance(a, ProductNode):
        pass
    else:
        raise AssertionError(type(a))
    assert len(a.children()) == len(b.children())
    for ca, cb in zip(a.children(), b.children()):
        _assert_tree_equal(ca, cb)


def _build_tree():
    l0a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.8, 0.2])
    l1a = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
    l0b = CategoricalInputNode(id=2, scope_var=0, probabilities=[0.3, 0.7])
    l1b = CategoricalInputNode(id=3, scope_var=1, probabilities=[0.25, 0.75])
    p0 = ProductNode(id=4, children=[l0a, l1a])
    p1 = ProductNode(id=5, children=[l0b, l1b])
    root = SumNode(id=6, children=[p0, p1], parameters=[0.6, 0.4])
    root.propagate_scope()
    return root


def test_roundtrip_tree():
    root = _build_tree()
    out = CircuitSerializer.loads(CircuitSerializer.dumps(root))
    _assert_tree_equal(root, out)


def test_roundtrip_shared_child_twice():
    leaf = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.2, 0.3, 0.5])
    root = SumNode(id=1, children=[leaf, leaf], parameters=[0.4, 0.6])
    root.propagate_scope()

    out = CircuitSerializer.loads(CircuitSerializer.dumps(root))
    assert len(out.children()) == 2
    assert out.children()[0] is out.children()[1]


def test_roundtrip_dag_two_parents():
    shared = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
    l1a = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.3, 0.7])
    l1b = CategoricalInputNode(id=2, scope_var=1, probabilities=[0.6, 0.4])
    p1 = ProductNode(id=3, children=[shared, l1a])
    p2 = ProductNode(id=4, children=[shared, l1b])
    root = SumNode(id=5, children=[p1, p2], parameters=[0.5, 0.5])
    root.propagate_scope()

    out = CircuitSerializer.loads(CircuitSerializer.dumps(root))
    assert out.children()[0].children()[0] is out.children()[1].children()[0]


def test_save_load_file():
    root = _build_tree()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.json"
        CircuitSerializer.save(root, path, indent=None)
        out = CircuitSerializer.load(path)
    _assert_tree_equal(root, out)


def test_circuit_save_load_wrapper():
    circuit = Circuit(_build_tree())
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.json"
        circuit.save(path, indent=None)
        restored = Circuit.load(path)
    _assert_tree_equal(circuit.root, restored.root)


def test_gaussian_rejected_on_load():
    payload = {
        "format": "gcw-circuit-v1",
        "backend": "numpy",
        "root": 0,
        "nodes": [
            {
                "id": 0,
                "kind": "gaussian",
                "children": [],
                "scope": [0],
                "mean": 0.0,
                "std": 1.0,
            }
        ],
    }
    import json

    with pytest.raises(ValueError, match="Gaussian"):
        CircuitSerializer.loads(json.dumps(payload))


def test_load_categorical_near_unity_sum():
    """Learned PCs from JSON may have PMFs that sum to ~1 due to float drift."""
    import json

    payload = {
        "format": "gcw-circuit-v1",
        "backend": "numpy",
        "root": 0,
        "nodes": [
            {
                "id": 0,
                "kind": "categorical",
                "children": [],
                "scope": [0],
                "params": [0.5, 0.4999999799440343],
            }
        ],
    }
    root = CircuitSerializer.loads(json.dumps(payload))
    assert root.probabilities_list() == pytest.approx([0.5, 0.4999999799440343])


def test_multi_var_categorical_rejected_on_load():
    payload = {
        "format": "gcw-circuit-v1",
        "backend": "numpy",
        "root": 0,
        "nodes": [
            {
                "id": 0,
                "kind": "categorical",
                "children": [],
                "scope": [0, 1],
                "params": [0.5, 0.5],
            }
        ],
    }
    import json

    with pytest.raises(ValueError, match="single-variable"):
        CircuitSerializer.loads(json.dumps(payload))


@pytest.mark.pyjuice
def test_pyjuice_roundtrip_likelihood():
    from sparc.builders import PyjuiceBuilder

    juice = pytest.importorskip("pyjuice", reason="pyjuice not installed")
    torch = pytest.importorskip("torch", reason="torch not installed")

    def _tiny_pc(device="cpu", num_vars=4, num_latents=2):
        rng = np.random.RandomState(0)
        data = torch.from_numpy(rng.randint(0, 3, size=(64, num_vars))).to(device)
        ns = juice.structures.HCLT(data.float(), num_latents=num_latents)
        return juice.compile(ns)

    pc = _tiny_pc()
    builder = PyjuiceBuilder(block_size=pc.root_ns.num_chs)
    circuit = builder.build(pc)
    assignment = {v: 0 for v in circuit.root.scope_as_list()}
    ll_before = circuit.likelihood(assignment)

    text = CircuitSerializer.dumps(circuit)
    restored_root = CircuitSerializer.loads(text)
    restored = Circuit(restored_root)
    ll_after = restored.likelihood(assignment)

    assert ll_before > 0.0
    assert ll_after == pytest.approx(ll_before)
    assert math.isfinite(restored.log_likelihood(assignment))
