"""Parity between object-graph and CompiledCircuit query paths."""

from __future__ import annotations

import pytest

from sparc import (
    CategoricalInputNode,
        ProductNode,
    SumNode,
    cw_distance,
    cw_distance_and_grad,
    exp_query,
    expected_squared_distance,
)

pytestmark = pytest.mark.eval


def _simple_pair():
    l0 = CategoricalInputNode(scope_var=0, probabilities=[0.6, 0.4])
    l1 = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
    p = ProductNode(children=[l0, l1])
    r1 = SumNode(children=[p], parameters=[1.0])
    r2 = SumNode(children=[p], parameters=[1.0])
    return r1, r2


def test_cw_compiled_matches_object():
    c1, c2 = _simple_pair()
    cc1, cc2 = c1.compile(), c2.compile()
    assert cw_distance(c1, c2) == pytest.approx(cw_distance(cc1, cc2))
    v_obj, _ = cw_distance_and_grad(c1, c2)
    v_cmp, _ = cw_distance_and_grad(cc1, cc2)
    assert v_obj == pytest.approx(v_cmp)


def test_exp_compiled_matches_object():
    c1, c2 = _simple_pair()
    cc1, cc2 = c1.compile(), c2.compile()
    assert exp_query(c1, c2) == pytest.approx(exp_query(cc1, cc2))


def test_esd_compiled_matches_object():
    c1, _ = _simple_pair()
    cc1 = c1.compile()
    assert expected_squared_distance(c1) == pytest.approx(
        expected_squared_distance(cc1)
    )


def test_mixed_types_raise():
    c1, c2 = _simple_pair()
    with pytest.raises(TypeError, match="same kind"):
        cw_distance(c1, c2.compile())
