"""Internal node factory with monotonic id allocation (one per build())."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np

from sparc.nodes import (
    BernoulliInputNode,
    CategoricalInputNode,
    DiscreteLogisticInputNode,
    IndicatorInputNode,
    LiteralInputNode,
    ProductNode,
    SumNode,
)

_ArrayLike = Union[Sequence[float], np.ndarray]


def _as_float_list(values: _ArrayLike) -> List[float]:
    if isinstance(values, np.ndarray):
        return values.astype(float).tolist()
    return [float(v) for v in values]


class _NodeFactory:
    def categorical(self, scope_var: int, probabilities: _ArrayLike) -> CategoricalInputNode:
        return CategoricalInputNode(
            int(scope_var), _as_float_list(probabilities)
        )

    def bernoulli(self, scope_var: int, p: float) -> BernoulliInputNode:
        return BernoulliInputNode(int(scope_var), float(p))

    def indicator(self, scope_var: int, value: int, num_cats: int) -> IndicatorInputNode:
        return IndicatorInputNode(
            int(scope_var), int(value), int(num_cats)
        )

    def literal(self, scope_var: int, value: int) -> LiteralInputNode:
        return LiteralInputNode(int(scope_var), int(value))

    def discrete_logistic(
        self, scope_var: int, mu: float, s: float, num_cats: int
    ) -> DiscreteLogisticInputNode:
        return DiscreteLogisticInputNode(
            int(scope_var), float(mu), float(s), int(num_cats)
        )

    def product(self, children: Iterable) -> ProductNode:
        return ProductNode(list(children))

    def sum(self, children: Iterable, parameters: _ArrayLike) -> SumNode:
        return SumNode(list(children), _as_float_list(parameters))
