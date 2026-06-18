"""Internal node factory with monotonic id allocation (one per build())."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np

from sparc.nodes import CategoricalInputNode, ProductNode, SumNode

_ArrayLike = Union[Sequence[float], np.ndarray]


def _as_float_list(values: _ArrayLike) -> List[float]:
    if isinstance(values, np.ndarray):
        return values.astype(float).tolist()
    return [float(v) for v in values]


class _NodeFactory:
    def __init__(self) -> None:
        self._next_id = 0

    def _alloc_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def categorical(self, scope_var: int, probabilities: _ArrayLike) -> CategoricalInputNode:
        return CategoricalInputNode(
            self._alloc_id(), int(scope_var), _as_float_list(probabilities)
        )

    def product(self, children: Iterable) -> ProductNode:
        return ProductNode(self._alloc_id(), list(children))

    def sum(self, children: Iterable, parameters: _ArrayLike) -> SumNode:
        return SumNode(self._alloc_id(), list(children), _as_float_list(parameters))
