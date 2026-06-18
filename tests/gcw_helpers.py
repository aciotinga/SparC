"""Shared helpers for GCW coupling / crossterm tests.

General utilities live in ``tests.sparc_helpers``; this module adds GCW-specific
stress patterns and re-exports common helpers for backward compatibility.
"""

from __future__ import annotations

import numpy as np

from sparc import CategoricalInputNode, SumNode
from tests.sparc_helpers import (
    empirical_marginal,
    exact_marginal,
    make_categorical,
    make_product,
    make_sum,
    nw_coupling_dense,
    sum_mixture_marginal,
    var_cardinalities,
    walk_pc_invariants,
)

__all__ = [
    "empirical_marginal",
    "exact_marginal",
    "make_categorical",
    "make_product",
    "make_sum",
    "nw_coupling_dense",
    "pollute_heap_with_couplings",
    "q_var_offset",
    "sum_mixture_marginal",
    "var_cardinalities",
    "walk_pc_invariants",
]


def q_var_offset(circuit1) -> int:
    vars1 = circuit1.scope_as_list()
    return (max(vars1) + 1) if vars1 else 0


def pollute_heap_with_couplings(
    gcw_coupling_circuit,
    *,
    rounds: int = 12,
    seed: int = 0,
) -> None:
    """Allocate many short-lived GCW couplings to perturb the object heap."""
    rng = np.random.default_rng(seed)
    for r in range(rounds):
        p = rng.dirichlet([1.0, 1.0, 1.0])
        q = rng.dirichlet([1.0, 1.0, 1.0])
        leaf1 = make_categorical(1000 + 3 * r, 0, p)
        leaf2 = make_categorical(1001 + 3 * r, 0, q)
        circ = gcw_coupling_circuit(leaf1, leaf2)
        circ.sample(500, seed=r)
