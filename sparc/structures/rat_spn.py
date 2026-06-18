"""Randomized, tensorized sum-product circuit structures.

A circuit is built from several independent *repetitions*. Each repetition
recursively partitions the variable scope into ``num_pieces`` random sub-scopes
down to ``depth`` levels (or until a single variable remains), placing a product
block at each split and a sum block of ``num_latents`` mixtures at each region.
The repetitions' top blocks are merged under a single root mixture, yielding a
mixture over random circuit shapes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from sparc.builders._factory import _NodeFactory
from sparc.circuit import Circuit
from sparc.structures._blocks import (
    input_block,
    product_block,
    root_circuit,
    summate,
)
from sparc.structures.distributions import (
    InputDistribution,
    resolve_input_distribution,
)


def _random_partition(scope: Sequence[int], num_pieces: int) -> List[List[int]]:
    items = list(scope)
    np.random.shuffle(items)
    pieces = [items[i::num_pieces] for i in range(num_pieces)]
    return [p for p in pieces if len(p) > 0]


def RAT_SPN(
    num_vars: int,
    num_latents: int,
    depth: int,
    num_repetitions: int,
    *,
    num_pieces: int = 2,
    input_dist: Optional[InputDistribution] = None,
    num_cats: int = 256,
    sum_concentration: float = 1.0,
    seed: Optional[int] = None,
) -> Circuit:
    """Build a randomized tensorized sum-product circuit.

    Parameters
    ----------
    num_vars:
        Number of observed variables (indexed ``0 .. num_vars - 1``).
    num_latents:
        Number of sum nodes per region.
    depth:
        Maximum recursive partition depth before regions are factorized.
    num_repetitions:
        Number of independent random partitions mixed at the root.
    num_pieces:
        Branching factor of each partition.
    input_dist:
        Leaf distribution; defaults to a categorical over ``num_cats``.
    sum_concentration:
        Dirichlet concentration for randomly initialized sum weights.
    seed:
        Optional RNG seed for reproducibility.
    """
    if num_vars < 1:
        raise ValueError("num_vars must be >= 1")
    if num_latents < 1:
        raise ValueError("num_latents must be >= 1")
    if num_pieces < 2:
        raise ValueError("num_pieces must be >= 2")
    if num_repetitions < 1:
        raise ValueError("num_repetitions must be >= 1")
    if seed is not None:
        np.random.seed(seed)

    factory = _NodeFactory()
    dist = resolve_input_distribution(input_dist, num_cats)

    def region(scope: Sequence[int], level: int):
        scope = list(scope)
        if len(scope) == 1:
            leaves = input_block(factory, scope[0], num_latents, dist)
            return summate(factory, [leaves], num_latents, sum_concentration)
        if level >= depth:
            # Factorize the remaining scope into per-variable leaves.
            var_blocks = [
                input_block(factory, v, num_latents, dist) for v in scope
            ]
            prod = product_block(factory, var_blocks)
            return summate(factory, [prod], num_latents, sum_concentration)
        parts = _random_partition(scope, num_pieces)
        if len(parts) == 1:
            # Degenerate split (all items in one piece): factorize instead.
            var_blocks = [
                input_block(factory, v, num_latents, dist) for v in scope
            ]
            prod = product_block(factory, var_blocks)
            return summate(factory, [prod], num_latents, sum_concentration)
        sub_blocks = [region(part, level + 1) for part in parts]
        prod = product_block(factory, sub_blocks)
        return summate(factory, [prod], num_latents, sum_concentration)

    full_scope = list(range(num_vars))
    rep_blocks = [region(full_scope, 0) for _ in range(num_repetitions)]
    root = summate(factory, rep_blocks, 1, sum_concentration)
    return root_circuit(root)
