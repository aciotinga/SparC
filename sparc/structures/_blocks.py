"""Block construction algebra for assembling probabilistic circuits.

A *block* is a list of scalar circuit nodes that all range over the same
variable scope -- the scalar-node analogue of a vector of latent states. Every
structure in this package is written purely in terms of the three primitives
below, which keeps each structure definition free of node-class details and id
bookkeeping:

- :func:`input_block` -- a block of leaf nodes over a single variable.
- :func:`product_block` / :func:`dense_product_block` -- decomposition by
  combining child blocks that range over disjoint scopes.
- :func:`summate` -- a block of dense mixtures over a collection of child
  blocks (mixing).

Because the scope partitioning produced by these helpers is fully determined by
the structure arguments (and the seeded RNG), two circuits built with identical
arguments are structurally decomposable and therefore valid inputs to the
pairwise transport / expectation queries.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from sparc.builders._factory import _NodeFactory
from sparc.circuit import Circuit
from sparc.structures.distributions import InputDistribution

Block = List[object]


def input_block(
    factory: _NodeFactory,
    var: int,
    num_nodes: int,
    dist: InputDistribution,
) -> Block:
    """A block of ``num_nodes`` leaf nodes over a single variable ``var``."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    return [dist.create(factory, int(var)) for _ in range(num_nodes)]


def product_block(factory: _NodeFactory, child_blocks: Sequence[Block]) -> Block:
    """Index-aligned products across ``child_blocks`` of equal size.

    Product ``i`` joins the ``i``-th node of every child block. The child blocks
    must range over disjoint scopes (the caller guarantees this by construction).
    """
    blocks = [list(b) for b in child_blocks if len(b) > 0]
    if not blocks:
        raise ValueError("product_block requires at least one non-empty block")
    k = len(blocks[0])
    for b in blocks:
        if len(b) != k:
            raise ValueError("all child blocks must have equal size")
    if len(blocks) == 1:
        return list(blocks[0])
    return [factory.product([b[i] for b in blocks]) for i in range(k)]


def dense_product_block(factory: _NodeFactory, left: Block, right: Block) -> Block:
    """All pairwise products of ``left`` and ``right`` (size ``len*len``).

    Used by product-dominated decompositions where every pair of sub-region
    states is combined before mixing.
    """
    if len(left) == 0 or len(right) == 0:
        raise ValueError("dense_product_block requires non-empty blocks")
    return [factory.product([a, b]) for a in left for b in right]


def summate(
    factory: _NodeFactory,
    child_blocks: Sequence[Block],
    num_nodes: int,
    concentration: float = 1.0,
) -> Block:
    """A block of ``num_nodes`` dense mixtures over all nodes in ``child_blocks``.

    Every output sum node mixes the union of nodes across the given child blocks
    (which must share a common scope), with weights drawn from a symmetric
    Dirichlet of the given concentration.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    children = [node for block in child_blocks for node in block]
    if not children:
        raise ValueError("summate requires at least one child node")
    out: Block = []
    for _ in range(num_nodes):
        weights = np.random.dirichlet(np.ones(len(children)) * concentration)
        out.append(factory.sum(children, weights))
    return out


def root_circuit(block: Block) -> Circuit:
    """Wrap a size-1 block's single node as a :class:`Circuit` root."""
    if len(block) != 1:
        raise ValueError(f"root block must have exactly one node, got {len(block)}")
    return Circuit(block[0])
