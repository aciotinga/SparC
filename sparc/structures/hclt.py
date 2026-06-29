"""Hidden tree-structured circuits derived from data dependencies.

A dependency tree over the observed variables is estimated from data (pairwise
mutual information + maximum spanning tree). Each tree variable gets a latent
block of ``num_latents`` states; the observed variable attaches as an emission
input block, and each child subtree contributes a transition mixture. The
construction generalizes the latent-chain recurrence of an HMM to an arbitrary
tree backbone.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from sparc.builders._factory import _NodeFactory
from sparc.nodes import CircuitNode
from sparc.structures._blocks import (
    Block,
    input_block,
    product_block,
    root_circuit,
    summate,
)
from sparc.structures._chowliu import (
    maximum_spanning_tree,
    mutual_information,
    rooted_children,
    tree_center,
)
from sparc.structures.distributions import (
    InputDistribution,
    resolve_input_distribution,
)


def hclt_block(
    factory: _NodeFactory,
    data: np.ndarray,
    variables: Sequence[int],
    num_latents: int,
    num_roots: int,
    *,
    num_bins: int,
    dist: InputDistribution,
    concentration: float,
) -> Block:
    """Build a hidden-tree sub-circuit and return its root block.

    ``data`` has one column per entry of ``variables`` (the global variable
    indices used for the leaf emissions). Returns a block of ``num_roots`` sum
    nodes ranging over all of ``variables``.
    """
    data = np.asarray(data)
    variables = list(variables)
    if data.ndim != 2 or data.shape[1] != len(variables):
        raise ValueError("data must be 2-D with one column per variable")
    if num_latents < 1:
        raise ValueError("num_latents must be >= 1")
    if num_roots < 1:
        raise ValueError("num_roots must be >= 1")

    n_vars = len(variables)
    if n_vars == 1:
        children_map = {0: []}
        root_local = 0
    else:
        mi = mutual_information(data, num_bins=num_bins)
        adjacency = maximum_spanning_tree(mi)
        root_local = tree_center(adjacency)
        children_map = rooted_children(adjacency, root_local)

    def compile_node(local: int) -> Block:
        var = variables[local]
        emission = input_block(factory, var, num_latents, dist)
        child_locals = children_map[local]
        if not child_locals:
            return emission
        blocks: List[Block] = [emission]
        for child in child_locals:
            child_block = compile_node(child)
            transition = summate(factory, [child_block], num_latents, concentration)
            blocks.append(transition)
        return product_block(factory, blocks)

    root_block = compile_node(root_local)
    return summate(factory, [root_block], num_roots, concentration)


def HCLT(
    data: np.ndarray,
    num_latents: int,
    *,
    num_bins: int = 32,
    num_roots: int = 1,
    input_dist: Optional[InputDistribution] = None,
    num_cats: int = 256,
    sum_concentration: float = 1.0,
    seed: Optional[int] = None,
) -> CircuitNode:
    """Build a hidden tree-structured circuit from a data matrix.

    Parameters
    ----------
    data:
        2-D array ``(n_samples, n_vars)`` used to estimate the dependency tree.
        Variables are indexed ``0 .. n_vars - 1``.
    num_latents:
        Number of latent states per variable.
    num_bins:
        Number of bins used when estimating pairwise mutual information.
    num_roots:
        Number of root mixtures before the final collapse to a single root.
    input_dist:
        Emission distribution; defaults to a categorical over ``num_cats``.
    sum_concentration:
        Dirichlet concentration for randomly initialized weights.
    seed:
        Optional RNG seed for reproducibility.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_samples, n_vars)")
    if seed is not None:
        np.random.seed(seed)

    factory = _NodeFactory()
    dist = resolve_input_distribution(input_dist, num_cats)
    variables = list(range(data.shape[1]))
    roots = hclt_block(
        factory,
        data,
        variables,
        num_latents,
        num_roots,
        num_bins=num_bins,
        dist=dist,
        concentration=sum_concentration,
    )
    if len(roots) == 1:
        return root_circuit(roots)
    return root_circuit(summate(factory, [roots], 1, sum_concentration))
