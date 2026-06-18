"""Recursive grid-decomposition circuit structures.

A circuit is built over a multi-dimensional grid of variables by recursively
cutting the current hypercube along candidate split coordinates. Each cut yields
a product of the two sub-hypercubes; a region mixes the products of all its
admissible cuts. Leaf hypercubes either factorize into per-variable inputs or
delegate to a user-supplied sub-circuit builder.

This file also provides a data-driven variant whose leaf regions are hidden
tree-structured sub-circuits.
"""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from sparc.builders._factory import _NodeFactory
from sparc.circuit import Circuit
from sparc.structures._blocks import (
    Block,
    dense_product_block,
    input_block,
    product_block,
    root_circuit,
    summate,
)
from sparc.structures.hclt import hclt_block
from sparc.structures.distributions import (
    InputDistribution,
    resolve_input_distribution,
)

_STRUCTURE_TYPES = ("sum_dominated", "prod_dominated")

Box = Tuple[Tuple[int, int], ...]
LeafBuilder = Callable[[_NodeFactory, List[int], int], Block]


def _strides(shape: Sequence[int]) -> List[int]:
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def _box_vars(box: Box, strides: Sequence[int]) -> List[int]:
    ranges = [range(lo, hi) for lo, hi in box]
    out = []
    for coord in itertools.product(*ranges):
        out.append(sum(c * s for c, s in zip(coord, strides)))
    return sorted(out)


def _normalize_intervals(
    split_intervals: Optional[Union[int, Sequence[int]]], ndim: int
) -> Optional[List[int]]:
    if split_intervals is None:
        return None
    if isinstance(split_intervals, int):
        return [split_intervals] * ndim
    intervals = list(split_intervals)
    if len(intervals) != ndim:
        raise ValueError("split_intervals length must match number of dimensions")
    return [int(x) for x in intervals]


def _split_coords(
    d: int,
    lo: int,
    hi: int,
    split_points: Optional[Sequence[Sequence[int]]],
    intervals: Optional[Sequence[int]],
) -> List[int]:
    if split_points is not None:
        return [s for s in split_points[d] if lo < s < hi]
    if intervals is not None:
        step = intervals[d]
        if step <= 0:
            raise ValueError("split interval must be positive")
        coords = []
        s = ((lo // step) + 1) * step
        while s < hi:
            coords.append(s)
            s += step
        return coords
    mid = (lo + hi) // 2
    return [mid] if lo < mid < hi else []


def _PD(
    data_shape: Sequence[int],
    num_latents: int,
    *,
    split_intervals: Optional[Union[int, Sequence[int]]],
    split_points: Optional[Sequence[Sequence[int]]],
    max_split_depth: Optional[int],
    structure_type: str,
    dist: InputDistribution,
    concentration: float,
    leaf_builder: Optional[LeafBuilder],
) -> Circuit:
    if structure_type not in _STRUCTURE_TYPES:
        raise ValueError(f"structure_type must be one of {_STRUCTURE_TYPES}")
    if num_latents < 1:
        raise ValueError("num_latents must be >= 1")

    shape = [int(x) for x in data_shape]
    ndim = len(shape)
    if ndim == 0 or any(s < 1 for s in shape):
        raise ValueError("data_shape must be non-empty with positive dimensions")
    strides = _strides(shape)
    intervals = _normalize_intervals(split_intervals, ndim)

    def leaf_region(box: Box, variables: List[int]) -> Block:
        if leaf_builder is not None:
            return leaf_builder(factory, variables, num_latents)
        var_blocks = [
            input_block(factory, v, num_latents, dist) for v in variables
        ]
        prod = product_block(factory, var_blocks)
        return summate(factory, [prod], num_latents, concentration)

    def construct(box: Box, depth: int) -> Block:
        variables = _box_vars(box, strides)
        if len(variables) == 1:
            return input_block(factory, variables[0], num_latents, dist)

        can_split = max_split_depth is None or depth < max_split_depth
        partitions: List[Tuple[Box, Box]] = []
        if can_split:
            for d in range(ndim):
                lo, hi = box[d]
                if hi - lo <= 1:
                    continue
                for s in _split_coords(d, lo, hi, split_points, intervals):
                    low_box = list(box)
                    high_box = list(box)
                    low_box[d] = (lo, s)
                    high_box[d] = (s, hi)
                    partitions.append((tuple(low_box), tuple(high_box)))

        if not partitions:
            return leaf_region(box, variables)

        prod_blocks: List[Block] = []
        for low_box, high_box in partitions:
            left = construct(low_box, depth + 1)
            right = construct(high_box, depth + 1)
            if structure_type == "prod_dominated":
                prod_blocks.append(dense_product_block(factory, left, right))
            else:
                prod_blocks.append(product_block(factory, [left, right]))
        return summate(factory, prod_blocks, num_latents, concentration)

    factory = _NodeFactory()
    full_box: Box = tuple((0, s) for s in shape)
    top = construct(full_box, 0)
    root = summate(factory, [top], 1, concentration)
    return root_circuit(root)


def PD(
    data_shape: Sequence[int],
    num_latents: int,
    *,
    split_intervals: Optional[Union[int, Sequence[int]]] = None,
    split_points: Optional[Sequence[Sequence[int]]] = None,
    max_split_depth: Optional[int] = None,
    structure_type: str = "sum_dominated",
    input_dist: Optional[InputDistribution] = None,
    num_cats: int = 256,
    sum_concentration: float = 1.0,
    seed: Optional[int] = None,
) -> Circuit:
    """Build a recursive grid-decomposition circuit over a hypercube of variables.

    Parameters
    ----------
    data_shape:
        Grid shape; the number of variables is the product of the dimensions,
        indexed in row-major (C) order.
    num_latents:
        Number of sum nodes per region.
    split_intervals:
        Spacing of candidate split coordinates per dimension (int or per-dim
        sequence). If neither this nor ``split_points`` is given, each dimension
        is split at its midpoint.
    split_points:
        Explicit candidate split coordinates per dimension.
    max_split_depth:
        Optional cap on recursion depth; regions reaching it are factorized.
    structure_type:
        ``"sum_dominated"`` (index-aligned products) or ``"prod_dominated"``
        (all pairwise products before mixing).
    input_dist:
        Leaf distribution; defaults to a categorical over ``num_cats``.
    sum_concentration:
        Dirichlet concentration for randomly initialized weights.
    seed:
        Optional RNG seed for reproducibility.

    Notes
    -----
    The result is smooth and decomposable (valid for likelihood, sampling, and
    maximum-likelihood training). Because a region mixes several distinct
    decompositions, it is generally not structurally decomposable, so pairwise
    transport / expectation queries between two such circuits may not apply.
    """
    if seed is not None:
        np.random.seed(seed)
    dist = resolve_input_distribution(input_dist, num_cats)
    return _PD(
        data_shape,
        num_latents,
        split_intervals=split_intervals,
        split_points=split_points,
        max_split_depth=max_split_depth,
        structure_type=structure_type,
        dist=dist,
        concentration=sum_concentration,
        leaf_builder=None,
    )


def PDHCLT(
    data: np.ndarray,
    data_shape: Sequence[int],
    num_latents: int,
    *,
    split_intervals: Optional[Union[int, Sequence[int]]] = None,
    split_points: Optional[Sequence[Sequence[int]]] = None,
    max_split_depth: Optional[int] = None,
    structure_type: str = "sum_dominated",
    input_dist: Optional[InputDistribution] = None,
    num_cats: int = 256,
    num_bins: int = 32,
    sum_concentration: float = 1.0,
    seed: Optional[int] = None,
) -> Circuit:
    """Grid-decomposition circuit whose leaf regions are hidden tree sub-circuits.

    Identical grid structure to :func:`PD`, but each leaf hypercube is built as a
    hidden tree-structured sub-circuit over its variables, estimated from the
    corresponding columns of ``data``.

    Parameters
    ----------
    data:
        2-D array ``(n_samples, n_vars)``; column ``v`` corresponds to variable
        ``v`` in the grid's row-major order.
    data_shape:
        Grid shape (its product must equal the number of data columns).
    num_latents:
        Number of sum nodes per region and per leaf-subtree variable.
    num_bins:
        Bins used for mutual-information estimation in the leaf sub-circuits.

    See :func:`PD` for the remaining parameters and the structural-decomposability
    note.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_samples, n_vars)")
    n_vars = int(np.prod([int(x) for x in data_shape]))
    if data.shape[1] != n_vars:
        raise ValueError("data columns must equal the product of data_shape")
    if seed is not None:
        np.random.seed(seed)

    dist = resolve_input_distribution(input_dist, num_cats)

    def leaf_builder(factory: _NodeFactory, variables: List[int], k: int) -> Block:
        data_sub = data[:, variables]
        return hclt_block(
            factory,
            data_sub,
            variables,
            k,
            k,
            num_bins=num_bins,
            dist=dist,
            concentration=sum_concentration,
        )

    return _PD(
        data_shape,
        num_latents,
        split_intervals=split_intervals,
        split_points=split_points,
        max_split_depth=max_split_depth,
        structure_type=structure_type,
        dist=dist,
        concentration=sum_concentration,
        leaf_builder=leaf_builder,
    )
