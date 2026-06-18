"""Dependency-tree estimation from data: mutual information + spanning tree.

Used to derive a tree-shaped dependency backbone over observed variables. The
implementation is self-contained NumPy: pairwise mutual information from binned
joint histograms, a maximum spanning tree (Prim), a graph-center root choice,
and a directed children map obtained by a breadth-first walk from the root.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np


def _discretize(data: np.ndarray, num_bins: int) -> np.ndarray:
    """Map each column to integer bins ``0 .. num_bins-1`` by min-max scaling."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("data must be 2-D (n_samples, n_vars)")
    n_vars = arr.shape[1]
    binned = np.zeros(arr.shape, dtype=np.int64)
    for j in range(n_vars):
        col = arr[:, j]
        lo = col.min()
        hi = col.max()
        if hi <= lo:
            binned[:, j] = 0
            continue
        scaled = (col - lo) / (hi - lo) * (num_bins - 1)
        binned[:, j] = np.clip(np.rint(scaled), 0, num_bins - 1).astype(np.int64)
    return binned


def mutual_information(data: np.ndarray, num_bins: int = 32) -> np.ndarray:
    """Pairwise mutual-information matrix of the columns of ``data``.

    Each variable is binned into ``num_bins`` levels; MI is computed from the
    normalized joint histogram of every pair. The diagonal is set to zero.
    """
    binned = _discretize(data, num_bins)
    n_samples, n_vars = binned.shape
    if n_samples < 1:
        raise ValueError("data must contain at least one sample")

    # Per-variable marginals.
    marginals: List[np.ndarray] = []
    for j in range(n_vars):
        counts = np.bincount(binned[:, j], minlength=num_bins).astype(np.float64)
        marginals.append(counts / n_samples)

    mi = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            joint = np.zeros((num_bins, num_bins), dtype=np.float64)
            np.add.at(joint, (binned[:, i], binned[:, j]), 1.0)
            joint /= n_samples
            pi = marginals[i][:, None]
            pj = marginals[j][None, :]
            denom = pi * pj
            mask = (joint > 0.0) & (denom > 0.0)
            val = float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))
            mi[i, j] = val
            mi[j, i] = val
    return mi


def maximum_spanning_tree(weights: np.ndarray) -> List[List[int]]:
    """Undirected maximum spanning tree of a symmetric weight matrix (Prim).

    Returns an adjacency list. For ``n <= 1`` the tree has no edges.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = w.shape[0]
    adjacency: List[List[int]] = [[] for _ in range(n)]
    if n <= 1:
        return adjacency

    in_tree = np.zeros(n, dtype=bool)
    best_weight = np.full(n, -np.inf, dtype=np.float64)
    best_parent = np.full(n, -1, dtype=np.int64)
    best_weight[0] = np.inf  # seed node

    for _ in range(n):
        u = -1
        u_w = -np.inf
        for v in range(n):
            if not in_tree[v] and best_weight[v] > u_w:
                u_w = best_weight[v]
                u = v
        if u == -1:
            break
        in_tree[u] = True
        if best_parent[u] != -1:
            p = int(best_parent[u])
            adjacency[u].append(p)
            adjacency[p].append(u)
        for v in range(n):
            if not in_tree[v] and w[u, v] > best_weight[v]:
                best_weight[v] = w[u, v]
                best_parent[v] = u
    return adjacency


def tree_center(adjacency: List[List[int]]) -> int:
    """A center node of a tree, found by iterative leaf peeling.

    Repeatedly removes current leaves until one or two nodes remain; returns one
    of them. This minimizes tree depth (eccentricity) from the chosen root.
    """
    n = len(adjacency)
    if n == 0:
        raise ValueError("empty tree has no center")
    if n == 1:
        return 0

    degree = [len(adj) for adj in adjacency]
    remaining = n
    leaves = deque(v for v in range(n) if degree[v] <= 1)
    pruned = [False] * n
    last_layer: List[int] = []

    while remaining > 2:
        layer_size = len(leaves)
        last_layer = []
        for _ in range(layer_size):
            leaf = leaves.popleft()
            pruned[leaf] = True
            last_layer.append(leaf)
            remaining -= 1
            for nb in adjacency[leaf]:
                if not pruned[nb]:
                    degree[nb] -= 1
                    if degree[nb] == 1:
                        leaves.append(nb)

    centers = [v for v in range(n) if not pruned[v]]
    return centers[0]


def rooted_children(adjacency: List[List[int]], root: int) -> Dict[int, List[int]]:
    """Directed children map of the tree obtained by a BFS from ``root``."""
    n = len(adjacency)
    children: Dict[int, List[int]] = {v: [] for v in range(n)}
    if n == 0:
        return children
    visited = [False] * n
    visited[root] = True
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if not visited[v]:
                visited[v] = True
                children[u].append(v)
                queue.append(v)
    return children


def chow_liu_children(
    data: np.ndarray, num_bins: int = 32
) -> Dict[int, List[int]]:
    """Convenience: data -> MI -> maximum spanning tree -> rooted children map."""
    mi = mutual_information(data, num_bins=num_bins)
    adjacency = maximum_spanning_tree(mi)
    root = tree_center(adjacency)
    return rooted_children(adjacency, root)
