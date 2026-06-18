"""Shared test utilities for the SparC library.

Centralizes brute-force references, finite-difference gradients, and PC
invariant checks so module-specific test files stay focused on behavior.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Iterable, Sequence

import numpy as np

from sparc import (
    BernoulliInputNode,
    CategoricalInputNode,
    Circuit,
    CircuitNode,
    ProductNode,
    SumNode,
    likelihood,
)


def fd_gradient_simplex(
    f: Callable[[Sequence[float]], float],
    params: Sequence[float],
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    """Symmetric FD on a simplex, returning a tangent-projected gradient."""
    n = len(params)
    dirs = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        p_plus = np.array(params, dtype=np.float64)
        p_minus = np.array(params, dtype=np.float64)
        p_plus[i] += eps
        p_plus[0] -= eps
        p_minus[i] -= eps
        p_minus[0] += eps
        dirs[i] = (f(p_plus) - f(p_minus)) / (2.0 * eps)
    g = dirs.copy()
    g[0] = 0.0
    g -= g.mean()
    return g


def project_to_simplex_tangent(g: Sequence[float]) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    return g - g.mean()


def assert_on_simplex(params: Sequence[float], *, tol: float = 1e-10) -> None:
    p = np.asarray(params, dtype=np.float64)
    assert (p >= -tol).all(), f"negative simplex entry: {p}"
    assert math.isclose(p.sum(), 1.0, abs_tol=tol), f"sum {p.sum()} != 1"


def var_cardinalities(root: CircuitNode) -> dict[int, int]:
    cards: dict[int, int] = {}

    def walk(node: CircuitNode) -> None:
        if isinstance(node, (CategoricalInputNode, BernoulliInputNode)):
            var = node.scope_as_list()[0]
            cards[var] = int(node.cardinality())
            return
        if isinstance(node, (SumNode, ProductNode)):
            for child in node.children():
                walk(child)

    walk(root)
    return cards


def enumerate_assignments(scope: Sequence[int], cards: dict[int, int]):
    ranges = [range(cards[v]) for v in scope]
    for values in itertools.product(*ranges):
        yield {scope[i]: values[i] for i in range(len(scope))}


def exact_total_mass(circuit: Circuit, *, tol: float = 1e-10) -> float:
    scope = sorted(circuit.root.scope_as_list())
    cards = var_cardinalities(circuit.root)
    total = 0.0
    for assignment in enumerate_assignments(scope, cards):
        mass = circuit.likelihood(assignment)
        if mass < -tol:
            raise ValueError(f"negative likelihood for {assignment}")
        total += mass
    return total


def exact_marginal(circuit: Circuit, var: int) -> np.ndarray:
    scope = sorted(circuit.root.scope_as_list())
    if var not in scope:
        raise ValueError(f"variable {var} not in scope {scope}")
    cards = var_cardinalities(circuit.root)
    counts = np.zeros(cards[var], dtype=np.float64)
    total = 0.0
    for assignment in enumerate_assignments(scope, cards):
        mass = circuit.likelihood(assignment)
        counts[assignment[var]] += mass
        total += mass
    if total <= 0.0:
        raise ValueError("zero total mass")
    return counts / total


def empirical_marginal(
    draws: Iterable[dict[int, int]], var: int, n_outcomes: int
) -> np.ndarray:
    counts = np.zeros(n_outcomes, dtype=np.float64)
    n = 0
    for row in draws:
        counts[row[var]] += 1.0
        n += 1
    if n == 0:
        raise ValueError("no draws")
    return counts / n


def walk_pc_invariants(root: CircuitNode, *, tol: float = 1e-10) -> None:
    if isinstance(root, (CategoricalInputNode, BernoulliInputNode)):
        assert_on_simplex(root.probabilities_list(), tol=tol)
        return
    if isinstance(root, SumNode):
        assert_on_simplex(root.parameters_list(), tol=tol)
        for child in root.children():
            walk_pc_invariants(child, tol=tol)
        return
    if isinstance(root, ProductNode):
        for child in root.children():
            walk_pc_invariants(child, tol=tol)
        return
    raise AssertionError(f"unsupported node type {type(root).__name__}")


def brute_force_inner_product(circ1: CircuitNode, circ2: CircuitNode) -> float:
    c1 = Circuit(circ1)
    c2 = Circuit(circ2)
    scope = sorted(circ1.scope_as_list())
    assert scope == sorted(circ2.scope_as_list())
    cards = var_cardinalities(circ1)
    total = 0.0
    for assignment in enumerate_assignments(scope, cards):
        total += c1.likelihood(assignment) * c2.likelihood(assignment)
    return total


def nw_coupling_dense(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n, m = len(p), len(q)
    plan = np.zeros((n, m), dtype=np.float64)
    i = j = 0
    p_rem, q_rem = p[0], q[0]
    eps = 1e-12
    while i < n and j < m:
        flow = min(p_rem, q_rem)
        plan[i, j] = flow
        p_rem -= flow
        q_rem -= flow
        if p_rem <= eps:
            i += 1
            if i < n:
                p_rem = p[i]
        if q_rem <= eps:
            j += 1
            if j < m:
                q_rem = q[j]
    return plan


def brute_force_cw_leaf(
    p: Sequence[float],
    q: Sequence[float],
    *,
    metric_p: float = 1.0,
    scale_factor: float = 1.0,
) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n, m = len(p), len(q)
    d = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            d[i, j] = abs(i - j) ** metric_p / scale_factor
    return float(np.sum(nw_coupling_dense(p, q) * d))


def make_categorical(
    node_id: int, scope_var: int, probabilities: Sequence[float]
) -> CategoricalInputNode:
    return CategoricalInputNode(
        id=node_id, scope_var=scope_var, probabilities=list(probabilities)
    )


def make_sum(
    node_id: int,
    scope_var: int,
    prob_list: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    id_base: int | None = None,
) -> SumNode:
    base = id_base if id_base is not None else node_id * 10
    leaves = [
        make_categorical(base + i, scope_var, probs)
        for i, probs in enumerate(prob_list)
    ]
    return SumNode(id=node_id, children=leaves, parameters=list(weights))


def make_product(
    node_id: int,
    var_probs: Sequence[tuple[int, Sequence[float]]],
    *,
    id_base: int | None = None,
) -> ProductNode:
    base = id_base if id_base is not None else node_id * 10
    leaves = [
        make_categorical(base + i, var, probs)
        for i, (var, probs) in enumerate(var_probs)
    ]
    return ProductNode(id=node_id, children=leaves)


def sum_mixture_marginal(
    prob_list: Sequence[Sequence[float]], weights: Sequence[float]
) -> np.ndarray:
    p = np.asarray(prob_list, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return w @ p
