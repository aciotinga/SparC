"""Optimization utilities for circuit parameters living on probability simplices.

Both sum-node weights and categorical PMFs are constrained to the simplex, so a
plain gradient step has to be projected back. This module provides
:func:`simplex_step`, :func:`apply_grads`, and :class:`MLETrainer`.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

from sparc.circuit import Circuit
from sparc.nodes import (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    ProductNode,
    SumNode,
)

PROB_FLOOR = 1e-20

GradLike = Union["object", Dict]  # GradBundle or a mapping


def iter_nodes(root: CircuitNode) -> Iterator[CircuitNode]:
    """Yield every node in the DAG once (id-deduplicated)."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        yield node
        if isinstance(node, (SumNode, ProductNode)):
            stack.extend(node.children())


def _project_tangent(x: np.ndarray, prob_floor: float) -> List[float]:
    """Tangent-space step result -> clip to a floor and renormalize."""
    x = np.clip(x, prob_floor, None)
    return (x / x.sum()).tolist()


def _project_euclidean(x: np.ndarray) -> List[float]:
    """Exact Euclidean projection of ``x`` onto the probability simplex."""
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(x - theta, 0.0).tolist()


def simplex_step(
    params: Sequence[float],
    grad: Sequence[float],
    lr: float,
    *,
    ascent: bool = False,
    method: str = "tangent",
    prob_floor: float = PROB_FLOOR,
) -> List[float]:
    """One projected-gradient step keeping ``params`` on the probability simplex.

    Args:
        params: Current probability vector (must sum to 1).
        grad: Gradient direction.
        lr: Step size.
        ascent: If ``True``, move along ``+grad`` (maximize); otherwise
            along ``-grad`` (minimize).
        method: ``"tangent"`` projects the gradient onto the simplex tangent,
            steps, then clips and renormalizes; ``"euclidean"`` steps then
            applies exact Euclidean projection onto the simplex.
        prob_floor: Minimum probability mass per entry after projection.

    Returns:
        Updated probability vector on the simplex.

    Raises:
        ValueError: If ``method`` is not ``"tangent"`` or ``"euclidean"``.
    """
    p = np.asarray(params, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    sign = 1.0 if ascent else -1.0
    if method == "tangent":
        g = g - g.mean()
        return _project_tangent(p + sign * lr * g, prob_floor)
    if method == "euclidean":
        return _project_euclidean(p + sign * lr * g)
    raise ValueError(f"unknown method {method!r}; use 'tangent' or 'euclidean'")


def _grad_dicts(grads: GradLike):
    """Accept a GradBundle or a ``(sum_grads, cat_grads)`` pair / mapping."""
    if hasattr(grads, "sum_grads") and hasattr(grads, "cat_grads"):
        return grads.sum_grads, grads.cat_grads
    if isinstance(grads, tuple) and len(grads) == 2:
        return grads
    raise TypeError("grads must be a GradBundle or a (sum_grads, cat_grads) tuple")


def apply_grads(
    circuit: Union[Circuit, CircuitNode],
    grads: GradLike,
    lr: float,
    *,
    ascent: bool = False,
    method: str = "tangent",
    prob_floor: float = PROB_FLOOR,
) -> None:
    """Apply one :func:`simplex_step` to every sum / categorical node in place.

    Only nodes whose ``id`` appears in the gradient dicts are updated.

    Args:
        circuit: :class:`~sparc.circuit.Circuit` or root :class:`~sparc.nodes.CircuitNode`.
        grads: :class:`~sparc.grad.GradBundle` or ``(sum_grads, cat_grads)`` tuple.
        lr: Step size passed to :func:`simplex_step`.
        ascent: If ``True``, ascend along the gradient; otherwise descend.
        method: Simplex projection method (``"tangent"`` or ``"euclidean"``).
        prob_floor: Minimum probability per entry after projection.
    """
    root = circuit.root if isinstance(circuit, Circuit) else circuit
    sum_grads, cat_grads = _grad_dicts(grads)
    for node in iter_nodes(root):
        nid = int(node.id)
        if isinstance(node, SumNode) and nid in sum_grads:
            node.set_parameters_list(
                simplex_step(
                    node.parameters_list(), sum_grads[nid], lr,
                    ascent=ascent, method=method, prob_floor=prob_floor,
                )
            )
        elif (
            isinstance(node, (CategoricalInputNode, BernoulliInputNode))
            and nid in cat_grads
        ):
            node.set_probabilities_list(
                simplex_step(
                    node.probabilities_list(), cat_grads[nid], lr,
                    ascent=ascent, method=method, prob_floor=prob_floor,
                )
            )


def global_grad_norm(grads: GradLike) -> float:
    """L2 norm over all sum and categorical gradient entries."""
    sum_grads, cat_grads = _grad_dicts(grads)
    sq = 0.0
    for d in (sum_grads, cat_grads):
        for v in d.values():
            a = np.asarray(v, dtype=np.float64)
            sq += float(a @ a)
    return sq ** 0.5


class MLETrainer:
    """Maximum-likelihood trainer via projected gradient ascent.

    Optimizes sum-node weights and categorical/Bernoulli leaf parameters in
    place using :func:`apply_grads` on the mean log-likelihood gradient.

    Args:
        circuit: Circuit whose parameters are optimized in place.
        lr: Learning rate for each projected gradient step.
        method: Simplex projection method (``"tangent"`` or ``"euclidean"``).
        prob_floor: Minimum probability per entry after each step.
    """

    def __init__(
        self,
        circuit: Circuit,
        lr: float = 1e-2,
        *,
        method: str = "tangent",
        prob_floor: float = PROB_FLOOR,
    ):
        self.circuit = circuit
        self.lr = lr
        self.method = method
        self.prob_floor = prob_floor

    def step(self, dataset: np.ndarray) -> float:
        """Run one ascent step over ``dataset``.

        Args:
            dataset: 2-D integer array ``(n_samples, n_columns)`` of complete
                assignments.

        Returns:
            Mean log-likelihood before the parameter update.
        """
        mean_ll, grads = self.circuit.mean_log_likelihood_and_grad(dataset)
        apply_grads(
            self.circuit, grads, self.lr,
            ascent=True, method=self.method, prob_floor=self.prob_floor,
        )
        return mean_ll

    def fit(
        self,
        dataset: np.ndarray,
        *,
        epochs: int = 100,
        callback=None,
    ) -> List[float]:
        """Run ``epochs`` projected gradient ascent steps.

        Args:
            dataset: 2-D integer array ``(n_samples, n_columns)`` of complete
                assignments.
            epochs: Number of optimization steps.
            callback: Optional ``callback(epoch, mean_ll)`` invoked after each
                step with the pre-update mean log-likelihood.

        Returns:
            List of mean log-likelihoods recorded before each step.
        """
        data = np.ascontiguousarray(dataset, dtype=np.int32)
        history: List[float] = []
        for epoch in range(epochs):
            mean_ll = self.step(data)
            history.append(mean_ll)
            if callback is not None:
                callback(epoch, mean_ll)
        return history
