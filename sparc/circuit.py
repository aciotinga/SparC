from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from sparc.nodes import (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    DiscreteLogisticInputNode,
    IndicatorInputNode,
    LiteralInputNode,
    ProductNode,
    SumNode,
)


def _clone_node(node, memo):
    """Deep-copy a circuit DAG in memory, preserving id-based node sharing."""
    cached = memo.get(node.id)
    if cached is not None:
        return cached
    if isinstance(node, CategoricalInputNode):
        new = CategoricalInputNode(
            node.id, node.scope_as_list()[0], node.probabilities_list()
        )
    elif isinstance(node, BernoulliInputNode):
        new = BernoulliInputNode(node.id, node.scope_as_list()[0], node.p())
    elif isinstance(node, LiteralInputNode):
        new = LiteralInputNode(node.id, node.scope_as_list()[0], node.value_at())
    elif isinstance(node, IndicatorInputNode):
        new = IndicatorInputNode(
            node.id, node.scope_as_list()[0], node.value_at(), node.num_categories()
        )
    elif isinstance(node, DiscreteLogisticInputNode):
        new = DiscreteLogisticInputNode(
            node.id,
            node.scope_as_list()[0],
            node.mu_value(),
            node.s_value(),
            node.num_categories(),
        )
    elif isinstance(node, SumNode):
        children = [_clone_node(c, memo) for c in node.children()]
        new = SumNode(node.id, children, node.parameters_list())
    elif isinstance(node, ProductNode):
        children = [_clone_node(c, memo) for c in node.children()]
        new = ProductNode(node.id, children)
    else:
        raise TypeError(
            f"clone() does not know how to copy {type(node).__name__}; "
            "custom leaf types should define their own cloning"
        )
    memo[node.id] = new
    return new


class Circuit:
    """Wrapper around a probabilistic circuit root node."""

    def __init__(self, root: CircuitNode):
        self.root = root
        if not root.scope_as_list():
            root.propagate_scope()

    def likelihood(self, assignment: dict[int, int]) -> float:
        from sparc.eval import likelihood

        return likelihood(self.root, assignment)

    def log_likelihood(self, assignment: dict[int, int]) -> float:
        from sparc.eval import log_likelihood

        return log_likelihood(self.root, assignment)

    def mean_log_likelihood_and_grad(self, dataset):
        """Mean LL over a dataset (list of ``{var: value}`` dicts) and its gradient.

        Returns ``(mean_ll, grads)`` where ``grads`` is a :class:`GradBundle`
        (``sum_grads`` / ``cat_grads`` keyed by ``node.id``) of the mean
        log-likelihood w.r.t. the circuit's linear parameters.
        """
        from sparc.grad import mean_log_likelihood_and_grad

        return mean_log_likelihood_and_grad(self.root, dataset)

    def sample(self, n_samples: int, seed: Optional[int] = None) -> list[dict[int, int]]:
        from sparc.eval import sample

        return sample(self.root, n_samples, seed)

    def compile(self):
        """Build a :class:`CompiledCircuit` for fast batched log-likelihood."""
        from sparc.eval import CompiledCircuit

        return CompiledCircuit(self.root)

    def batched_log_likelihood(self, data, var_to_col=None):
        """Vectorized log-likelihood over a 2-D integer dataset."""
        return self.compile().log_likelihood(data, var_to_col)

    def clone(self) -> "Circuit":
        """Return an independent deep copy (fast in-memory DAG copy)."""
        return Circuit(_clone_node(self.root, {}))

    def save(
        self,
        path: Union[str, Path],
        *,
        indent: int = 2,
        encoding: str = "utf-8",
    ) -> None:
        from sparc.io.serializer import CircuitSerializer

        CircuitSerializer.save(self.root, path, indent=indent, encoding=encoding)

    @classmethod
    def load(cls, path: Union[str, Path], *, encoding: str = "utf-8") -> "Circuit":
        from sparc.io.serializer import CircuitSerializer

        root = CircuitSerializer.load(path, encoding=encoding)
        return cls(root)
