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
    """High-level wrapper around a probabilistic circuit root node.

    A circuit is a directed acyclic graph of sum (mixture), product
    (factorization), and input (leaf) nodes. This class provides inference,
    sampling, batched evaluation, gradient computation, cloning, and JSON
    serialization on top of the underlying :class:`~sparc.nodes.CircuitNode`
    DAG.

    Args:
        root: Root node of the circuit. If its scope is empty,
            :meth:`~sparc.nodes.CircuitNode.propagate_scope` is called
            automatically.
    """

    def __init__(self, root: CircuitNode):
        self.root = root
        if not root.scope_as_list():
            root.propagate_scope()

    def likelihood(self, assignment: dict[int, int]) -> float:
        """Evaluate the probability of a single complete assignment.

        Args:
            assignment: Mapping from variable index to observed value.

        Returns:
            The likelihood :math:`P(\\mathbf{x})`.
        """
        from sparc.eval import likelihood

        return likelihood(self.root, assignment)

    def log_likelihood(self, assignment: dict[int, int]) -> float:
        """Evaluate the log-probability of a single complete assignment.

        Args:
            assignment: Mapping from variable index to observed value.

        Returns:
            The log-likelihood :math:`\\log P(\\mathbf{x})`.
        """
        from sparc.eval import log_likelihood

        return log_likelihood(self.root, assignment)

    def mean_log_likelihood_and_grad(self, dataset):
        """Mean log-likelihood over a dataset and its gradient.

        Args:
            dataset: Iterable of ``{var: value}`` dicts, one per datapoint.

        Returns:
            A pair ``(mean_ll, grads)`` where ``mean_ll`` is the average
            log-likelihood and ``grads`` is a :class:`~sparc.grad.GradBundle`
            (``sum_grads`` / ``cat_grads`` keyed by ``node.id``) of the mean
            log-likelihood w.r.t. the circuit's linear parameters.
        """
        from sparc.grad import mean_log_likelihood_and_grad

        return mean_log_likelihood_and_grad(self.root, dataset)

    def sample(self, n_samples: int, seed: Optional[int] = None) -> list[dict[int, int]]:
        """Draw ancestral samples from the circuit.

        Args:
            n_samples: Number of independent samples to draw.
            seed: Optional RNG seed for reproducibility.

        Returns:
            A list of ``{var: value}`` assignment dicts, one per sample.
        """
        from sparc.eval import sample

        return sample(self.root, n_samples, seed)

    def compile(self):
        """Build a :class:`~sparc.eval.CompiledCircuit` for fast batched evaluation.

        Returns:
            A flattened, vectorized evaluator for the same circuit.
        """
        from sparc.eval import CompiledCircuit

        return CompiledCircuit(self.root)

    def batched_log_likelihood(self, data, var_to_col=None):
        """Vectorized log-likelihood over a 2-D integer dataset.

        Args:
            data: Integer array of shape ``(n_samples, n_vars)`` where each
                column holds values for one variable.
            var_to_col: Optional mapping from variable index to column index.
                When omitted, columns are assumed to follow variable order
                ``0, 1, ...``.

        Returns:
            1-D array of log-likelihoods, one per row of ``data``.
        """
        return self.compile().log_likelihood(data, var_to_col)

    def clone(self) -> "Circuit":
        """Return an independent deep copy of the circuit.

        Shared subgraphs in the original DAG remain shared in the copy
        (deduplicated by ``node.id``).

        Returns:
            A new :class:`Circuit` wrapping a deep-copied root node.
        """
        return Circuit(_clone_node(self.root, {}))

    def save(
        self,
        path: Union[str, Path],
        *,
        indent: int = 2,
        encoding: str = "utf-8",
    ) -> None:
        """Serialize the circuit to a JSON file (``gcw-circuit-v1`` format).

        Args:
            path: Output file path.
            indent: JSON indentation level.
            encoding: File encoding.
        """
        from sparc.io.serializer import CircuitSerializer

        CircuitSerializer.save(self.root, path, indent=indent, encoding=encoding)

    @classmethod
    def load(cls, path: Union[str, Path], *, encoding: str = "utf-8") -> "Circuit":
        """Load a circuit from a JSON file (``gcw-circuit-v1`` format).

        Args:
            path: Input file path.
            encoding: File encoding.

        Returns:
            A :class:`Circuit` wrapping the deserialized root node.
        """
        from sparc.io.serializer import CircuitSerializer

        root = CircuitSerializer.load(path, encoding=encoding)
        return cls(root)
