from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

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

    def likelihood(self, data: np.ndarray, var_to_col: Optional[dict[int, int]] = None):
        """Evaluate likelihood for a 1-D or 2-D assignment array.

        Args:
            data: 1-D array (index ``i`` = value for variable ``i``) or 2-D
                batch ``(n_samples, n_columns)``. Integer arrays require every
                scoped variable to be observed. Floating arrays may use
                ``numpy.nan`` to mark missing variables, which are marginalized
                out.
            var_to_col: Optional mapping from variable index to column index
                for 2-D batches (default: column ``i`` holds variable ``i``).

        Returns:
            A scalar if ``data`` is 1-D, otherwise an array of shape
            ``(n_samples,)``.
        """
        from sparc.eval import likelihood

        return likelihood(self.root, data, var_to_col)

    def log_likelihood(self, data: np.ndarray, var_to_col: Optional[dict[int, int]] = None):
        """Evaluate log-likelihood for a 1-D or 2-D assignment array.

        Args:
            data: 1-D array (index ``i`` = value for variable ``i``) or 2-D
                batch ``(n_samples, n_columns)``. Integer arrays require every
                scoped variable to be observed. Floating arrays may use
                ``numpy.nan`` to mark missing variables, which are marginalized
                out.
            var_to_col: Optional mapping from variable index to column index
                for 2-D batches (default: column ``i`` holds variable ``i``).

        Returns:
            A scalar if ``data`` is 1-D, otherwise an array of shape
            ``(n_samples,)``.
        """
        from sparc.eval import log_likelihood

        return log_likelihood(self.root, data, var_to_col)

    def mean_log_likelihood_and_grad(
        self,
        dataset: np.ndarray,
        var_to_col: Optional[dict[int, int]] = None,
    ):
        """Mean log-likelihood over a dataset and its gradient.

        Args:
            dataset: 2-D assignment array ``(n_samples, n_columns)``. Integer
                arrays require every scoped variable to be observed. Floating
                arrays may use ``numpy.nan`` to mark missing variables, which
                are marginalized out.
            var_to_col: Optional mapping from variable index to column index
                (default: column ``i`` holds variable ``i``).

        Returns:
            A pair ``(mean_ll, grads)`` where ``mean_ll`` is the average
            log-likelihood and ``grads`` is a :class:`~sparc.grad.GradBundle`
            (``sum_grads`` / ``cat_grads`` keyed by ``node.id``) of the mean
            log-likelihood w.r.t. the circuit's linear parameters.
        """
        from sparc.grad import mean_log_likelihood_and_grad

        return mean_log_likelihood_and_grad(self.root, dataset, var_to_col)

    def sample(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Draw ancestral samples from the circuit.

        Args:
            n_samples: Number of independent samples to draw.
            seed: Optional RNG seed for reproducibility.

        Returns:
            Integer array of shape ``(n_samples, max_var + 1)`` where index
            ``i`` holds the sampled value for variable ``i`` (``-1`` if not in
            scope).
        """
        from sparc.eval import sample

        return sample(self.root, n_samples, seed)

    def compile(self):
        """Build a :class:`~sparc._graph.CompiledCircuit` for fast inference.

        Compile once when topology is fixed. Call
        :meth:`~sparc.eval.CompiledCircuit.refresh_parameters` after parameter
        updates during training.
        """
        from sparc._graph import CompiledCircuit

        return CompiledCircuit(self.root)

    def deep_compile(
        self,
        path: Union[str, Path, None] = None,
        *,
        compiler: Optional[str] = None,
        flags: Sequence[str] | None = None,
        parallel: bool = True,
        tile: int = 128,
        isa: Optional[str] = None,
        performance: str = "max",
        compile_opt: str = "fast",
        mode: str = "ultra",
        simd: Optional[str] = None,
        use_cache: bool = True,
    ):
        """Deep-compile the circuit to unrolled native C for fast inference.

        Returns a :class:`~sparc.deep_compile.DeepCompiledCircuit` with
        ``likelihood`` / ``log_likelihood`` methods. Parameters live in a
        mutable tape; call :meth:`refresh_parameters` after weight/PMF updates.

        When *path* is omitted, compiled artifacts are stored in a managed
        temporary directory and deleted by :meth:`DeepCompiledCircuit.close`
        (also called automatically when the object is garbage-collected or
        when used as a context manager).

        Requires a C compiler (gcc/clang) at deep-compile time.

        Args:
            path: Optional output path stem (e.g. ``/tmp/model`` →
                ``model.c`` + ``.so``). Omit to use a managed temp directory.
            compiler: Optional compiler executable (auto-detected if omitted).
            flags: Compiler flags (default from *compile_opt*: ``-O2`` for ``fast``).
            parallel: Use OpenMP row-block parallelism (default ``True``).
            tile: Row tile size for parallel batch evaluation (default ``128``).
            isa: ``None`` auto-detects best host ISA (``avx512`` / ``avx2`` /
                ``scalar``); manual override allowed.
            performance: ``"max"`` disables L3 workspace throttling (default).
            compile_opt: ``"fast"`` (default) or ``"max"`` (``-O3 -march=native``).
            mode: ``"ultra"`` (default) unrolled SIMD op schedule; ``"compat"``
                uses legacy SparcOp dispatch tables.
            use_cache: Reuse cached ``.dll`` / ``.so`` when topology matches.
            simd: Deprecated alias for *isa*.

        Returns:
            :class:`~sparc.deep_compile.DeepCompiledCircuit`
        """
        from sparc.deep_compile import deep_compile_circuit

        return deep_compile_circuit(
            self.root,
            path,
            compiler=compiler,
            flags=flags,
            parallel=parallel,
            tile=tile,
            isa=isa,
            performance=performance,  # type: ignore[arg-type]
            compile_opt=compile_opt,
            mode=mode,
            simd=simd,
            use_cache=use_cache,
        )

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
