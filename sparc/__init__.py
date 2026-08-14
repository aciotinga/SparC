"""SparC: fast, modular sparse probabilistic circuits in Cython (CPU-only).

SparC implements probabilistic circuits (PCs) with typed Cython evaluation,
hard-forward differentiable sampling, differentiable Wasserstein-type
queries, and composable structure builders.
Import node types and query functions from this package; see subpackages
:mod:`sparc.optim`, :mod:`sparc.builders`, :mod:`sparc.structures`, and
:mod:`sparc.io` for training, random construction, built-in structures, and
serialization.

Install from PyPI with ``pip install sparc-pc`` (import name: ``sparc``).
Full documentation: https://sparc-docs.readthedocs.io
"""

from __future__ import annotations

import importlib
import sys
import types
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sparc-pc")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "CircuitNode",
    "SumNode",
    "ProductNode",
    "InputNode",
    "FiniteDiscreteInputNode",
    "CategoricalInputNode",
    "BernoulliInputNode",
    "IndicatorInputNode",
    "LiteralInputNode",
    "DiscreteLogisticInputNode",
    "ContinuousInputNode",
    "GaussianInputNode",
    "Evidence",
    "RandomState",
    "likelihood",
    "log_likelihood",
    "sample",
    "DifferentiableSample",
    "CompiledCircuit",
    "mean_log_likelihood_and_grad",
    "GradBundle",
    "GroundMetric",
    "PNormMetric",
    "CircuitSerializer",
    "load_learned_pc",
    "cw_distance",
    "cw_distance_and_grad",
    "expected_squared_distance",
    "expected_squared_distance_and_grad",
    "exp_query",
    "exp_query_and_grad",
    "log_exp_query",
    "log_exp_query_and_grad",
    "gcw_crossterm",
    "gcw_crossterm_and_grad",
    "gcw_coupling_circuit",
    "set_num_threads",
    "get_num_threads",
]

_LAZY_EXPORTS = {
    "CompiledCircuit": ("sparc._graph", "CompiledCircuit"),
    "likelihood": ("sparc.eval", "likelihood"),
    "log_likelihood": ("sparc.eval", "log_likelihood"),
    "sample": ("sparc.eval", "sample"),
    "DifferentiableSample": ("sparc.sampling", "DifferentiableSample"),
    "GradBundle": ("sparc.grad", "GradBundle"),
    "mean_log_likelihood_and_grad": ("sparc.grad", "mean_log_likelihood_and_grad"),
    "CircuitSerializer": ("sparc.io", "CircuitSerializer"),
    "load_learned_pc": ("sparc.io", "load_learned_pc"),
    "GroundMetric": ("sparc.metrics", "GroundMetric"),
    "PNormMetric": ("sparc.metrics", "PNormMetric"),
    "CircuitNode": ("sparc.nodes", "CircuitNode"),
    "SumNode": ("sparc.nodes", "SumNode"),
    "ProductNode": ("sparc.nodes", "ProductNode"),
    "InputNode": ("sparc.nodes", "InputNode"),
    "FiniteDiscreteInputNode": ("sparc.nodes", "FiniteDiscreteInputNode"),
    "CategoricalInputNode": ("sparc.nodes", "CategoricalInputNode"),
    "BernoulliInputNode": ("sparc.nodes", "BernoulliInputNode"),
    "IndicatorInputNode": ("sparc.nodes", "IndicatorInputNode"),
    "LiteralInputNode": ("sparc.nodes", "LiteralInputNode"),
    "DiscreteLogisticInputNode": ("sparc.nodes", "DiscreteLogisticInputNode"),
    "ContinuousInputNode": ("sparc.nodes", "ContinuousInputNode"),
    "GaussianInputNode": ("sparc.nodes", "GaussianInputNode"),
    "Evidence": ("sparc.nodes", "Evidence"),
    "RandomState": ("sparc.nodes", "RandomState"),
    "cw_distance": ("sparc.queries", "cw_distance"),
    "cw_distance_and_grad": ("sparc.queries", "cw_distance_and_grad"),
    "expected_squared_distance": ("sparc.queries", "expected_squared_distance"),
    "expected_squared_distance_and_grad": (
        "sparc.queries",
        "expected_squared_distance_and_grad",
    ),
    "exp_query": ("sparc.queries", "exp_query"),
    "exp_query_and_grad": ("sparc.queries", "exp_query_and_grad"),
    "log_exp_query": ("sparc.queries", "log_exp_query"),
    "log_exp_query_and_grad": ("sparc.queries", "log_exp_query_and_grad"),
    "gcw_crossterm": ("sparc.queries", "gcw_crossterm"),
    "gcw_crossterm_and_grad": ("sparc.queries", "gcw_crossterm_and_grad"),
    "gcw_coupling_circuit": ("sparc.queries", "gcw_coupling_circuit"),
}


def set_num_threads(n: int) -> None:
    """Cap the OpenMP team used by compiled batched likelihood.

    Call once at the top of a script (or assign ``sparc.num_threads = n``).
    Does not change object-graph eval, sampling, or query kernels. ``n=1``
    forces serial compiled batches. No-op if SparC was built without OpenMP.
    """
    from sparc._graph import _omp_set_num_threads

    _omp_set_num_threads(int(n))


def get_num_threads() -> int:
    """Current OpenMP max-thread cap (always ``1`` without OpenMP)."""
    from sparc._graph import _omp_max_threads

    return int(_omp_max_threads())


class _SparCModule(types.ModuleType):
    @property
    def num_threads(self) -> int:
        return get_num_threads()

    @num_threads.setter
    def num_threads(self, n: int) -> None:
        set_num_threads(n)


sys.modules[__name__].__class__ = _SparCModule


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"num_threads"})
