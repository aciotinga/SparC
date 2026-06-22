"""SparC: fast, modular sparse probabilistic circuits in Cython (CPU-only).

SparC implements probabilistic circuits (PCs) with typed Cython evaluation,
differentiable Wasserstein-type queries, and composable structure builders.
Import the high-level :class:`~sparc.circuit.Circuit` wrapper and query
functions from this package; see subpackages :mod:`sparc.optim`,
:mod:`sparc.builders`, :mod:`sparc.structures`, and :mod:`sparc.io` for
training, random construction, built-in structures, and serialization.

Full documentation: build locally with ``pip install -e ".[docs]" && mkdocs serve``.
"""

from __future__ import annotations

import importlib

__version__ = "0.4.1"

__all__ = [
    "Circuit",
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
    "Evidence",
    "RandomState",
    "likelihood",
    "log_likelihood",
    "sample",
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
]

_LAZY_EXPORTS = {
    "Circuit": ("sparc.circuit", "Circuit"),
    "CompiledCircuit": ("sparc._graph", "CompiledCircuit"),
    "likelihood": ("sparc.eval", "likelihood"),
    "log_likelihood": ("sparc.eval", "log_likelihood"),
    "sample": ("sparc.eval", "sample"),
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
    return sorted(set(globals()) | set(__all__))
