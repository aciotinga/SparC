"""SparC: fast, modular sparse probabilistic circuits in Cython (CPU-only).

SparC implements probabilistic circuits (PCs) with typed Cython evaluation,
differentiable Wasserstein-type queries, and composable structure builders.
Import the high-level :class:`~sparc.circuit.Circuit` wrapper and query
functions from this package; see subpackages :mod:`sparc.optim`,
:mod:`sparc.builders`, :mod:`sparc.structures`, and :mod:`sparc.io` for
training, random construction, built-in structures, and serialization.

Full documentation: build locally with ``pip install -e ".[docs]" && mkdocs serve``.
"""

from sparc.circuit import Circuit
from sparc.eval import CompiledCircuit, likelihood, log_likelihood, sample
from sparc.grad import GradBundle, mean_log_likelihood_and_grad
from sparc.io import CircuitSerializer, load_learned_pc
from sparc.metrics import GroundMetric, PNormMetric
from sparc.nodes import (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    DiscreteLogisticInputNode,
    Evidence,
    FiniteDiscreteInputNode,
    IndicatorInputNode,
    InputNode,
    LiteralInputNode,
    ProductNode,
    RandomState,
    SumNode,
)
from sparc.queries import (
    cw_distance,
    cw_distance_and_grad,
    exp_query,
    exp_query_and_grad,
    expected_squared_distance,
    expected_squared_distance_and_grad,
    gcw_coupling_circuit,
    gcw_crossterm,
    gcw_crossterm_and_grad,
    log_exp_query,
    log_exp_query_and_grad,
)

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
