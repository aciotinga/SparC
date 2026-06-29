"""Route circuit query entry points through the compiled fast path when requested.

Used by ``conftest.py`` to run the same tests against object-graph and
``CompiledCircuit`` implementations without rewriting every call site.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Callable

import sparc
from sparc._graph import CompiledCircuit
from sparc.nodes import CircuitNode

QUERY_EXPORTS = (
    "likelihood",
    "log_likelihood",
    "sample",
    "mean_log_likelihood_and_grad",
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
)

def ensure_exports_loaded() -> dict[str, Callable[..., Any]]:
    """Force lazy sparc exports to load and return the object-path callables."""
    return {name: getattr(sparc, name) for name in QUERY_EXPORTS}

def as_compiled(obj: Any) -> CompiledCircuit:
    if isinstance(obj, CompiledCircuit):
        return obj
    if isinstance(obj, CircuitNode):
        return CompiledCircuit(obj)
    raise TypeError(f"cannot compile {type(obj)!r} for the fast path")


def build_wrappers(orig: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    def likelihood(circuit: Any, data: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit).likelihood(data, *args, **kwargs)

    def log_likelihood(circuit: Any, data: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit).log_likelihood(data, *args, **kwargs)

    def sample(circuit: Any, n_samples: int, seed: Any = None, *args: Any, **kwargs: Any):
        return as_compiled(circuit).sample(n_samples, seed, *args, **kwargs)

    def mean_log_likelihood_and_grad(circuit: Any, dataset: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit).mean_log_likelihood_and_grad(dataset, *args, **kwargs)

    def cw_distance(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any) -> float:
        return as_compiled(circuit1).cw_distance(as_compiled(circuit2), *args, **kwargs)

    def cw_distance_and_grad(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit1).cw_distance_and_grad(
            as_compiled(circuit2), *args, **kwargs
        )

    def expected_squared_distance(circuit: Any, *args: Any, **kwargs: Any) -> float:
        return as_compiled(circuit).expected_squared_distance(*args, **kwargs)

    def expected_squared_distance_and_grad(circuit: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit).expected_squared_distance_and_grad(*args, **kwargs)

    def exp_query(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any) -> float:
        return as_compiled(circuit1).exp_query(as_compiled(circuit2), *args, **kwargs)

    def exp_query_and_grad(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit1).exp_query_and_grad(as_compiled(circuit2), *args, **kwargs)

    def log_exp_query(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any) -> float:
        return as_compiled(circuit1).log_exp_query(as_compiled(circuit2), *args, **kwargs)

    def log_exp_query_and_grad(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit1).log_exp_query_and_grad(
            as_compiled(circuit2), *args, **kwargs
        )

    def gcw_crossterm(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any) -> float:
        return as_compiled(circuit1).gcw_crossterm(as_compiled(circuit2), *args, **kwargs)

    def gcw_crossterm_and_grad(circuit1: Any, circuit2: Any, *args: Any, **kwargs: Any):
        return as_compiled(circuit1).gcw_crossterm_and_grad(
            as_compiled(circuit2), *args, **kwargs
        )

    return {
        "likelihood": likelihood,
        "log_likelihood": log_likelihood,
        "sample": sample,
        "mean_log_likelihood_and_grad": mean_log_likelihood_and_grad,
        "cw_distance": cw_distance,
        "cw_distance_and_grad": cw_distance_and_grad,
        "expected_squared_distance": expected_squared_distance,
        "expected_squared_distance_and_grad": expected_squared_distance_and_grad,
        "exp_query": exp_query,
        "exp_query_and_grad": exp_query_and_grad,
        "log_exp_query": log_exp_query,
        "log_exp_query_and_grad": log_exp_query_and_grad,
        "gcw_crossterm": gcw_crossterm,
        "gcw_crossterm_and_grad": gcw_crossterm_and_grad,
    }


def _patch_module_bindings(
    monkeypatch: Any,
    module: Any,
    orig: dict[str, Callable[..., Any]],
    wrappers: dict[str, Callable[..., Any]],
) -> None:
    for name, original in orig.items():
        if getattr(module, name, None) is original:
            monkeypatch.setattr(module, name, wrappers[name], raising=False)


def install_compiled_routing(monkeypatch: Any, orig: dict[str, Callable[..., Any]]) -> None:
    wrappers = build_wrappers(orig)
    for name, fn in wrappers.items():
        monkeypatch.setattr(sparc, name, fn, raising=False)
    for mod_name, module in list(sys.modules.items()):
        if mod_name.startswith("tests."):
            _patch_module_bindings(monkeypatch, module, orig, wrappers)


class CircuitBackend:
    """Explicit helper for tests that need to branch on the active path."""

    def __init__(self, compiled: bool):
        self.compiled = compiled

    def prepare(self, obj: Any) -> Any:
        if not self.compiled:
            if isinstance(obj, (CircuitNode, CompiledCircuit)):
                return obj
            return obj
        return as_compiled(obj)

    def prepare_pair(self, a: Any, b: Any) -> tuple[Any, Any]:
        if not self.compiled:
            return a, b
        return as_compiled(a), as_compiled(b)
