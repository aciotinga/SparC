"""Differentiable queries over pairs of probabilistic circuits.

This subpackage provides Wasserstein-type distances and expectations used
for optimization, distributionally robust training, and circuit comparison:

- :func:`cw_distance` / :func:`cw_distance_and_grad` -- Circuit-Wasserstein
  :math:`W_p^p`.
- :func:`gcw_crossterm` / :func:`gcw_crossterm_and_grad` -- Gromov-Circuit-
  Wasserstein cross-term.
- :func:`gcw_coupling_circuit` -- materialize the GCW coupling as a circuit.
- :func:`exp_query` / :func:`log_exp_query` and gradient variants --
  :math:`E_Q[P(X)]` and its log.
- :func:`expected_squared_distance` and gradient variant -- single-circuit ESD.

Pairwise queries require structurally compatible circuits (matching scopes and
decompositions). CW and GCW gradient variants return gradients w.r.t.
**circuit2** only; expectation queries return gradients for both circuits.
"""

from __future__ import annotations

import importlib

__all__ = [
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
    "cw_distance": ("sparc.queries.cw", "cw_distance"),
    "cw_distance_and_grad": ("sparc.queries.cw", "cw_distance_and_grad"),
    "expected_squared_distance": ("sparc.queries.esd", "expected_squared_distance"),
    "expected_squared_distance_and_grad": (
        "sparc.queries.esd",
        "expected_squared_distance_and_grad",
    ),
    "exp_query": ("sparc.queries.expectation", "exp_query"),
    "exp_query_and_grad": ("sparc.queries.expectation", "exp_query_and_grad"),
    "log_exp_query": ("sparc.queries.expectation", "log_exp_query"),
    "log_exp_query_and_grad": (
        "sparc.queries.expectation",
        "log_exp_query_and_grad",
    ),
    "gcw_crossterm": ("sparc.queries.gcw", "gcw_crossterm"),
    "gcw_crossterm_and_grad": ("sparc.queries.gcw", "gcw_crossterm_and_grad"),
    "gcw_coupling_circuit": ("sparc.queries.gcw", "gcw_coupling_circuit"),
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
