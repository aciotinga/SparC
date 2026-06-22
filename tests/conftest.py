"""Pytest configuration: dual object-graph / compiled execution for circuit tests."""

from __future__ import annotations

import pytest

from tests.dual_path import CircuitBackend, ensure_exports_loaded, install_compiled_routing

# Modules that are compiled-only, parity-only, or do not exercise circuit queries.
NO_DUAL_PATH_MODULES = frozenset(
    {
        "test_compiled_eval",
        "test_compiled_queries",
        "test_refresh_parameters",
        "test_nodes",
        "test_nodes_extended",
        "test_solvers",
        "test_metrics",
    }
)

_ORIGINALS: dict | None = None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "no_dual_path: opt out of object-graph vs compiled parametrization",
    )


def _skip_dual_path_item(item: pytest.Item) -> bool:
    if item.get_closest_marker("no_dual_path") is not None:
        return True
    module = item.module.__name__.rsplit(".", 1)[-1]
    return module in NO_DUAL_PATH_MODULES


def _skip_dual_path_module(module) -> bool:
    name = module.__name__.rsplit(".", 1)[-1]
    return name in NO_DUAL_PATH_MODULES


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "use_compiled" not in metafunc.fixturenames:
        return
    module = metafunc.module
    if _skip_dual_path_module(module):
        metafunc.parametrize("use_compiled", [False], ids=["object"])
        return
    metafunc.parametrize(
        "use_compiled",
        [False, True],
        ids=["object", "compiled"],
        indirect=False,
    )


@pytest.fixture
def use_compiled(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture
def circuit_backend(use_compiled: bool) -> CircuitBackend:
    return CircuitBackend(use_compiled)


@pytest.fixture(autouse=True)
def _route_compiled_queries(
    request: pytest.FixtureRequest,
    use_compiled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global _ORIGINALS
    if _skip_dual_path_item(request.node):
        return
    if _ORIGINALS is None:
        _ORIGINALS = ensure_exports_loaded()
    if use_compiled:
        install_compiled_routing(monkeypatch, _ORIGINALS)
