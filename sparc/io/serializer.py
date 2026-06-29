"""Plaintext (JSON) serialization for a single-root circuit graph.

Shared children (DAGs) are deduplicated by object identity. On disk, circuits
use the ``gcw-circuit-v1`` JSON format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

_FORMAT = "gcw-circuit-v1"


def _unwrap_root(root: object) -> CircuitNode:
    if not isinstance(root, CircuitNode):
        raise TypeError(f"Expected CircuitNode, got {type(root)!r}")
    return root


def _iter_children(node: CircuitNode) -> List[CircuitNode]:
    if isinstance(node, (ProductNode, SumNode)):
        return node.children()
    return []


def _node_kind(node: CircuitNode) -> str:
    if isinstance(node, CategoricalInputNode):
        return "categorical"
    if isinstance(node, BernoulliInputNode):
        return "bernoulli"
    if isinstance(node, LiteralInputNode):
        return "literal"
    if isinstance(node, IndicatorInputNode):
        return "indicator"
    if isinstance(node, DiscreteLogisticInputNode):
        return "discrete_logistic"
    if isinstance(node, SumNode):
        return "sum"
    if isinstance(node, ProductNode):
        return "product"
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def _collect_postorder(root: CircuitNode) -> Tuple[List[CircuitNode], int, Dict[int, int]]:
    pyid_to_idx: Dict[int, int] = {}
    nodes: List[CircuitNode] = []

    def visit(n: CircuitNode) -> int:
        key = id(n)
        if key in pyid_to_idx:
            return pyid_to_idx[key]
        for ch in _iter_children(n):
            visit(ch)
        idx = len(nodes)
        pyid_to_idx[key] = idx
        nodes.append(n)
        return idx

    root_idx = visit(root)
    return nodes, root_idx, pyid_to_idx


def _build_record(node: CircuitNode, node_id: int, pyid_to_idx: Dict[int, int]) -> Dict[str, Any]:
    kind = _node_kind(node)
    children = [pyid_to_idx[id(ch)] for ch in _iter_children(node)]
    rec: Dict[str, Any] = {"id": node_id, "kind": kind, "children": children}
    if kind == "sum":
        rec["params"] = list(node.parameters_list())
    elif kind == "categorical":
        rec["scope"] = node.scope_as_list()
        rec["params"] = list(node.probabilities_list())
    elif kind == "bernoulli":
        rec["scope"] = node.scope_as_list()
        rec["p"] = float(node.p())
    elif kind == "literal":
        rec["scope"] = node.scope_as_list()
        rec["value"] = int(node.value_at())
    elif kind == "indicator":
        rec["scope"] = node.scope_as_list()
        rec["value"] = int(node.value_at())
        rec["num_cats"] = int(node.num_categories())
    elif kind == "discrete_logistic":
        rec["scope"] = node.scope_as_list()
        rec["mu"] = float(node.mu_value())
        rec["s"] = float(node.s_value())
        rec["num_cats"] = int(node.num_categories())
    return rec


def _validate_categorical_params(scope: List[int], params: Any) -> List[float]:
    if len(scope) != 1:
        raise ValueError(f"SparC supports single-variable categoricals only; got scope={scope!r}")
    if not isinstance(params, list):
        raise ValueError(f"categorical params must be a list, got {type(params)!r}")
    if params and isinstance(params[0], list):
        raise ValueError("SparC does not support multi-dimensional categorical PMFs")
    return [float(x) for x in params]


def _instantiate(kind: str, *, node_id: int, children, scope, params) -> CircuitNode:
    if kind == "sum":
        return SumNode(children, params, id=node_id)
    if kind == "product":
        return ProductNode(children, id=node_id)
    if kind == "categorical":
        assert scope is not None
        probs = _validate_categorical_params(scope, params)
        return CategoricalInputNode(scope[0], probs, id=node_id)
    raise ValueError(f"Unknown kind {kind!r}")


class CircuitSerializer:
    """Save and load a single-root circuit as UTF-8 JSON.

    On disk, circuits use the ``gcw-circuit-v1`` format. Shared children in
    the in-memory DAG are deduplicated by object identity during serialization.
    """

    @staticmethod
    def dumps(root: object, *, indent: Optional[int] = 2) -> str:
        """Serialize a circuit root to a JSON string.

        Args:
            root: :class:`~sparc.nodes.CircuitNode`.
            indent: JSON indentation level, or ``None`` for compact output.

        Returns:
            UTF-8 JSON string in ``gcw-circuit-v1`` format.
        """
        root_node = _unwrap_root(root)
        nodes, root_idx, pyid_to_idx = _collect_postorder(root_node)
        payload = {
            "format": _FORMAT,
            "backend": "numpy",
            "root": root_idx,
            "nodes": [_build_record(n, i, pyid_to_idx) for i, n in enumerate(nodes)],
        }
        return json.dumps(payload, indent=indent)

    @staticmethod
    def loads(text: str, *, device: Any = None) -> CircuitNode:
        """Deserialize a circuit root from a JSON string.

        Args:
            text: UTF-8 JSON in ``gcw-circuit-v1`` format.
            device: Ignored; kept for API compatibility.

        Returns:
            Root :class:`~sparc.nodes.CircuitNode` with scope propagated.

        Raises:
            ValueError: If the format, node ids, or node records are invalid.
        """
        del device  # kept for call-site compatibility
        data = json.loads(text)
        if data.get("format") != _FORMAT:
            raise ValueError(f"Expected format {_FORMAT!r}, got {data.get('format')!r}")
        root_idx = int(data["root"])
        raw_nodes: List[Dict[str, Any]] = data["nodes"]
        by_id = {int(rec["id"]): rec for rec in raw_nodes}
        max_id = max(by_id) if by_id else -1
        if sorted(by_id.keys()) != list(range(max_id + 1)):
            raise ValueError("Node ids must be contiguous from 0..N-1")
        if root_idx < 0 or root_idx > max_id:
            raise ValueError(f"Root index {root_idx} out of range 0..{max_id}")
        built: List[Optional[CircuitNode]] = [None] * (max_id + 1)

        for nid in range(max_id + 1):
            rec = by_id[nid]
            kind = rec["kind"]
            if kind == "gaussian":
                raise ValueError("Gaussian input nodes are not supported in SparC")
            child_ids = [int(x) for x in rec["children"]]
            for cid in child_ids:
                if cid >= nid:
                    raise ValueError(
                        f"Invalid edge {nid} -> {cid}: child id must be less than parent id"
                    )
                if built[cid] is None:
                    raise ValueError(f"Missing child node {cid}")
            children_objs = [built[cid] for cid in child_ids]

            if kind == "sum":
                params = rec["params"]
                if len(params) != len(children_objs):
                    raise ValueError(
                        f"Sum node {nid}: len(params)={len(params)} != len(children)={len(children_objs)}"
                    )
                built[nid] = _instantiate(kind, node_id=nid, children=children_objs, scope=None, params=params)
            elif kind == "product":
                built[nid] = _instantiate(kind, node_id=nid, children=children_objs, scope=None, params=None)
            elif kind == "categorical":
                scope = [int(x) for x in rec["scope"]]
                built[nid] = _instantiate(kind, node_id=nid, children=children_objs, scope=scope, params=rec["params"])
            elif kind == "bernoulli":
                scope = [int(x) for x in rec["scope"]]
                if len(scope) != 1:
                    raise ValueError(f"bernoulli leaf requires a single-variable scope; got {scope!r}")
                built[nid] = BernoulliInputNode(scope[0], float(rec["p"]), id=nid)
            elif kind == "literal":
                scope = [int(x) for x in rec["scope"]]
                if len(scope) != 1:
                    raise ValueError(f"literal leaf requires a single-variable scope; got {scope!r}")
                built[nid] = LiteralInputNode(scope[0], int(rec["value"]), id=nid)
            elif kind == "indicator":
                scope = [int(x) for x in rec["scope"]]
                if len(scope) != 1:
                    raise ValueError(f"indicator leaf requires a single-variable scope; got {scope!r}")
                built[nid] = IndicatorInputNode(
                    scope[0], int(rec["value"]), int(rec["num_cats"]), id=nid
                )
            elif kind == "discrete_logistic":
                scope = [int(x) for x in rec["scope"]]
                if len(scope) != 1:
                    raise ValueError(f"discrete_logistic leaf requires a single-variable scope; got {scope!r}")
                built[nid] = DiscreteLogisticInputNode(
                    scope[0],
                    float(rec["mu"]),
                    float(rec["s"]),
                    int(rec["num_cats"]),
                    id=nid,
                )
            else:
                raise ValueError(f"Unknown kind {kind!r}")

        root = built[root_idx]
        if root is None:
            raise ValueError(f"Root index {root_idx} has no node")
        root.propagate_scope()
        return root

    @staticmethod
    def save(root: object, path: Union[str, Path], *, indent: int = 2, encoding: str = "utf-8") -> None:
        """Write a circuit to a JSON file.

        Args:
            root: :class:`~sparc.nodes.CircuitNode`.
            path: Output file path.
            indent: JSON indentation level.
            encoding: File encoding.
        """
        Path(path).write_text(CircuitSerializer.dumps(root, indent=indent), encoding=encoding)

    @staticmethod
    def load(path: Union[str, Path], *, device: Any = None, encoding: str = "utf-8") -> CircuitNode:
        """Load a circuit root from a JSON file.

        Args:
            path: Input file path.
            device: Ignored; kept for API compatibility.
            encoding: File encoding.

        Returns:
            Root :class:`~sparc.nodes.CircuitNode`.
        """
        return CircuitSerializer.loads(Path(path).read_text(encoding=encoding), device=device)
