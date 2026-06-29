"""Emit unrolled SIMD op-schedule calls for ultra deep compile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sparc.deep_compile.tape import TapeLayout

NODE_SUM = 0
NODE_PRODUCT = 1
NODE_INPUT = 2

ULTRA_LEAF_BIN = 0
ULTRA_LEAF_TBL = 1
ULTRA_PRODUCT = 2
ULTRA_SUM = 3
ULTRA_COPY = 4


@dataclass
class GraphPlan:
    """Emit-time graph analysis for a frozen circuit snapshot."""

    snap: dict[str, Any]
    layout: TapeLayout
    leaf_key_to_slot: dict[tuple[int, int], int] = field(default_factory=dict)
    skip_nodes: set[int] = field(default_factory=set)
    copy_from: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> GraphPlan:
        layout = TapeLayout.from_snapshot(snap)
        plan = cls(snap=snap, layout=layout)
        plan._analyze_leaves()
        plan._analyze_unary_products()
        return plan

    def _analyze_leaves(self) -> None:
        kinds = self.snap["kinds"]
        for n in range(self.layout.n_nodes):
            if kinds[n] != NODE_INPUT:
                continue
            var = int(self.snap["leaf_var"][n])
            pmf_off = int(self.layout.leaf_pmf_off[n])
            key = (var, pmf_off)
            if key in self.leaf_key_to_slot:
                self.skip_nodes.add(n)
                self.copy_from[n] = self.leaf_key_to_slot[key]
            else:
                self.leaf_key_to_slot[key] = n

    def _analyze_unary_products(self) -> None:
        kinds = self.snap["kinds"]
        child_off = self.snap["child_off"]
        children_flat = self.snap["children_flat"]
        for n in range(self.layout.n_nodes):
            if kinds[n] != NODE_PRODUCT:
                continue
            start = int(child_off[n])
            stop = int(child_off[n + 1])
            if stop - start == 1:
                self.skip_nodes.add(n)
                self.copy_from[n] = int(children_flat[start])


def _children(snap: dict[str, Any], node: int) -> list[int]:
    start = int(snap["child_off"][node])
    stop = int(snap["child_off"][node + 1])
    return [int(snap["children_flat"][i]) for i in range(start, stop)]


def _sum_weights(snap: dict[str, Any], layout: TapeLayout, node: int) -> list[int]:
    start = int(snap["child_off"][node])
    stop = int(snap["child_off"][node + 1])
    base = layout.sum_w_base
    return [base + i for i in range(start, stop)]


def _int_array_literal(values: list[int]) -> str:
    if not values:
        return "(const int32_t[]){}"
    inner = ", ".join(str(v) for v in values)
    return f"(const int32_t[]){{{inner}}}"


def emit_op_call(
    snap: dict[str, Any],
    layout: TapeLayout,
    plan: GraphPlan,
    node: int,
) -> list[str]:
    """Return C source lines for one unrolled op call."""
    if node in plan.skip_nodes:
        src = plan.copy_from[node]
        return [
            f"  /* node {node}: copy from {src} */",
            f"  sparc_ultra_ws_copy({node}, {src}, r0, rn, ws_stride, workspace);",
        ]

    kind = snap["kinds"][node]
    if kind == NODE_INPUT:
        var = int(snap["leaf_var"][node])
        pmf_off = int(layout.leaf_pmf_off[node])
        card = int(snap["leaf_card"][node])
        if card == 2:
            return [
                f"  /* node {node}: leaf_bin var={var} */",
                "  sparc_ultra_leaf_bin(log_space, "
                f"{node}, {var}, {pmf_off}, tape, data, data_stride, col_for_var, "
                "r0, rn, ws_stride, workspace);",
            ]
        return [
            f"  /* node {node}: leaf_tbl var={var} card={card} */",
            "  sparc_ultra_leaf_tbl(log_space, "
            f"{node}, {var}, {pmf_off}, {card}, tape, data, data_stride, col_for_var, "
            "r0, rn, ws_stride, workspace);",
        ]

    if kind == NODE_PRODUCT:
        children = _children(snap, node)
        ch_lit = _int_array_literal(children)
        return [
            f"  /* node {node}: product fanin={len(children)} */",
            "  sparc_ultra_product(log_space, "
            f"{node}, r0, rn, ws_stride, workspace, {len(children)}, {ch_lit});",
        ]

    if kind == NODE_SUM:
        children = _children(snap, node)
        weights = _sum_weights(snap, layout, node)
        w_lit = _int_array_literal(weights)
        c_lit = _int_array_literal(children)
        return [
            f"  /* node {node}: sum fanin={len(children)} */",
            "  sparc_ultra_sum(log_space, "
            f"{node}, tape, r0, rn, ws_stride, workspace, {len(children)}, {w_lit}, {c_lit});",
        ]

    raise ValueError(f"unknown node kind {kind}")


def emit_schedule_body(
    snap: dict[str, Any],
    layout: TapeLayout,
    plan: GraphPlan,
) -> list[str]:
    """Emit unrolled op calls for all nodes in topological order."""
    lines: list[str] = []
    for n in range(layout.n_nodes):
        lines.extend(emit_op_call(snap, layout, plan, n))
    return lines
