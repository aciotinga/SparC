"""Emit SparcOp dispatch-table C glue from a CompiledCircuit codegen snapshot."""

from __future__ import annotations

from typing import Any

from sparc.deep_compile.tape import TapeLayout

NODE_SUM = 0
NODE_PRODUCT = 1
NODE_INPUT = 2

SPARC_OP_LEAF_BIN = 0
SPARC_OP_LEAF_TBL = 1
SPARC_OP_PROD_LIN = 2
SPARC_OP_PROD_LOG = 3
SPARC_OP_SUM_LIN = 4
SPARC_OP_SUM_LOG = 5

SPECIALIZE_THRESHOLD = 4096
ROW_STACK_NODE_LIMIT = 8192


def _build_op_lists(
    snap: dict[str, Any],
    layout: TapeLayout,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[int, int]]]:
    """Return (log_ops, lin_ops, leaf_var_pairs)."""
    kinds = snap["kinds"]
    n_nodes = layout.n_nodes
    leaf_var_pairs: list[tuple[int, int]] = []
    var_to_leaf: dict[int, int] = {}
    log_ops: list[dict[str, Any]] = []
    lin_ops: list[dict[str, Any]] = []

    for n in range(n_nodes):
        if kinds[n] != NODE_INPUT:
            continue
        var = layout.leaf_var[n]
        if var not in var_to_leaf:
            var_to_leaf[var] = len(leaf_var_pairs)
            leaf_var_pairs.append((len(leaf_var_pairs), var))

    for n in range(n_nodes):
        kind = kinds[n]
        if kind == NODE_INPUT:
            var = layout.leaf_var[n]
            leaf_idx = var_to_leaf[var]
            card = int(snap["leaf_card"][n])
            op_kind = SPARC_OP_LEAF_BIN if card == 2 else SPARC_OP_LEAF_TBL
            entry = {
                "kind": op_kind,
                "node": n,
                "leaf_idx": leaf_idx,
                "pmf_off": layout.leaf_pmf_off[n],
                "leaf_card": card,
            }
            log_ops.append(entry)
            lin_ops.append(dict(entry))
        elif kind == NODE_PRODUCT:
            base = {
                "node": n,
                "leaf_idx": -1,
                "pmf_off": 0,
                "leaf_card": 0,
            }
            log_ops.append({**base, "kind": SPARC_OP_PROD_LOG})
            lin_ops.append({**base, "kind": SPARC_OP_PROD_LIN})
        elif kind == NODE_SUM:
            base = {
                "node": n,
                "leaf_idx": -1,
                "pmf_off": 0,
                "leaf_card": 0,
            }
            log_ops.append({**base, "kind": SPARC_OP_SUM_LOG})
            lin_ops.append({**base, "kind": SPARC_OP_SUM_LIN})
        else:
            raise ValueError(f"unknown node kind {kind}")

    return log_ops, lin_ops, leaf_var_pairs


def _emit_op_entry(op: dict[str, Any]) -> str:
    return (
        f"  {{ {op['kind']}, {op['node']}, {op['leaf_idx']}, "
        f"{op['pmf_off']}, {op['leaf_card']} }}"
    )


def _emit_csr_arrays(snap: dict[str, Any], layout: TapeLayout) -> list[str]:
    n_nodes = layout.n_nodes
    n_edges = int(snap["child_off"][n_nodes])
    child_off = ", ".join(str(int(x)) for x in snap["child_off"])
    children = ", ".join(str(int(x)) for x in snap["children_flat"])
    return [
        f"#define SPARC_N_EDGES {n_edges}",
        f"#define SPARC_SUM_W_BASE {layout.n_leaf_pmf}",
        f"static const int16_t sparc_child_off[{n_nodes + 1}] = {{ {child_off} }};",
        f"static const int16_t sparc_children_flat[{n_edges}] = {{ {children} }};",
        "",
    ]


def _eval_row_args() -> str:
    return (
        "        SPARC_N_NODES, SPARC_ROOT_INDEX,\n"
        "        sparc_child_off, sparc_children_flat, SPARC_SUM_W_BASE,\n"
        "        workspace, out, 1, 0"
    )


def _eval_batch_args() -> str:
    return (
        "        SPARC_N_NODES, SPARC_ROOT_INDEX,\n"
        "        sparc_child_off, sparc_children_flat, SPARC_SUM_W_BASE,\n"
        "        workspace, out, tile, parallel"
    )


def _emit_specialized_tile_fn(
    fn_name: str,
    log_space: int,
    ops_table: str,
    n_ops: int,
) -> list[str]:
    """Emit unrolled per-op tile evaluator (no outer op loop)."""
    lines = [
        f"static void {fn_name}(",
        "    double* tape,",
        "    const int32_t* leaf_ev,",
        "    int32_t leaf_ev_stride,",
        "    int32_t r0,",
        "    int32_t rn,",
        "    int32_t ws_stride,",
        "    double* workspace",
        ") {",
    ]
    for i in range(n_ops):
        lines.append(
            f"  sparc_eval_one_op({log_space}, &{ops_table}[{i}], tape, leaf_ev,"
            f" leaf_ev_stride, r0, rn, ws_stride, sparc_child_off, sparc_children_flat,"
            f" SPARC_SUM_W_BASE, workspace);"
        )
    lines.append("}")
    lines.append("")
    return lines


def emit_c_source(
    snap: dict[str, Any],
    *,
    parallel: bool = True,
    tile: int = 128,
) -> str:
    """Generate glue .c with SparcOp table and dispatch entry points."""
    layout = TapeLayout.from_snapshot(snap)
    log_ops, lin_ops, leaf_var_pairs = _build_op_lists(snap, layout)

    n_leaf = len(leaf_var_pairs)
    n_log_ops = len(log_ops)
    n_lin_ops = len(lin_ops)
    n_nodes = layout.n_nodes
    par_flag = 1 if parallel else 0
    specialized = n_nodes >= SPECIALIZE_THRESHOLD
    use_heap_row_ws = n_nodes > ROW_STACK_NODE_LIMIT

    parts = [
        "/* Auto-generated by SparC deep_compile. Do not edit. */",
        '#include "sparc_deep_rt.h"',
        "",
        "#include <stdlib.h>",
        "",
    ]
    if parallel:
        parts.extend(
            [
                "#ifdef _OPENMP",
                "#include <omp.h>",
                "#endif",
                "",
            ]
        )
    parts.extend(
        [
        "#ifdef _MSC_VER",
        "#define SPARC_EXPORT __declspec(dllexport)",
        "#else",
        "#define SPARC_EXPORT",
        "#endif",
        "",
        f"#define SPARC_N_NODES {n_nodes}",
        f"#define SPARC_ROOT_INDEX {layout.root_index}",
        f"#define SPARC_N_LEAF {n_leaf}",
        f"#define SPARC_TILE_DEFAULT {tile}",
        f"#define SPARC_PARALLEL_DEFAULT {par_flag}",
        f"#define SPARC_SPECIALIZED {1 if specialized else 0}",
        "",
        ]
    )
    parts.extend(_emit_csr_arrays(snap, layout))
    parts.extend(
        [
            f"static const SparcOp sparc_ops_log[{n_log_ops}] = {{",
        ]
    )
    parts.extend(_emit_op_entry(op) + "," for op in log_ops)
    parts.append("};")
    parts.append("")
    parts.append(f"static const SparcOp sparc_ops_lin[{n_lin_ops}] = {{")
    parts.extend(_emit_op_entry(op) + "," for op in lin_ops)
    parts.append("};")
    parts.append("")

    if n_leaf > 0:
        leaf_vars = ", ".join(str(var) for _, var in leaf_var_pairs)
        parts.append(f"static const int32_t sparc_leaf_vars[{n_leaf}] = {{ {leaf_vars} }};")
        parts.append("")
        parts.append("static void sparc_fill_leaf_ev_row(const int32_t* ev, int32_t* leaf_ev) {")
        for leaf_idx, var in leaf_var_pairs:
            parts.append(f"  leaf_ev[{leaf_idx}] = ev[{var}];")
        parts.append("}")
        parts.append("")

    if specialized:
        parts.extend(_emit_specialized_tile_fn(
            "sparc_eval_tile_log_special", 1, "sparc_ops_log", n_log_ops
        ))
        parts.extend(_emit_specialized_tile_fn(
            "sparc_eval_tile_lin_special", 0, "sparc_ops_lin", n_lin_ops
        ))
        parts.extend(
            [
                "static void sparc_eval_specialized_batch(",
                "    int log_space,",
                "    double* tape,",
                "    const int32_t* leaf_ev,",
                "    int32_t leaf_ev_stride,",
                "    int32_t n_rows,",
                "    double* workspace,",
                "    double* out,",
                "    int32_t tile,",
                "    int32_t parallel",
                ") {",
                "  int32_t r0, rn, lr;",
                "  if (tile < 1) tile = 128;",
                "#ifdef _OPENMP",
                "  if (parallel) {",
                "#pragma omp parallel",
                "    {",
                "      int32_t tid = omp_get_thread_num();",
                "      double* ws_tile = workspace + (size_t)tid * (size_t)SPARC_N_NODES * (size_t)tile;",
                "#pragma omp for schedule(static)",
                "      for (r0 = 0; r0 < n_rows; r0 += tile) {",
                "        rn = n_rows - r0;",
                "        if (rn > tile) rn = tile;",
                "        if (log_space) {",
                "          sparc_eval_tile_log_special(",
                "              tape, leaf_ev, leaf_ev_stride, r0, rn, tile, ws_tile);",
                "        } else {",
                "          sparc_eval_tile_lin_special(",
                "              tape, leaf_ev, leaf_ev_stride, r0, rn, tile, ws_tile);",
                "        }",
                "        for (lr = 0; lr < rn; ++lr) {",
                "          out[r0 + lr] = ws_tile[(size_t)SPARC_ROOT_INDEX * (size_t)tile + (size_t)lr];",
                "        }",
                "      }",
                "    }",
                "    return;",
                "  }",
                "#else",
                "  (void)parallel;",
                "#endif",
                "  for (r0 = 0; r0 < n_rows; r0 += tile) {",
                "    rn = n_rows - r0;",
                "    if (rn > tile) rn = tile;",
                "    if (log_space) {",
                "      sparc_eval_tile_log_special(",
                "          tape, leaf_ev, leaf_ev_stride, r0, rn, tile, workspace);",
                "    } else {",
                "      sparc_eval_tile_lin_special(",
                "          tape, leaf_ev, leaf_ev_stride, r0, rn, tile, workspace);",
                "    }",
                "    for (lr = 0; lr < rn; ++lr) {",
                "      out[r0 + lr] = workspace[(size_t)SPARC_ROOT_INDEX * (size_t)tile + (size_t)lr];",
                "    }",
                "  }",
                "}",
                "",
            ]
        )

    eval_row_args = _eval_row_args()

    if use_heap_row_ws:
        row_ws_alloc = [
            "  double* workspace = (double*)malloc((size_t)SPARC_N_NODES * sizeof(double));",
            "  double out;",
            "  if (workspace == NULL) { return 0.0; }",
        ]
        row_ws_free = "  free(workspace);"
        row_ws_body_lin = [
            *row_ws_alloc,
            "  sparc_eval_row(0, tape, ev, workspace, &out);",
            row_ws_free,
            "  return out;",
        ]
        row_ws_body_log = [
            *row_ws_alloc,
            "  sparc_eval_row(1, tape, ev, workspace, &out);",
            row_ws_free,
            "  return out;",
        ]
    else:
        row_ws_body_lin = [
            "  double workspace[SPARC_N_NODES];",
            "  double out;",
            "  sparc_eval_row(0, tape, ev, workspace, &out);",
            "  return out;",
        ]
        row_ws_body_log = [
            "  double workspace[SPARC_N_NODES];",
            "  double out;",
            "  sparc_eval_row(1, tape, ev, workspace, &out);",
            "  return out;",
        ]

    parts.extend(
        [
            "static void sparc_eval_row(",
            "    int log_space,",
            "    double* tape,",
            "    const int32_t* ev,",
            "    double* workspace,",
            "    double* out",
            ") {",
            "  int32_t leaf_ev[SPARC_N_LEAF];",
            "  if (SPARC_N_LEAF > 0) {",
            "    sparc_fill_leaf_ev_row(ev, leaf_ev);",
            "  }",
            "  if (log_space) {",
            "    sparc_dispatch()->eval_log_batch(",
            "        tape, leaf_ev, 1, 1,",
            "        sparc_ops_log,",
            "        (int32_t)(sizeof(sparc_ops_log) / sizeof(sparc_ops_log[0])),",
            eval_row_args,
            "    );",
            "  } else {",
            "    sparc_dispatch()->eval_lin_batch(",
            "        tape, leaf_ev, 1, 1,",
            "        sparc_ops_lin,",
            "        (int32_t)(sizeof(sparc_ops_lin) / sizeof(sparc_ops_lin[0])),",
            eval_row_args,
            "    );",
            "  }",
            "}",
            "",
            "SPARC_EXPORT",
            "double sparc_likelihood_row(double* tape, const int32_t* ev) {",
            *row_ws_body_lin,
            "}",
            "",
            "SPARC_EXPORT",
            "double sparc_log_likelihood_row(double* tape, const int32_t* ev) {",
            *row_ws_body_log,
            "}",
            "",
            "static void sparc_eval_batch_impl(",
            "    int log_space,",
            "    double* tape,",
            "    const int32_t* leaf_ev,",
            "    int32_t leaf_ev_stride,",
            "    int32_t n_rows,",
            "    double* workspace,",
            "    double* out,",
            "    int32_t tile,",
            "    int32_t parallel",
            ") {",
        ]
    )

    if specialized:
        parts.extend(
            [
                "  sparc_eval_specialized_batch(",
                "      log_space, tape, leaf_ev, leaf_ev_stride, n_rows,",
                "      workspace, out, tile, parallel",
                "  );",
            ]
        )
    else:
        eval_batch_args = _eval_batch_args()
        parts.extend(
            [
                "  if (log_space) {",
                "    sparc_dispatch()->eval_log_batch(",
                "        tape, leaf_ev, leaf_ev_stride, n_rows,",
                "        sparc_ops_log,",
                "        (int32_t)(sizeof(sparc_ops_log) / sizeof(sparc_ops_log[0])),",
                eval_batch_args,
                "    );",
                "  } else {",
                "    sparc_dispatch()->eval_lin_batch(",
                "        tape, leaf_ev, leaf_ev_stride, n_rows,",
                "        sparc_ops_lin,",
                "        (int32_t)(sizeof(sparc_ops_lin) / sizeof(sparc_ops_lin[0])),",
                eval_batch_args,
                "    );",
                "  }",
            ]
        )

    parts.extend(
        [
            "}",
            "",
            "SPARC_EXPORT",
            "void sparc_likelihood_batch(",
            "    double* tape,",
            "    const int32_t* leaf_ev,",
            "    int32_t leaf_ev_stride,",
            "    int32_t n_rows,",
            "    double* workspace,",
            "    double* out,",
            "    int32_t tile,",
            "    int32_t parallel",
            ") {",
            "  sparc_eval_batch_impl(",
            "      0, tape, leaf_ev, leaf_ev_stride, n_rows, workspace, out, tile, parallel",
            "  );",
            "}",
            "",
            "SPARC_EXPORT",
            "void sparc_log_likelihood_batch(",
            "    double* tape,",
            "    const int32_t* leaf_ev,",
            "    int32_t leaf_ev_stride,",
            "    int32_t n_rows,",
            "    double* workspace,",
            "    double* out,",
            "    int32_t tile,",
            "    int32_t parallel",
            ") {",
            "  sparc_eval_batch_impl(",
            "      1, tape, leaf_ev, leaf_ev_stride, n_rows, workspace, out, tile, parallel",
            "  );",
            "}",
            "",
        ]
    )

    return "\n".join(parts)


def leaf_var_order(snap: dict[str, Any]) -> list[int]:
    """Return variable ids in leaf_ev row order."""
    layout = TapeLayout.from_snapshot(snap)
    _, _, leaf_var_pairs = _build_op_lists(snap, layout)
    return [var for _, var in sorted(leaf_var_pairs, key=lambda p: p[0])]
