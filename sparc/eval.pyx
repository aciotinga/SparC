# distutils: language = c++
# cython: boundscheck=False, wraparound=False
"""Evaluation queries: object-graph path for live circuits, flat path via
:class:`~sparc._graph.CompiledCircuit`.
"""

import time

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp

from sparc._graph cimport (
    CompiledCircuit,
    SP_MISSING_EVIDENCE,
    _coerce_likelihood_data_with_missing,
    _flat_sample_node,
    _leaf_column_map,
    _max_var_from_scope,
)
from sparc._mathutils cimport sp_logsumexp, sp_safe_log
from sparc.nodes cimport (
    CircuitNode,
    Evidence,
    InputNode,
    NODE_INPUT,
    NODE_PRODUCT,
    NODE_SUM,
    ProductNode,
    RandomState,
    SumNode,
)


# Re-export CompiledCircuit from the graph module.
from sparc._graph import CompiledCircuit


cdef Evidence _evidence_from_row(
    const int[:] row,
    Py_ssize_t n_cols,
    int max_var,
    unordered_set[int]& scope,
    object var_to_col,
    bint allow_missing,
) except *:
    cdef Evidence ev = Evidence.__new__(Evidence)
    cdef int v
    cdef int col
    cdef int val
    cdef int card
    ev.init_dense(max_var + 1)
    if var_to_col is None:
        if n_cols <= max_var:
            raise ValueError(
                f"assignment array length {n_cols} is shorter than required "
                f"width {max_var + 1}"
            )
        for v in sorted(scope):
            val = row[v]
            if val == SP_MISSING_EVIDENCE:
                if not allow_missing:
                    raise ValueError(f"outcome value must be non-negative, got {val}")
                continue
            if val < 0:
                raise ValueError(f"outcome value must be non-negative, got {val}")
            ev.set_var(v, val)
    else:
        for v in sorted(scope):
            col = int(var_to_col[v])
            if col < 0 or col >= n_cols:
                raise ValueError(
                    f"variable {v} maps to column {col} out of range "
                    f"[0, {n_cols})"
                )
            val = row[col]
            if val == SP_MISSING_EVIDENCE:
                if not allow_missing:
                    raise ValueError(f"outcome value must be non-negative, got {val}")
                continue
            if val < 0:
                raise ValueError(f"outcome value must be non-negative, got {val}")
            ev.set_var(v, val)
    return ev


# --- Object-graph queries -----------------------------------------------------

cdef class _QueryContext:
    cdef Evidence evidence
    cdef unordered_map[size_t, double] memo
    cdef bint log_space

    cdef bint memo_get(self, size_t node_id, double* out) noexcept:
        cdef unordered_map[size_t, double].iterator it = self.memo.find(node_id)
        if it == self.memo.end():
            return False
        out[0] = self.memo[node_id]
        return True


cdef double _eval_sum(SumNode node, _QueryContext ctx) except *:
    cdef size_t i
    cdef size_t n = node.num_children()
    cdef double result
    cdef vector[double] terms
    if ctx.log_space:
        terms.resize(n)
        for i in range(n):
            terms[i] = sp_safe_log(node.parameter_at(i)) + _eval(node.child_at(i), ctx)
        return sp_logsumexp(terms)
    result = 0.0
    for i in range(n):
        result += node.parameter_at(i) * _eval(node.child_at(i), ctx)
    return result


cdef double _eval_product(ProductNode node, _QueryContext ctx) except *:
    cdef size_t i
    cdef size_t n = node.num_children()
    cdef double result
    if ctx.log_space:
        result = 0.0
        for i in range(n):
            result += _eval(node.child_at(i), ctx)
        return result
    result = 1.0
    for i in range(n):
        result *= _eval(node.child_at(i), ctx)
    return result


cdef double _eval_impl(CircuitNode node, _QueryContext ctx) except *:
    cdef double p
    if node.node_kind == NODE_INPUT:
        p = (<InputNode>node).prob_c(ctx.evidence)
        if ctx.log_space:
            return sp_safe_log(p)
        return p
    if node.node_kind == NODE_PRODUCT:
        return _eval_product(<ProductNode>node, ctx)
    if node.node_kind == NODE_SUM:
        return _eval_sum(<SumNode>node, ctx)
    raise TypeError(f"unsupported node type for query: {type(node).__name__}")


cdef double _eval(CircuitNode node, _QueryContext ctx) except *:
    cdef double cached
    cdef double result
    if ctx.memo_get(node.id, &cached):
        return cached
    result = _eval_impl(node, ctx)
    ctx.memo[node.id] = result
    return result


cdef double _run_query_evidence(CircuitNode root, Evidence evidence, bint log_space) except *:
    cdef _QueryContext ctx = _QueryContext()
    ctx.evidence = evidence
    ctx.log_space = log_space
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    return _eval(root, ctx)


cdef object _likelihood_impl(
    CircuitNode root, object data, object var_to_col, bint log_space
):
    cdef cnp.ndarray arr
    cdef bint allow_missing
    arr, allow_missing = _coerce_likelihood_data_with_missing(data, True)
    cdef int max_var = _max_var_from_scope(root.scope)
    cdef Py_ssize_t n_rows
    cdef Py_ssize_t r
    cdef Evidence ev
    cdef object out
    cdef double val
    if arr.ndim == 1:
        ev = _evidence_from_row(
            arr, arr.shape[0], max_var, root.scope, var_to_col, allow_missing
        )
        return _run_query_evidence(root, ev, log_space)
    n_rows = arr.shape[0]
    out = np.empty(n_rows, dtype=np.float64)
    for r in range(n_rows):
        ev = _evidence_from_row(
            arr[r], arr.shape[1], max_var, root.scope, var_to_col, allow_missing
        )
        val = _run_query_evidence(root, ev, log_space)
        out[r] = val
    return out


cpdef object likelihood(CircuitNode root, object data, object var_to_col=None):
    """Evaluate likelihood for a 1-D or 2-D assignment array.

    Integer arrays require every scoped variable to be observed. Floating arrays
    may use ``numpy.nan`` to mark missing variables; those variables are
    marginalized out (summed over all outcomes).
    """
    return _likelihood_impl(root, data, var_to_col, False)


cpdef object log_likelihood(CircuitNode root, object data, object var_to_col=None):
    """Evaluate log-likelihood for a 1-D or 2-D assignment array.

    Integer arrays require every scoped variable to be observed. Floating arrays
    may use ``numpy.nan`` to mark missing variables; those variables are
    marginalized out (summed over all outcomes).
    """
    return _likelihood_impl(root, data, var_to_col, True)


# --- Object-graph sampling ----------------------------------------------------

cdef void _sample_node(CircuitNode node, RandomState rng, int* out) except *:
    cdef size_t i
    cdef size_t idx
    cdef double u
    cdef double cum
    cdef SumNode s
    cdef ProductNode prod
    if node.node_kind == NODE_INPUT:
        (<InputNode>node).sample_into_c(rng, out)
    elif node.node_kind == NODE_PRODUCT:
        prod = <ProductNode>node
        for i in range(prod.num_children()):
            _sample_node(prod.child_at(i), rng, out)
    elif node.node_kind == NODE_SUM:
        s = <SumNode>node
        u = rng.next_double()
        cum = 0.0
        idx = s.num_children() - 1
        for i in range(s.num_children()):
            cum += s.parameter_at(i)
            if u < cum:
                idx = i
                break
        _sample_node(s.child_at(idx), rng, out)
    else:
        raise TypeError(f"unsupported node type for sampling: {type(node).__name__}")


cpdef cnp.ndarray sample(CircuitNode root, Py_ssize_t n_samples, object seed=None):
    """Draw ancestral samples as a 2-D integer array."""
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    cdef unsigned long long rng_seed
    if seed is None:
        rng_seed = <unsigned long long>time.time_ns()
    else:
        rng_seed = <unsigned long long>int(seed)
    cdef RandomState rng = RandomState(rng_seed)
    cdef int max_var = _max_var_from_scope(root.scope)
    cdef size_t width = <size_t>(max_var + 1)
    cdef object out = np.full((n_samples, width), -1, dtype=np.int32)
    cdef int[:, ::1] out_view = out
    cdef Py_ssize_t i
    for i in range(n_samples):
        _sample_node(root, rng, &out_view[i, 0])
    return out
