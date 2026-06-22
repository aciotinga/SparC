# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
# cython: boundscheck=False, wraparound=False
"""Evaluation queries: object-graph path for live circuits, flat path via
:class:`~sparc._graph.CompiledCircuit`.
"""

import time

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.math cimport exp, INFINITY, isfinite, log

import numpy as np

from sparc._graph cimport CompiledCircuit, graph_safe_log
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


cpdef object _compiled_query_impl(CompiledCircuit g, object assignment, bint log_space):
    return _compiled_query(g, assignment, log_space)


cpdef list _compiled_sample_impl(CompiledCircuit g, Py_ssize_t n_samples, object seed=None):
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    cdef unsigned long long rng_seed
    if seed is None:
        rng_seed = <unsigned long long>time.time_ns()
    else:
        rng_seed = <unsigned long long>int(seed)
    cdef RandomState rng = RandomState(rng_seed)
    cdef size_t width = <size_t>(g.max_var + 1)
    cdef vector[int] buf
    cdef Py_ssize_t i
    cdef size_t base
    cdef int v
    cdef dict row
    cdef list results = []
    cdef size_t root_index = g.root_index
    if n_samples > 0:
        buf.assign(<size_t>n_samples * width, -1)
        with nogil:
            for i in range(n_samples):
                _flat_sample_node(g, root_index, rng, buf.data() + <size_t>i * width)
        for i in range(n_samples):
            base = <size_t>i * width
            row = {}
            for v in g.variables:
                if buf[base + <size_t>v] >= 0:
                    row[v] = buf[base + <size_t>v]
            results.append(row)
    return results


# --- Object-graph single-datapoint queries ------------------------------------

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


cdef double _run_query(CircuitNode root, object assignment, bint log_space) except *:
    cdef _QueryContext ctx = _QueryContext()
    ctx.evidence = Evidence(assignment)
    ctx.log_space = log_space
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    ctx.evidence.require_vars(root.scope)
    return _eval(root, ctx)


cpdef double likelihood(CircuitNode root, object assignment) except *:
    """Evaluate the probability of a single complete assignment (object-graph)."""
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    return _run_query(root, assignment, False)


cpdef double log_likelihood(CircuitNode root, object assignment) except *:
    """Evaluate log-probability of a single assignment (object-graph)."""
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    return _run_query(root, assignment, True)


# --- Object-graph sampling ----------------------------------------------------

cdef void _sample_node(CircuitNode node, RandomState rng, dict out) except *:
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


cpdef list sample(CircuitNode root, Py_ssize_t n_samples, object seed=None):
    """Draw ancestral samples from a circuit (object-graph)."""
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
    cdef list results = []
    cdef Py_ssize_t i
    cdef dict row
    for i in range(n_samples):
        row = {}
        _sample_node(root, rng, row)
        results.append(row)
    return results


# --- CompiledCircuit flat helpers (used by grad and CompiledCircuit methods) --

cdef void _build_evidence_vector(
    CompiledCircuit g, object assignment, vector[int]& ev
) except *:
    cdef int max_var = g.max_var
    cdef object key
    cdef object value
    cdef int var
    cdef int outcome
    cdef size_t n
    cdef int v
    cdef int val
    cdef int card
    ev.assign(max_var + 1, -1)
    if isinstance(assignment, np.ndarray):
        arr = np.asarray(assignment, dtype=np.int32).ravel()
        if arr.size <= max_var:
            raise ValueError(
                f"assignment array length {arr.size} is shorter than required "
                f"width {max_var + 1}"
            )
        for var in range(max_var + 1):
            ev[var] = int(arr[var])
    else:
        for key, value in assignment.items():
            var = int(key)
            outcome = int(value)
            if var < 0:
                raise ValueError(f"variable index must be non-negative, got {var}")
            if outcome < 0:
                raise ValueError(f"outcome value must be non-negative, got {outcome}")
            if var <= max_var:
                ev[var] = outcome
    for v in g.variables:
        if ev[v] < 0:
            raise ValueError(f"missing evidence for variable {v}")
    for n in range(g.n_nodes):
        if g.kinds[n] == NODE_INPUT:
            v = g.leaf_var[n]
            val = ev[v]
            card = g.leaf_card[n]
            if val < 0 or val >= card:
                raise ValueError(
                    f"evidence for variable {v}: outcome {val} out of range "
                    f"[0, {card})"
                )


cdef double _flat_eval(
    CompiledCircuit g, const int* ev, bint log_space, double[::1] val
) noexcept nogil:
    cdef size_t n
    cdef size_t k
    cdef size_t start
    cdef size_t stop
    cdef int kind
    cdef int var
    cdef int value
    cdef double acc
    cdef double max_log
    cdef double sum_exp
    cdef double term
    for n in range(g.n_nodes):
        kind = g.kinds[n]
        if kind == NODE_INPUT:
            var = g.leaf_var[n]
            value = ev[var]
            if log_space:
                val[n] = g.leaf_logpmf_flat[g.leaf_pmf_off[n] + value]
            else:
                val[n] = g.leaf_pmf_flat[g.leaf_pmf_off[n] + value]
        elif kind == NODE_PRODUCT:
            start = g.child_off[n]
            stop = g.child_off[n + 1]
            if log_space:
                acc = 0.0
                for k in range(start, stop):
                    acc += val[g.children_flat[k]]
            else:
                acc = 1.0
                for k in range(start, stop):
                    acc *= val[g.children_flat[k]]
            val[n] = acc
        else:
            start = g.child_off[n]
            stop = g.child_off[n + 1]
            if log_space:
                max_log = -INFINITY
                for k in range(start, stop):
                    term = g.sum_logw_flat[k] + val[g.children_flat[k]]
                    if term > max_log:
                        max_log = term
                if not isfinite(max_log) or max_log == -INFINITY:
                    val[n] = -INFINITY
                else:
                    sum_exp = 0.0
                    for k in range(start, stop):
                        term = g.sum_logw_flat[k] + val[g.children_flat[k]]
                        if isfinite(term) and term > -INFINITY:
                            sum_exp += exp(term - max_log)
                    if sum_exp <= 0.0:
                        val[n] = -INFINITY
                    else:
                        val[n] = max_log + log(sum_exp)
            else:
                acc = 0.0
                for k in range(start, stop):
                    acc += g.sum_w_flat[k] * val[g.children_flat[k]]
                val[n] = acc
    return val[g.root_index]


cdef double _compiled_query(CompiledCircuit g, object assignment, bint log_space) except *:
    cdef vector[int] ev
    _build_evidence_vector(g, assignment, ev)
    cdef object val_arr = np.empty(g.n_nodes, dtype=np.float64)
    cdef double[::1] val = val_arr
    cdef double result
    with nogil:
        result = _flat_eval(g, ev.data(), log_space, val)
    return result


cdef void _flat_sample_node(
    CompiledCircuit g, size_t n, RandomState rng, int* out
) noexcept nogil:
    cdef int kind = g.kinds[n]
    cdef size_t start
    cdef size_t stop
    cdef size_t k
    cdef size_t idx
    cdef size_t off
    cdef double u
    cdef double cum
    cdef int var
    cdef int value
    cdef int card
    if kind == NODE_INPUT:
        var = g.leaf_var[n]
        card = g.leaf_card[n]
        off = g.leaf_pmf_off[n]
        u = rng.next_double()
        cum = 0.0
        value = card - 1
        for k in range(<size_t>card):
            cum += g.leaf_pmf_flat[off + k]
            if u < cum:
                value = <int>k
                break
        out[var] = value
    elif kind == NODE_PRODUCT:
        start = g.child_off[n]
        stop = g.child_off[n + 1]
        for k in range(start, stop):
            _flat_sample_node(g, g.children_flat[k], rng, out)
    else:
        start = g.child_off[n]
        stop = g.child_off[n + 1]
        u = rng.next_double()
        cum = 0.0
        idx = g.children_flat[stop - 1]
        for k in range(start, stop):
            cum += g.sum_w_flat[k]
            if u < cum:
                idx = g.children_flat[k]
                break
        _flat_sample_node(g, idx, rng, out)
