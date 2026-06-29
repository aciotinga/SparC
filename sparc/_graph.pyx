# distutils: language = c++
# cython: boundscheck=False, wraparound=False
"""Flattened circuit representation for nogil fast-path inference.

:class:`CompiledCircuit` lays the (deduplicated) DAG out in CSR arrays so
likelihood, gradient, sampling, and optimal-transport queries can sweep the
structure without Python object traversal. Created once via
:meth:`~sparc.circuit.Circuit.compile`; call :meth:`refresh_parameters` after
parameter updates during training.
"""

from cython.operator cimport dereference as deref, preincrement as inc
from libc.math cimport exp, INFINITY, isfinite, log
from libc.stdint cimport uint64_t
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

import time

import numpy as np
cimport numpy as cnp

from sparc.nodes cimport (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    DiscreteLogisticInputNode,
    FiniteDiscreteInputNode,
    IndicatorInputNode,
    InputNode,
    InternalNode,
    LiteralInputNode,
    NODE_INPUT,
    NODE_PRODUCT,
    NODE_SUM,
    ProductNode,
    RandomState,
    SumNode,
)
from sparc._mathutils cimport SP_MAX_SUM_FANIN, sp_logsumexp_ptr


cdef int SP_MISSING_EVIDENCE = -1


cdef inline double sp_graph_sigmoid(double x) noexcept nogil:
    cdef double z
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    z = exp(x)
    return z / (1.0 + z)


cdef inline double graph_safe_log(double x) noexcept nogil:
    if x > 0.0:
        return log(x)
    return -INFINITY


cdef inline uint64_t _scope_sig(CircuitNode node) noexcept:
    cdef uint64_t sig = 0
    cdef uint64_t h
    cdef unordered_set[int].iterator it = node.scope.begin()
    while it != node.scope.end():
        h = <uint64_t><unsigned int>deref(it)
        h = (h ^ (h >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
        h = (h ^ (h >> 27)) * <uint64_t>0x94d049bb133111eb
        h = h ^ (h >> 31)
        sig ^= h
        inc(it)
    return sig ^ (<uint64_t>node.scope.size() * <uint64_t>0x9e3779b97f4a7c15)


cdef inline bint _scope_eq_nodes(CircuitNode a, CircuitNode b) noexcept:
    if a.scope.size() != b.scope.size():
        return False
    cdef unordered_set[int].iterator it = a.scope.begin()
    while it != a.scope.end():
        if b.scope.find(deref(it)) == b.scope.end():
            return False
        inc(it)
    return True


cdef inline bint _scope_eq_flat(
    const vector[int]& flat_a,
    size_t off_a,
    int size_a,
    const vector[int]& flat_b,
    size_t off_b,
    int size_b,
) noexcept nogil:
    cdef int i
    if size_a != size_b:
        return False
    for i in range(size_a):
        if flat_a[off_a + i] != flat_b[off_b + i]:
            return False
    return True


cdef void match_prod_children_flat(
    CompiledCircuit g0,
    size_t n0,
    CompiledCircuit g1,
    size_t n1,
    vector[int]& row_ind,
    vector[int]& col_ind,
    str query_name,
) except *:
    """Match product children by precomputed scope metadata (nogil-safe data)."""
    cdef size_t n = g0.child_off[n0 + 1] - g0.child_off[n0]
    cdef size_t m = g1.child_off[n1 + 1] - g1.child_off[n1]
    cdef size_t i
    cdef size_t j
    cdef size_t c0
    cdef size_t c1
    cdef uint64_t sig
    cdef int matched
    cdef size_t remaining
    cdef vector[int] q_children
    cdef vector[uint64_t] q_sigs

    if n != m:
        raise ValueError(
            f"{query_name} incompatible: product nodes have different numbers "
            f"of children ({n} vs {m})"
        )
    row_ind.assign(n, 0)
    col_ind.assign(n, 0)
    q_children.resize(m)
    q_sigs.resize(m)
    for j in range(m):
        c1 = g1.children_flat[g1.child_off[n1] + j]
        q_children[j] = <int>c1
        q_sigs[j] = g1.scope_sig[c1]
    remaining = m
    for i in range(n):
        c0 = g0.children_flat[g0.child_off[n0] + i]
        sig = g0.scope_sig[c0]
        matched = -1
        for j in range(m):
            if q_sigs[j] == 0xFFFFFFFFFFFFFFFF:
                continue
            if q_sigs[j] != sig:
                continue
            c1 = <size_t>q_children[j]
            if not _scope_eq_flat(
                g0.scope_vars_flat, g0.scope_vars_off[c0], g0.scope_size[c0],
                g1.scope_vars_flat, g1.scope_vars_off[c1], g1.scope_size[c1],
            ):
                continue
            if matched >= 0:
                raise ValueError(
                    f"{query_name} incompatible: duplicate child scope among "
                    "Q product children"
                )
            matched = <int>j
        if matched < 0:
            raise ValueError(
                f"{query_name} incompatible: no matching Q child for P child "
                f"scope signature {sig}"
            )
        row_ind[i] = <int>i
        col_ind[i] = matched
        q_sigs[matched] = 0xFFFFFFFFFFFFFFFF
        remaining -= 1
    if remaining != 0:
        raise ValueError(
            f"{query_name} incompatible: unmatched Q product children remain"
        )


cdef int _max_var_from_scope(unordered_set[int]& scope) noexcept:
    cdef int max_var = -1
    cdef int v
    cdef unordered_set[int].iterator it = scope.begin()
    while it != scope.end():
        v = deref(it)
        if v > max_var:
            max_var = v
        inc(it)
    return max_var


cdef cnp.ndarray _coerce_data_array(object data, bint allow_1d) except *:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    cdef cnp.ndarray arr = np.ascontiguousarray(data, dtype=np.int32)
    if allow_1d and arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr
    if allow_1d:
        raise ValueError("data must be 1-D or 2-D (n_samples, n_columns)")
    raise ValueError("data must be 2-D (n_samples, n_columns)")


cdef cnp.ndarray _coerce_likelihood_data(object data, bint allow_1d) except *:
    cdef bint allow_missing
    cdef cnp.ndarray arr
    arr, allow_missing = _coerce_likelihood_data_with_missing(data, allow_1d)
    return arr


cdef tuple _coerce_likelihood_data_with_missing(
    object data, bint allow_1d
) except *:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    cdef cnp.ndarray farr
    cdef cnp.ndarray out
    cdef cnp.ndarray arr
    cdef Py_ssize_t i
    cdef double fv
    cdef int iv
    if np.issubdtype(data.dtype, np.floating):
        farr = np.ascontiguousarray(data, dtype=np.float64)
        if allow_1d and farr.ndim == 1:
            out = np.empty(farr.shape[0], dtype=np.int32)
        elif farr.ndim == 2:
            out = np.empty((farr.shape[0], farr.shape[1]), dtype=np.int32)
        elif allow_1d:
            raise ValueError("data must be 1-D or 2-D (n_samples, n_columns)")
        else:
            raise ValueError("data must be 2-D (n_samples, n_columns)")
        for i in range(out.size):
            fv = farr.flat[i]
            if np.isnan(fv):
                out.flat[i] = SP_MISSING_EVIDENCE
            else:
                iv = int(fv)
                if iv != fv:
                    raise ValueError(
                        f"observed outcome must be an integer, got {fv}"
                    )
                if iv < 0:
                    raise ValueError(
                        f"outcome value must be non-negative, got {iv}"
                    )
                out.flat[i] = iv
        return out, True
    arr = np.ascontiguousarray(data, dtype=np.int32)
    if allow_1d and arr.ndim == 1:
        return arr, False
    if arr.ndim == 2:
        return arr, False
    if allow_1d:
        raise ValueError("data must be 1-D or 2-D (n_samples, n_columns)")
    raise ValueError("data must be 2-D (n_samples, n_columns)")


cdef void _leaf_column_map(
    CompiledCircuit g,
    object var_to_col,
    Py_ssize_t n_cols,
    vector[int]& leaf_col,
) except *:
    cdef size_t i
    cdef int var
    cdef int col
    leaf_col.assign(g.n_nodes, -1)
    for i in range(g.n_nodes):
        if g.kinds[i] != NODE_INPUT:
            continue
        var = g.leaf_var[i]
        if var_to_col is None:
            col = var
        else:
            col = int(var_to_col[var])
        if col < 0 or col >= n_cols:
            raise ValueError(
                f"variable {var} maps to column {col} out of range "
                f"[0, {n_cols})"
            )
        leaf_col[i] = col


cdef void _build_evidence_vector(
    CompiledCircuit g, cnp.ndarray row, vector[int]& ev, bint allow_missing
) except *:
    cdef int max_var = g.max_var
    cdef cnp.ndarray arr = np.asarray(row, dtype=np.int32).ravel()
    cdef size_t n
    cdef int v
    cdef int val
    cdef int card
    if arr.size <= max_var:
        raise ValueError(
            f"assignment array length {arr.size} is shorter than required "
            f"width {max_var + 1}"
        )
    ev.assign(max_var + 1, SP_MISSING_EVIDENCE)
    for v in range(max_var + 1):
        ev[v] = int(arr[v])
    for n in range(g.n_nodes):
        if g.kinds[n] == NODE_INPUT:
            v = g.leaf_var[n]
            val = ev[v]
            card = g.leaf_card[n]
            if val == SP_MISSING_EVIDENCE:
                if not allow_missing:
                    raise ValueError(
                        f"outcome value must be non-negative, got {val}"
                    )
                continue
            if val < 0 or val >= card:
                raise ValueError(
                    f"evidence for variable {v}: outcome {val} out of range "
                    f"[0, {card})"
                )


cdef void _flat_eval_sum_node(
    CompiledCircuit g,
    size_t n,
    bint log_space,
    double[::1] val,
) noexcept nogil:
    """Evaluate a single sum node into val[n] (1-D activation buffer)."""
    cdef size_t start = g.child_off[n]
    cdef size_t stop = g.child_off[n + 1]
    cdef size_t k
    cdef size_t nf = stop - start
    cdef double acc
    cdef double max_log
    cdef double sum_exp
    cdef double term
    cdef double terms[SP_MAX_SUM_FANIN]
    if log_space:
        if nf == 0:
            val[n] = -INFINITY
        elif nf <= SP_MAX_SUM_FANIN:
            for k in range(start, stop):
                terms[k - start] = g.sum_logw_flat[k] + val[g.children_flat[k]]
            val[n] = sp_logsumexp_ptr(terms, nf)
        else:
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


cdef void _flat_eval_sum_node_batch(
    CompiledCircuit g,
    size_t n,
    Py_ssize_t n_rows,
    bint log_space,
    double[:, ::1] val,
) noexcept nogil:
    """Evaluate a single sum node across all batch lanes (val[n, r])."""
    cdef size_t start = g.child_off[n]
    cdef size_t stop = g.child_off[n + 1]
    cdef size_t k
    cdef size_t nf = stop - start
    cdef Py_ssize_t r
    cdef double acc
    cdef double terms[SP_MAX_SUM_FANIN]
    cdef double max_log
    cdef double sum_exp
    cdef double term
    if log_space:
        if nf == 0:
            for r in range(n_rows):
                val[n, r] = -INFINITY
        elif nf <= SP_MAX_SUM_FANIN:
            for r in range(n_rows):
                for k in range(start, stop):
                    terms[k - start] = (
                        g.sum_logw_flat[k] + val[g.children_flat[k], r]
                    )
                val[n, r] = sp_logsumexp_ptr(terms, nf)
        else:
            for r in range(n_rows):
                max_log = -INFINITY
                for k in range(start, stop):
                    term = g.sum_logw_flat[k] + val[g.children_flat[k], r]
                    if term > max_log:
                        max_log = term
                if not isfinite(max_log) or max_log == -INFINITY:
                    val[n, r] = -INFINITY
                else:
                    sum_exp = 0.0
                    for k in range(start, stop):
                        term = g.sum_logw_flat[k] + val[g.children_flat[k], r]
                        if isfinite(term) and term > -INFINITY:
                            sum_exp += exp(term - max_log)
                    if sum_exp <= 0.0:
                        val[n, r] = -INFINITY
                    else:
                        val[n, r] = max_log + log(sum_exp)
    else:
        for r in range(n_rows):
            acc = 0.0
            for k in range(start, stop):
                acc += g.sum_w_flat[k] * val[g.children_flat[k], r]
            val[n, r] = acc


cdef void _validate_batch_data(
    CompiledCircuit g,
    const int[:, ::1] data,
    const vector[int]& leaf_col,
    bint allow_missing,
) except *:
    cdef Py_ssize_t n_rows = data.shape[0]
    cdef Py_ssize_t r
    cdef size_t n
    cdef int col
    cdef int value
    cdef int card
    for n in range(g.n_nodes):
        if g.kinds[n] != NODE_INPUT:
            continue
        col = leaf_col[n]
        card = g.leaf_card[n]
        for r in range(n_rows):
            value = data[r, col]
            if value == SP_MISSING_EVIDENCE:
                if not allow_missing:
                    raise ValueError(
                        f"value {value} out of range [0, {card}) in column {col}"
                    )
                continue
            if value < 0 or value >= card:
                raise ValueError(
                    f"value {value} out of range [0, {card}) in column {col}"
                )


cdef void _flat_eval_batch(
    CompiledCircuit g,
    const int[:, ::1] data,
    const vector[int]& leaf_col,
    Py_ssize_t n_rows,
    bint log_space,
    double[:, ::1] val,
    double[::1] out,
) noexcept nogil:
    cdef size_t n
    cdef size_t k
    cdef size_t start
    cdef size_t stop
    cdef int kind
    cdef int col
    cdef int value
    cdef size_t off
    cdef Py_ssize_t r
    cdef double acc
    for n in range(g.n_nodes):
        kind = g.kinds[n]
        if kind == NODE_INPUT:
            col = leaf_col[n]
            off = g.leaf_pmf_off[n]
            if log_space:
                for r in range(n_rows):
                    value = data[r, col]
                    if value < 0:
                        val[n, r] = 0.0
                    else:
                        val[n, r] = g.leaf_logpmf_flat[off + value]
            else:
                for r in range(n_rows):
                    value = data[r, col]
                    if value < 0:
                        val[n, r] = 1.0
                    else:
                        val[n, r] = g.leaf_pmf_flat[off + value]
        elif kind == NODE_PRODUCT:
            start = g.child_off[n]
            stop = g.child_off[n + 1]
            if log_space:
                for r in range(n_rows):
                    acc = 0.0
                    for k in range(start, stop):
                        acc += val[g.children_flat[k], r]
                    val[n, r] = acc
            else:
                for r in range(n_rows):
                    acc = 1.0
                    for k in range(start, stop):
                        acc *= val[g.children_flat[k], r]
                    val[n, r] = acc
        else:
            _flat_eval_sum_node_batch(g, n, n_rows, log_space, val)
    for r in range(n_rows):
        out[r] = val[g.root_index, r]


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
    for n in range(g.n_nodes):
        kind = g.kinds[n]
        if kind == NODE_INPUT:
            var = g.leaf_var[n]
            value = ev[var]
            if value < 0:
                val[n] = 0.0 if log_space else 1.0
            elif log_space:
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
            _flat_eval_sum_node(g, n, log_space, val)
    return val[g.root_index]


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


cdef class CompiledCircuit:
    """Flattened, cache-friendly circuit for nogil inference.

    Compile once when topology is fixed. After parameter updates, call
    :meth:`refresh_parameters` before subsequent fast-path queries.
    """

    def __cinit__(self):
        self.node_objs = []
        self.variables = []
        self.max_var = -1
        self._metric_pools = {}

    def __init__(self, root):
        if not isinstance(root, CircuitNode):
            raise TypeError("CompiledCircuit expects a CircuitNode")
        cdef CircuitNode r = <CircuitNode>root
        if r.scope.size() == 0:
            r.propagate_scope()
        self._build(r)
        self.variables = sorted(r.scope)
        if self.variables:
            self.max_var = self.variables[len(self.variables) - 1]
        else:
            self.max_var = -1

    cdef void _build(self, CircuitNode root) except *:
        cdef list order = []
        cdef dict index_of = {}
        self._postorder(root, index_of, order)
        self.n_nodes = len(order)
        self.root_index = index_of[root.id]
        self.node_objs = order

        self.kinds.assign(self.n_nodes, 0)
        self.child_off.assign(self.n_nodes + 1, 0)
        self.leaf_kind.assign(self.n_nodes, LEAF_GENERIC)
        self.leaf_var.assign(self.n_nodes, -1)
        self.leaf_card.assign(self.n_nodes, 0)
        self.leaf_trainable.assign(self.n_nodes, 0)
        self.leaf_pmf_off.assign(self.n_nodes + 1, 0)
        self.node_ids.assign(self.n_nodes, 0)
        self.scope_sig.assign(self.n_nodes, 0)
        self.scope_size.assign(self.n_nodes, 0)
        self.scope_vars_off.assign(self.n_nodes + 1, 0)

        cdef size_t n
        cdef CircuitNode node
        cdef InternalNode internal
        cdef int kind
        cdef int support

        for n in range(self.n_nodes):
            node = <CircuitNode>order[n]
            self.node_ids[n] = node.id
            kind = node.node_kind
            self.kinds[n] = kind
            self._fill_scope(node, n)
            if kind == NODE_INPUT:
                if not isinstance(node, FiniteDiscreteInputNode):
                    raise TypeError(
                        "CompiledCircuit only supports FiniteDiscreteInputNode "
                        f"leaves, got {type(node).__name__}"
                    )
                self.child_off[n + 1] = 0
                self._classify_leaf(node, n)
                support = self.leaf_card[n]
                self.leaf_pmf_off[n + 1] = <size_t>support
            else:
                internal = <InternalNode>node
                self.child_off[n + 1] = internal.num_children()
                self.leaf_pmf_off[n + 1] = 0

        for n in range(self.n_nodes):
            self.child_off[n + 1] += self.child_off[n]
            self.leaf_pmf_off[n + 1] += self.leaf_pmf_off[n]
            self.scope_vars_off[n + 1] += self.scope_vars_off[n]

        self.children_flat.assign(self.child_off[self.n_nodes], 0)
        self.sum_w_flat.assign(self.child_off[self.n_nodes], 0.0)
        self.sum_logw_flat.assign(self.child_off[self.n_nodes], 0.0)
        self.leaf_pmf_flat.assign(self.leaf_pmf_off[self.n_nodes], 0.0)
        self.leaf_logpmf_flat.assign(self.leaf_pmf_off[self.n_nodes], 0.0)

        cdef size_t base
        cdef size_t lbase
        cdef size_t soff
        cdef size_t k
        cdef SumNode s
        cdef FiniteDiscreteInputNode leaf
        cdef double pmf
        cdef int v
        cdef unordered_set[int].iterator sit

        self.scope_vars_flat.assign(self.scope_vars_off[self.n_nodes], 0)
        for n in range(self.n_nodes):
            node = <CircuitNode>order[n]
            soff = self.scope_vars_off[n]
            for v in sorted(node.scope):
                self.scope_vars_flat[soff] = v
                soff += 1

        for n in range(self.n_nodes):
            node = <CircuitNode>order[n]
            if node.node_kind == NODE_INPUT:
                leaf = <FiniteDiscreteInputNode>node
                lbase = self.leaf_pmf_off[n]
                for k in range(<size_t>self.leaf_card[n]):
                    pmf = leaf.pmf_at(k)
                    self.leaf_pmf_flat[lbase + k] = pmf
                    self.leaf_logpmf_flat[lbase + k] = graph_safe_log(pmf)
            else:
                internal = <InternalNode>node
                base = self.child_off[n]
                for k in range(internal.num_children()):
                    self.children_flat[base + k] = index_of[internal.child_at(k).id]
                if node.node_kind == NODE_SUM:
                    s = <SumNode>node
                    for k in range(s.num_children()):
                        self.sum_w_flat[base + k] = s.parameter_at(k)
                        self.sum_logw_flat[base + k] = graph_safe_log(s.parameter_at(k))

    cdef void _fill_scope(self, CircuitNode node, size_t n) except *:
        cdef int v
        self.scope_sig[n] = _scope_sig(node)
        self.scope_size[n] = <int>node.scope.size()
        self.scope_vars_off[n + 1] = <size_t>node.scope.size()

    cdef void _classify_leaf(self, CircuitNode node, size_t n) except *:
        cdef FiniteDiscreteInputNode leaf = <FiniteDiscreteInputNode>node
        if isinstance(node, CategoricalInputNode):
            self.leaf_kind[n] = LEAF_CATEGORICAL
            self.leaf_trainable[n] = 1
        elif isinstance(node, BernoulliInputNode):
            self.leaf_kind[n] = LEAF_BERNOULLI
            self.leaf_trainable[n] = 1
        elif isinstance(node, IndicatorInputNode):
            self.leaf_kind[n] = LEAF_INDICATOR
        elif isinstance(node, LiteralInputNode):
            self.leaf_kind[n] = LEAF_LITERAL
        elif isinstance(node, DiscreteLogisticInputNode):
            self.leaf_kind[n] = LEAF_DISCRETE_LOGISTIC
        else:
            self.leaf_kind[n] = LEAF_GENERIC
            if isinstance(node, FiniteDiscreteInputNode):
                self.leaf_trainable[n] = 1
        self.leaf_var[n] = leaf.scope_var_c()
        self.leaf_card[n] = <int>leaf.support_size()

    cdef void _postorder(self, CircuitNode node, dict index_of, list order) except *:
        if node.id in index_of:
            return
        cdef size_t k
        cdef InternalNode internal
        if node.node_kind != NODE_INPUT:
            internal = <InternalNode>node
            for k in range(internal.num_children()):
                self._postorder(internal.child_at(k), index_of, order)
        order.append(node)
        index_of[node.id] = len(order) - 1

    def refresh_parameters(self):
        """Re-copy sum weights and leaf PMFs from live nodes after an update."""
        self._refresh_sum_weights()
        self._refresh_leaf_pmfs()

    def build_metric_pools(self, metric):
        """Precompute per-leaf pairwise cost matrices for *metric* (GIL setup).

        Pools are cached on the compiled circuit keyed by ``id(metric)`` and
        reused by nogil ESD / OT solvers on the fast path.
        """
        cdef size_t n
        cdef int var
        cdef int card
        cdef vector[double] tmp
        cdef object key
        cdef dict pools = {}
        for n in range(self.n_nodes):
            if self.kinds[n] != NODE_INPUT:
                continue
            var = self.leaf_var[n]
            card = self.leaf_card[n]
            key = ("pairwise", var, card)
            if key not in pools:
                metric.pairwise(var, <size_t>card, tmp)
                pools[key] = np.asarray(tmp, dtype=np.float64).copy()
        self._metric_pools[id(metric)] = pools
        return pools

    cdef void _refresh_sum_weights(self) except *:
        cdef size_t n
        cdef size_t base
        cdef size_t k
        cdef CircuitNode node
        cdef SumNode s
        for n in range(self.n_nodes):
            if self.kinds[n] != NODE_SUM:
                continue
            node = <CircuitNode>self.node_objs[n]
            s = <SumNode>node
            base = self.child_off[n]
            for k in range(s.num_children()):
                self.sum_w_flat[base + k] = s.parameter_at(k)
                self.sum_logw_flat[base + k] = graph_safe_log(s.parameter_at(k))

    cdef void _refresh_leaf_pmfs(self) except *:
        cdef size_t n
        cdef size_t lbase
        cdef size_t k
        cdef FiniteDiscreteInputNode leaf
        cdef double pmf
        for n in range(self.n_nodes):
            if self.kinds[n] != NODE_INPUT:
                continue
            leaf = <FiniteDiscreteInputNode>self.node_objs[n]
            lbase = self.leaf_pmf_off[n]
            for k in range(<size_t>self.leaf_card[n]):
                pmf = leaf.pmf_at(k)
                self.leaf_pmf_flat[lbase + k] = pmf
                self.leaf_logpmf_flat[lbase + k] = graph_safe_log(pmf)

    def log_likelihood(self, object data, object var_to_col=None):
        """Log-likelihood for a 1-D or 2-D integer assignment array."""
        return self._likelihood_impl(data, var_to_col, True)

    def likelihood(self, object data, object var_to_col=None):
        """Likelihood for a 1-D or 2-D integer assignment array."""
        return self._likelihood_impl(data, var_to_col, False)

    cdef object _likelihood_impl(
        self, object data, object var_to_col, bint log_space
    ):
        cdef cnp.ndarray arr
        cdef bint allow_missing
        arr, allow_missing = _coerce_likelihood_data_with_missing(data, True)
        cdef vector[int] leaf_col
        cdef Py_ssize_t n_rows
        cdef Py_ssize_t n_cols
        cdef object out
        cdef double[::1] out_view
        cdef const int[:, ::1] data_view
        cdef vector[int] ev
        cdef object val_arr
        cdef double[::1] val
        cdef double result
        if arr.ndim == 1:
            _build_evidence_vector(self, arr, ev, allow_missing)
            val_arr = np.empty(self.n_nodes, dtype=np.float64)
            val = val_arr
            with nogil:
                result = _flat_eval(self, ev.data(), log_space, val)
            return result
        n_rows = arr.shape[0]
        n_cols = arr.shape[1]
        _leaf_column_map(self, var_to_col, n_cols, leaf_col)
        out = np.empty(n_rows, dtype=np.float64)
        out_view = out
        data_view = arr
        self._score(data_view, leaf_col, out_view, log_space, allow_missing)
        return out

    cdef void _score(
        self,
        const int[:, ::1] data,
        const vector[int]& leaf_col,
        double[::1] out,
        bint log_space,
        bint allow_missing,
    ) except *:
        cdef Py_ssize_t n_rows = data.shape[0]
        cdef object val_arr = np.empty((self.n_nodes, n_rows), dtype=np.float64)
        cdef double[:, ::1] val_view = val_arr
        _validate_batch_data(self, data, leaf_col, allow_missing)
        with nogil:
            _flat_eval_batch(
                self, data, leaf_col, n_rows, log_space, val_view, out
            )

    def sample(self, Py_ssize_t n_samples, object seed=None):
        """Draw samples as a 2-D integer array of shape ``(n_samples, max_var+1)``."""
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        cdef unsigned long long rng_seed
        if seed is None:
            rng_seed = <unsigned long long>time.time_ns()
        else:
            rng_seed = <unsigned long long>int(seed)
        cdef RandomState rng = RandomState(rng_seed)
        cdef size_t width = <size_t>(self.max_var + 1)
        cdef object out = np.full((n_samples, width), -1, dtype=np.int32)
        cdef int[:, ::1] out_view = out
        cdef Py_ssize_t i
        cdef size_t root_index = self.root_index
        if n_samples > 0:
            with nogil:
                for i in range(n_samples):
                    _flat_sample_node(
                        self, root_index, rng, &out_view[i, 0]
                    )
        return out

    def mean_log_likelihood_and_grad(self, object dataset, object var_to_col=None):
        """Mean log-likelihood and gradient over a dataset (flat nogil path)."""
        from sparc.grad import compiled_mean_log_likelihood_and_grad
        return compiled_mean_log_likelihood_and_grad(self, dataset, var_to_col)

    def cw_distance(self, other, double metric_p=1.0, double scale_factor=1.0, object metric=None):
        from sparc.queries.cw import cw_distance
        return cw_distance(self, other, metric_p=metric_p, scale_factor=scale_factor, metric=metric)

    def cw_distance_and_grad(self, other, double metric_p=1.0, double scale_factor=1.0, object metric=None):
        from sparc.queries.cw import cw_distance_and_grad
        return cw_distance_and_grad(self, other, metric_p=metric_p, scale_factor=scale_factor, metric=metric)

    def gcw_crossterm(self, other, double metric_p=1.0, double scale_factor_1=1.0,
                      double scale_factor_2=1.0, object metric1=None, object metric2=None):
        from sparc.queries.gcw import gcw_crossterm
        return gcw_crossterm(self, other, metric_p=metric_p,
                               scale_factor_1=scale_factor_1, scale_factor_2=scale_factor_2,
                               metric1=metric1, metric2=metric2)

    def gcw_crossterm_and_grad(self, other, double metric_p=1.0, double scale_factor_1=1.0,
                               double scale_factor_2=1.0, object metric1=None, object metric2=None):
        from sparc.queries.gcw import gcw_crossterm_and_grad
        return gcw_crossterm_and_grad(self, other, metric_p=metric_p,
                                      scale_factor_1=scale_factor_1, scale_factor_2=scale_factor_2,
                                      metric1=metric1, metric2=metric2)

    def exp_query(self, other):
        from sparc.queries.expectation import exp_query
        return exp_query(self, other)

    def exp_query_and_grad(self, other):
        from sparc.queries.expectation import exp_query_and_grad
        return exp_query_and_grad(self, other)

    def log_exp_query(self, other):
        from sparc.queries.expectation import log_exp_query
        return log_exp_query(self, other)

    def log_exp_query_and_grad(self, other):
        from sparc.queries.expectation import log_exp_query_and_grad
        return log_exp_query_and_grad(self, other)

    def expected_squared_distance(self, double metric_p=1.0, double scale_factor=1.0, object metric=None):
        from sparc.queries.esd import expected_squared_distance
        return expected_squared_distance(self, metric_p=metric_p, scale_factor=scale_factor, metric=metric)

    def expected_squared_distance_and_grad(self, double metric_p=1.0, double scale_factor=1.0, object metric=None):
        from sparc.queries.esd import expected_squared_distance_and_grad
        return expected_squared_distance_and_grad(self, metric_p=metric_p, scale_factor=scale_factor, metric=metric)
