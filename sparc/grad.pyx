# distutils: language = c++
# cython: boundscheck=False, wraparound=False
"""Differentiable mean log-likelihood (reverse-mode AD over the circuit DAG).

``GradBundle`` is the single gradient container used everywhere in SparC:
``value`` plus ``sum_grads`` / ``cat_grads`` dicts keyed by ``node.id``. Two-
circuit queries simply return a pair of ``GradBundle`` objects.
"""

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.math cimport exp, INFINITY, isfinite, log

import numpy as np
cimport numpy as cnp

from sparc._graph cimport (
    CompiledCircuit,
    _coerce_data_array,
    _flat_eval_batch,
    _leaf_column_map,
    _max_var_from_scope,
    _validate_batch_data,
)
from sparc._mathutils cimport sp_logsumexp, sp_safe_log
from sparc.eval cimport _evidence_from_row
from sparc.nodes cimport (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    Evidence,
    FiniteDiscreteInputNode,
    InputNode,
    NODE_INPUT,
    NODE_PRODUCT,
    NODE_SUM,
    ProductNode,
    SumNode,
)

cdef double NEG_INF = float("-inf")


cdef class GradBundle:
    """Gradient container returned by query and likelihood routines.

    Attributes:
        value: Scalar objective value (mean log-likelihood or query value).
        sum_grads: ``SumNode.id`` -> gradient w.r.t. that node's parameters.
        cat_grads: Leaf ``id`` -> gradient w.r.t. leaf probabilities.

    Gradients are w.r.t. linear parameters (no simplex projection). Project
    onto the simplex tangent before stepping (see :mod:`sparc.optim`).
    """

    def __cinit__(self):
        self.value = 0.0
        self.sum_grads = {}
        self.cat_grads = {}


cdef object grad_arr(dict store, object key, size_t n):
    cdef object arr = store.get(key)
    if arr is None:
        arr = np.zeros(n, dtype=np.float64)
        store[key] = arr
    return arr


cdef class _LLGradContext:
    """Reverse-mode AD over the log-likelihood DAG, accumulated over a dataset."""

    cdef Evidence evidence
    cdef unordered_map[size_t, double] memo
    cdef unordered_map[size_t, double] adjoints
    cdef list tape
    cdef dict sum_grads
    cdef dict cat_grads

    def __cinit__(self):
        self.tape = []
        self.sum_grads = {}
        self.cat_grads = {}

    cdef void _reset(self):
        self.memo.clear()
        self.adjoints.clear()
        self.tape = []

    cdef inline void _add_adjoint(self, size_t nid, double val):
        self.adjoints[nid] = self.adjoints[nid] + val

    cdef double _forward(self, CircuitNode node) except *:
        cdef double result
        if self.memo.count(node.id):
            return self.memo[node.id]
        result = self._forward_impl(node)
        self.memo[node.id] = result
        self.tape.append(node)
        return result

    cdef double _forward_impl(self, CircuitNode node) except *:
        cdef ProductNode prod
        cdef SumNode s
        cdef size_t i
        cdef size_t n
        cdef double total
        cdef vector[double] terms
        if node.node_kind == NODE_INPUT:
            return sp_safe_log((<InputNode>node).prob_c(self.evidence))
        if node.node_kind == NODE_PRODUCT:
            prod = <ProductNode>node
            n = prod.num_children()
            total = 0.0
            for i in range(n):
                total += self._forward(prod.child_at(i))
            return total
        if node.node_kind == NODE_SUM:
            s = <SumNode>node
            n = s.num_children()
            terms.resize(n)
            for i in range(n):
                terms[i] = sp_safe_log(s.parameter_at(i)) + self._forward(s.child_at(i))
            return sp_logsumexp(terms)
        raise TypeError(f"unsupported node type for query: {type(node).__name__}")

    cdef void _run_backward(self) except *:
        cdef ssize_t k
        cdef CircuitNode node
        cdef double bar
        for k in range(<ssize_t>len(self.tape) - 1, -1, -1):
            node = <CircuitNode>self.tape[k]
            if self.adjoints.count(node.id) == 0:
                continue
            bar = self.adjoints[node.id]
            if bar == 0.0:
                continue
            if node.node_kind == NODE_INPUT:
                self._backward_leaf(<InputNode>node, bar)
            elif node.node_kind == NODE_PRODUCT:
                self._backward_prod(<ProductNode>node, bar)
            elif node.node_kind == NODE_SUM:
                self._backward_sum(<SumNode>node, bar)

    cdef void _backward_leaf(self, InputNode node, double bar) except *:
        cdef FiniteDiscreteInputNode leaf
        cdef int var
        cdef int value
        cdef double p_v
        cdef object arr
        if not (isinstance(node, CategoricalInputNode)
                or isinstance(node, BernoulliInputNode)):
            return
        leaf = <FiniteDiscreteInputNode>node
        var = leaf.scope_var_c()
        value = self.evidence.get(var)
        p_v = leaf.pmf_at(<size_t>value)
        if p_v <= 0.0:
            return
        arr = grad_arr(self.cat_grads, leaf.id, leaf.support_size())
        arr[value] += bar / p_v

    cdef void _backward_prod(self, ProductNode node, double bar) except *:
        cdef size_t i
        cdef size_t n = node.num_children()
        for i in range(n):
            self._add_adjoint(node.child_at(i).id, bar)

    cdef void _backward_sum(self, SumNode node, double bar) except *:
        cdef size_t i
        cdef size_t n = node.num_children()
        cdef double ll = self.memo[node.id]
        cdef CircuitNode child
        cdef double weight
        cdef double child_ll
        cdef double log_w
        cdef object arr
        if ll == NEG_INF:
            return
        arr = grad_arr(self.sum_grads, node.id, n)
        for i in range(n):
            child = node.child_at(i)
            weight = node.parameter_at(i)
            child_ll = self.memo[child.id]
            log_w = sp_safe_log(weight)
            if log_w > NEG_INF and child_ll > NEG_INF:
                self._add_adjoint(child.id, bar * exp(log_w + child_ll - ll))
            if child_ll > NEG_INF:
                arr[i] += bar * exp(child_ll - ll)

    cdef tuple solve_dataset(
        self, CircuitNode root, cnp.ndarray arr, object var_to_col
    ):
        cdef Py_ssize_t n = arr.shape[0]
        cdef Py_ssize_t idx
        cdef double total_ll = 0.0
        cdef double inv_n
        cdef double ll
        cdef GradBundle grads
        cdef int max_var = _max_var_from_scope(root.scope)
        if root.scope.size() == 0:
            raise ValueError(
                "root scope is empty; call propagate_scope() on the circuit first"
            )
        if n == 0:
            raise ValueError("dataset must contain at least one datapoint")
        inv_n = 1.0 / <double>n
        for idx in range(n):
            self.evidence = _evidence_from_row(
                arr[idx], arr.shape[1], max_var, root.scope, var_to_col
            )
            self._reset()
            ll = self._forward(root)
            total_ll += ll
            if ll > NEG_INF:
                self._add_adjoint(root.id, inv_n)
                self._run_backward()
        grads = GradBundle()
        grads.value = total_ll * inv_n
        grads.sum_grads = self.sum_grads
        grads.cat_grads = self.cat_grads
        return (grads.value, grads)


# --- Flattened nogil gradient path --------------------------------------------

cdef void _flat_grad_core(
    CompiledCircuit g,
    const int[:, ::1] data,
    const vector[int]& leaf_col,
    Py_ssize_t n_rows,
    double[:, ::1] val,
    double[::1] ll_scratch,
    double* adj,
    double* sum_pool,
    double* cat_pool,
    double inv_n,
    double* total_ll_out,
) noexcept nogil:
    cdef Py_ssize_t r
    cdef ssize_t ni
    cdef size_t n
    cdef size_t k
    cdef size_t start
    cdef size_t stop
    cdef size_t off
    cdef size_t child
    cdef int kind
    cdef int col
    cdef int value
    cdef double bar
    cdef double ll
    cdef double child_ll
    cdef double log_w
    cdef double p_v
    cdef double total = 0.0
    _flat_eval_batch(g, data, leaf_col, n_rows, True, val, ll_scratch)
    for r in range(n_rows):
        total += val[g.root_index, r]
        if val[g.root_index, r] <= -INFINITY:
            continue
        for n in range(g.n_nodes):
            adj[n] = 0.0
        adj[g.root_index] = inv_n
        for ni in range(<ssize_t>g.n_nodes - 1, -1, -1):
            n = <size_t>ni
            bar = adj[n]
            if bar == 0.0:
                continue
            kind = g.kinds[n]
            if kind == NODE_INPUT:
                if g.leaf_trainable[n]:
                    col = leaf_col[n]
                    value = data[r, col]
                    off = g.leaf_pmf_off[n]
                    p_v = g.leaf_pmf_flat[off + <size_t>value]
                    if p_v > 0.0:
                        cat_pool[off + <size_t>value] += bar / p_v
            elif kind == NODE_PRODUCT:
                start = g.child_off[n]
                stop = g.child_off[n + 1]
                for k in range(start, stop):
                    adj[g.children_flat[k]] += bar
            else:
                ll = val[n, r]
                if ll == -INFINITY:
                    continue
                start = g.child_off[n]
                stop = g.child_off[n + 1]
                for k in range(start, stop):
                    child = g.children_flat[k]
                    child_ll = val[child, r]
                    log_w = g.sum_logw_flat[k]
                    if log_w > -INFINITY and child_ll > -INFINITY:
                        adj[child] += bar * exp(log_w + child_ll - ll)
                    if child_ll > -INFINITY:
                        sum_pool[k] += bar * exp(child_ll - ll)
    total_ll_out[0] = total


cdef tuple _flat_solve_dataset(
    CompiledCircuit g, cnp.ndarray arr, object var_to_col
):
    cdef Py_ssize_t n = arr.shape[0]
    if n == 0:
        raise ValueError("dataset must contain at least one datapoint")
    cdef double inv_n = 1.0 / <double>n
    cdef vector[int] leaf_col
    _leaf_column_map(g, var_to_col, arr.shape[1], leaf_col)
    _validate_batch_data(g, arr, leaf_col)

    cdef object val_arr = np.empty((g.n_nodes, n), dtype=np.float64)
    cdef double[:, ::1] val_view = val_arr
    cdef object ll_scratch_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] ll_scratch = ll_scratch_arr
    cdef vector[double] adj
    cdef vector[double] sum_pool
    cdef vector[double] cat_pool
    adj.assign(g.n_nodes, 0.0)
    sum_pool.assign(g.children_flat.size(), 0.0)
    cat_pool.assign(g.leaf_pmf_flat.size(), 0.0)
    cdef double total_ll = 0.0
    cdef const int[:, ::1] data_view = arr
    with nogil:
        _flat_grad_core(
            g, data_view, leaf_col, n,
            val_view, ll_scratch, adj.data(), sum_pool.data(), cat_pool.data(),
            inv_n, &total_ll,
        )

    cdef GradBundle grads = GradBundle()
    grads.value = total_ll * inv_n
    cdef dict sum_grads = {}
    cdef dict cat_grads = {}
    cdef size_t nn
    cdef size_t start
    cdef size_t stop
    cdef size_t off
    cdef size_t k
    cdef int card
    cdef object out_arr
    for nn in range(g.n_nodes):
        if g.kinds[nn] == NODE_SUM:
            start = g.child_off[nn]
            stop = g.child_off[nn + 1]
            out_arr = np.empty(stop - start, dtype=np.float64)
            for k in range(start, stop):
                out_arr[k - start] = sum_pool[k]
            sum_grads[g.node_ids[nn]] = out_arr
        elif g.kinds[nn] == NODE_INPUT and g.leaf_trainable[nn]:
            off = g.leaf_pmf_off[nn]
            card = g.leaf_card[nn]
            out_arr = np.empty(card, dtype=np.float64)
            for k in range(<size_t>card):
                out_arr[k] = cat_pool[off + k]
            cat_grads[g.node_ids[nn]] = out_arr
    grads.sum_grads = sum_grads
    grads.cat_grads = cat_grads
    return (grads.value, grads)


def mean_log_likelihood_and_grad(CircuitNode root, object dataset, object var_to_col=None):
    """Mean log-likelihood and gradient (object-graph path)."""
    cdef cnp.ndarray arr = _coerce_data_array(dataset, False)
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    return _LLGradContext().solve_dataset(root, arr, var_to_col)


def compiled_mean_log_likelihood_and_grad(
    CompiledCircuit g, object dataset, object var_to_col=None
):
    """Mean log-likelihood and gradient (flat nogil path)."""
    cdef cnp.ndarray arr = _coerce_data_array(dataset, False)
    return _flat_solve_dataset(g, arr, var_to_col)
