# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
# cython: boundscheck=False, wraparound=False
"""Evaluation queries: per-datapoint likelihood / log-likelihood / sampling,
plus a compiled, vectorized batched log-likelihood fast path.

Dispatch over node types uses the integer ``node_kind`` tag plus the leaf
vtable (``InputNode.prob_c`` / ``sample_into_c``), so new leaf types are
evaluated without any change here.
"""

import time

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.math cimport exp, INFINITY, isfinite, log

import numpy as np

from sparc._graph cimport CompiledGraph
from sparc._mathutils cimport sp_logsumexp, sp_safe_log
from sparc.nodes cimport (
    CircuitNode,
    Evidence,
    FiniteDiscreteInputNode,
    InputNode,
    InternalNode,
    NODE_INPUT,
    NODE_PRODUCT,
    NODE_SUM,
    ProductNode,
    RandomState,
    SumNode,
)


# --- Single-datapoint queries -------------------------------------------------

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


# --- Flattened nogil fast path ------------------------------------------------

cdef void _build_evidence_vector(
    CompiledGraph g, object assignment, vector[int]& ev
) except *:
    """Materialize a var-indexed evidence array and validate it.

    Reproduces the object-path semantics exactly: missing scope variables and
    out-of-range outcomes raise the same ``ValueError`` messages as
    :class:`~sparc.nodes.Evidence`.
    """
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
    for key, value in assignment.items():
        var = int(key)
        outcome = int(value)
        if var < 0:
            raise ValueError(f"variable index must be non-negative, got {var}")
        if outcome < 0:
            raise ValueError(f"outcome value must be non-negative, got {outcome}")
        if var <= max_var:
            ev[var] = outcome
    # require_vars: every scope variable must be present.
    for v in g.variables:
        if ev[v] < 0:
            raise ValueError(f"missing evidence for variable {v}")
    # validate_value: every leaf outcome must lie in range.
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
    CompiledGraph g, const int* ev, bint log_space, double[::1] val
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
        else:  # NODE_SUM
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


cdef double _flat_query(CompiledGraph g, object assignment, bint log_space) except *:
    cdef vector[int] ev
    _build_evidence_vector(g, assignment, ev)
    cdef object val_arr = np.empty(g.n_nodes, dtype=np.float64)
    cdef double[::1] val = val_arr
    cdef double result
    with nogil:
        result = _flat_eval(g, ev.data(), log_space, val)
    return result


cdef double _dispatch_query(CircuitNode root, object assignment, bint log_space) except *:
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    cdef CompiledGraph g = CompiledGraph()
    g.build(root)
    if g.has_fallback:
        return _run_query(root, assignment, log_space)
    return _flat_query(g, assignment, log_space)


cpdef double likelihood(CircuitNode root, object assignment) except *:
    """Evaluate the probability of a single complete assignment.

    Args:
        root: Circuit root node with propagated scope.
        assignment: ``{var: value}`` mapping covering the root scope.

    Returns:
        The likelihood :math:`P(\\mathbf{x})`.
    """
    return _dispatch_query(root, assignment, False)


cpdef double log_likelihood(CircuitNode root, object assignment) except *:
    """Evaluate the log-probability of a single complete assignment.

    Args:
        root: Circuit root node with propagated scope.
        assignment: ``{var: value}`` mapping covering the root scope.

    Returns:
        The log-likelihood :math:`\\log P(\\mathbf{x})`.
    """
    return _dispatch_query(root, assignment, True)


# --- Sampling -----------------------------------------------------------------

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


cdef void _flat_sample_node(
    CompiledGraph g, size_t n, RandomState rng, int* out
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
    else:  # NODE_SUM
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


cdef list _sample_object(CircuitNode root, Py_ssize_t n_samples, RandomState rng):
    cdef list results = []
    cdef Py_ssize_t i
    cdef dict row
    for i in range(n_samples):
        row = {}
        _sample_node(root, rng, row)
        results.append(row)
    return results


cpdef list sample(CircuitNode root, Py_ssize_t n_samples, object seed=None):
    """Draw ancestral samples from a circuit.

    Args:
        root: Circuit root node with propagated scope.
        n_samples: Number of independent samples (must be non-negative).
        seed: Optional RNG seed; defaults to a time-based seed when omitted.

    Returns:
        List of ``{var: value}`` assignment dicts, one per sample.
    """
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
    cdef CompiledGraph g = CompiledGraph()
    g.build(root)
    if g.has_fallback:
        return _sample_object(root, n_samples, rng)

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


# --- Compiled batched evaluation ----------------------------------------------

cdef class CompiledCircuit:
    """Flattened, cache-friendly representation for vectorized log-likelihood.

    Supports circuits whose leaves are ``FiniteDiscreteInputNode`` over a single
    variable (the only leaf family the OT/expectation queries use). The DAG is
    laid out in a single post-order schedule so a whole dataset can be scored in
    one ``nogil`` pass.
    """

    cdef vector[int] kinds            # per node
    cdef vector[size_t] child_off     # CSR offsets into children_flat
    cdef vector[size_t] children_flat
    cdef vector[double] sum_logw_flat  # aligned with children_flat (sum nodes)
    cdef vector[int] leaf_var          # scope var (input nodes), else -1
    cdef vector[int] leaf_card
    cdef vector[size_t] leaf_logpmf_off
    cdef vector[double] leaf_logpmf_flat
    cdef size_t n_nodes
    cdef size_t root_index
    cdef readonly list variables       # sorted scope variables

    def __init__(self, root):
        from sparc.circuit import Circuit
        if isinstance(root, Circuit):
            root = root.root
        if not isinstance(root, CircuitNode):
            raise TypeError("CompiledCircuit expects a CircuitNode or Circuit")
        cdef CircuitNode r = <CircuitNode>root
        if r.scope.size() == 0:
            r.propagate_scope()
        self._build(r)
        self.variables = sorted(r.scope)

    cdef void _build(self, CircuitNode root) except *:
        cdef list order = []
        cdef dict index_of = {}
        self._postorder(root, index_of, order)
        self.n_nodes = len(order)
        self.root_index = index_of[root.id]

        self.child_off.assign(self.n_nodes + 1, 0)
        self.leaf_var.assign(self.n_nodes, -1)
        self.leaf_card.assign(self.n_nodes, 0)
        self.leaf_logpmf_off.assign(self.n_nodes + 1, 0)
        self.kinds.assign(self.n_nodes, 0)

        cdef size_t n
        cdef size_t i
        cdef size_t k
        cdef CircuitNode node
        cdef InternalNode internal
        cdef SumNode s
        cdef FiniteDiscreteInputNode leaf
        cdef double pmf
        # First pass: kinds + counts.
        for n in range(self.n_nodes):
            node = <CircuitNode>order[n]
            self.kinds[n] = node.node_kind
            if node.node_kind == NODE_INPUT:
                if not isinstance(node, FiniteDiscreteInputNode):
                    raise TypeError(
                        "CompiledCircuit only supports FiniteDiscreteInputNode "
                        f"leaves, got {type(node).__name__}"
                    )
                leaf = <FiniteDiscreteInputNode>node
                self.leaf_card[n] = <int>leaf.support_size()
                self.child_off[n + 1] = 0
                self.leaf_logpmf_off[n + 1] = <size_t>leaf.support_size()
            else:
                internal = <InternalNode>node
                self.child_off[n + 1] = internal.num_children()
                self.leaf_logpmf_off[n + 1] = 0
        # Prefix sums -> CSR offsets.
        for n in range(self.n_nodes):
            self.child_off[n + 1] += self.child_off[n]
            self.leaf_logpmf_off[n + 1] += self.leaf_logpmf_off[n]
        self.children_flat.assign(self.child_off[self.n_nodes], 0)
        self.sum_logw_flat.assign(self.child_off[self.n_nodes], 0.0)
        self.leaf_logpmf_flat.assign(self.leaf_logpmf_off[self.n_nodes], 0.0)
        # Second pass: fill children / weights / leaf pmf.
        cdef size_t base
        cdef size_t lbase
        for n in range(self.n_nodes):
            node = <CircuitNode>order[n]
            if node.node_kind == NODE_INPUT:
                leaf = <FiniteDiscreteInputNode>node
                self.leaf_var[n] = leaf.scope_var_c()
                lbase = self.leaf_logpmf_off[n]
                for k in range(leaf.support_size()):
                    pmf = leaf.pmf_at(k)
                    self.leaf_logpmf_flat[lbase + k] = sp_safe_log(pmf)
            else:
                internal = <InternalNode>node
                base = self.child_off[n]
                for k in range(internal.num_children()):
                    self.children_flat[base + k] = index_of[internal.child_at(k).id]
                if node.node_kind == NODE_SUM:
                    s = <SumNode>node
                    for k in range(s.num_children()):
                        self.sum_logw_flat[base + k] = sp_safe_log(s.parameter_at(k))

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

    def log_likelihood(self, object data, object var_to_col=None):
        """Batched log-likelihood over a 2-D integer dataset.

        Args:
            data: Integer array of shape ``(n_samples, n_columns)``.
            var_to_col: Optional mapping from circuit variable index to column
                in ``data``. Defaults to identity over ``self.variables``.

        Returns:
            1-D float64 array of log-likelihoods, one per row of ``data``.
        """
        cdef object arr = np.ascontiguousarray(data, dtype=np.int32)
        if arr.ndim != 2:
            raise ValueError("data must be 2-D (n_samples, n_columns)")
        cdef Py_ssize_t n_rows = arr.shape[0]
        cdef Py_ssize_t n_cols = arr.shape[1]

        # Map each leaf's variable to a data column.
        cdef vector[int] leaf_col
        leaf_col.assign(self.n_nodes, -1)
        cdef size_t i
        cdef int var
        cdef int col
        for i in range(self.n_nodes):
            if self.kinds[i] == NODE_INPUT:
                var = self.leaf_var[i]
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

        cdef object out = np.empty(n_rows, dtype=np.float64)
        cdef double[::1] out_view = out
        cdef const int[:, ::1] data_view = arr
        self._score(data_view, leaf_col, out_view)
        return out

    cdef void _score(
        self,
        const int[:, ::1] data,
        const vector[int]& leaf_col,
        double[::1] out,
    ) except *:
        cdef Py_ssize_t n_rows = data.shape[0]
        cdef Py_ssize_t r
        cdef size_t n
        cdef size_t k
        cdef size_t start
        cdef size_t stop
        cdef int kind
        cdef int col
        cdef int value
        cdef int card
        cdef double acc
        cdef double max_log
        cdef double sum_exp
        cdef double term
        cdef vector[double] val
        val.assign(self.n_nodes, 0.0)
        with nogil:
            for r in range(n_rows):
                for n in range(self.n_nodes):
                    kind = self.kinds[n]
                    if kind == NODE_INPUT:
                        col = leaf_col[n]
                        value = data[r, col]
                        card = self.leaf_card[n]
                        if value < 0 or value >= card:
                            with gil:
                                raise ValueError(
                                    f"value {value} out of range [0, {card}) "
                                    f"in column {col}"
                                )
                        val[n] = self.leaf_logpmf_flat[self.leaf_logpmf_off[n] + value]
                    elif kind == NODE_PRODUCT:
                        start = self.child_off[n]
                        stop = self.child_off[n + 1]
                        acc = 0.0
                        for k in range(start, stop):
                            acc += val[self.children_flat[k]]
                        val[n] = acc
                    else:  # NODE_SUM
                        start = self.child_off[n]
                        stop = self.child_off[n + 1]
                        max_log = -1.0e308
                        for k in range(start, stop):
                            term = self.sum_logw_flat[k] + val[self.children_flat[k]]
                            if term > max_log:
                                max_log = term
                        if max_log <= -1.0e308:
                            val[n] = -1.0e308
                        else:
                            sum_exp = 0.0
                            for k in range(start, stop):
                                term = self.sum_logw_flat[k] + val[self.children_flat[k]]
                                sum_exp += exp(term - max_log)
                            val[n] = max_log + log(sum_exp)
                out[r] = val[self.root_index]
