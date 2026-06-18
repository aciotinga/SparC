# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
# cython: boundscheck=False, wraparound=False
"""Shared flattened circuit representation used by every query fast path.

A :class:`CompiledGraph` lays the (deduplicated) DAG out in a single post-order
schedule of plain C++ vectors -- no ``PyObject`` traversal -- so likelihood,
gradient, sampling and the optimal-transport queries can sweep the structure
``nogil``. Leaves are classified into a small closed set of tags
(:c:enum:`LeafKind`); their PMFs are precomputed into flat pools. Any leaf that
is not a recognized built-in marks the graph ``has_fallback`` so callers can
route to the object/vtable path with identical behavior.
"""

from libc.math cimport exp, INFINITY, log

from sparc.nodes cimport (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    DiscreteLogisticInputNode,
    FiniteDiscreteInputNode,
    IndicatorInputNode,
    InternalNode,
    LiteralInputNode,
    NODE_INPUT,
    NODE_SUM,
    SumNode,
)


cdef inline double sp_graph_sigmoid(double x) noexcept nogil:
    cdef double z
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    z = exp(x)
    return z / (1.0 + z)


cdef inline double _safe_log(double x) noexcept nogil:
    # Matches sparc._mathutils.sp_safe_log so the flattened log-space path is
    # numerically identical to the object/vtable path.
    if x > 0.0:
        return log(x)
    return -INFINITY


cdef class CompiledGraph:
    def __cinit__(self):
        self.node_objs = []
        self.variables = []
        self.has_fallback = False
        self.max_var = -1

    cdef void build(self, CircuitNode root) except *:
        # Callers are responsible for having propagated scope (the query layer
        # preserves the "scope is empty" contract by checking before building).
        cdef list order = []
        cdef dict index_of = {}
        self._postorder(root, index_of, order)
        self.n_nodes = len(order)
        self.root_index = index_of[root.id]
        self.node_objs = order

        self.kinds.assign(self.n_nodes, 0)
        self.child_off.assign(self.n_nodes + 1, 0)
        self.leaf_kind.assign(self.n_nodes, LEAF_FALLBACK)
        self.leaf_var.assign(self.n_nodes, -1)
        self.leaf_card.assign(self.n_nodes, 0)
        self.leaf_trainable.assign(self.n_nodes, 0)
        self.leaf_pmf_off.assign(self.n_nodes + 1, 0)
        self.node_ids.assign(self.n_nodes, 0)

        cdef size_t n
        cdef size_t k
        cdef CircuitNode node
        cdef InternalNode internal
        cdef FiniteDiscreteInputNode leaf
        cdef int kind
        cdef int support

        # First pass: kinds, counts, leaf classification.
        for n in range(self.n_nodes):
            node = <CircuitNode>order[n]
            self.node_ids[n] = node.id
            kind = node.node_kind
            self.kinds[n] = kind
            if kind == NODE_INPUT:
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

        self.children_flat.assign(self.child_off[self.n_nodes], 0)
        self.sum_w_flat.assign(self.child_off[self.n_nodes], 0.0)
        self.sum_logw_flat.assign(self.child_off[self.n_nodes], 0.0)
        self.leaf_pmf_flat.assign(self.leaf_pmf_off[self.n_nodes], 0.0)
        self.leaf_logpmf_flat.assign(self.leaf_pmf_off[self.n_nodes], 0.0)

        cdef size_t base
        cdef size_t lbase
        cdef SumNode s
        cdef double pmf

        # Second pass: children / weights / leaf pmf pools.
        for n in range(self.n_nodes):
            node = <CircuitNode>order[n]
            if node.node_kind == NODE_INPUT:
                if self.leaf_kind[n] != LEAF_FALLBACK:
                    leaf = <FiniteDiscreteInputNode>node
                    lbase = self.leaf_pmf_off[n]
                    for k in range(<size_t>self.leaf_card[n]):
                        pmf = leaf.pmf_at(k)
                        self.leaf_pmf_flat[lbase + k] = pmf
                        self.leaf_logpmf_flat[lbase + k] = _safe_log(pmf)
            else:
                internal = <InternalNode>node
                base = self.child_off[n]
                for k in range(internal.num_children()):
                    self.children_flat[base + k] = index_of[internal.child_at(k).id]
                if node.node_kind == NODE_SUM:
                    s = <SumNode>node
                    for k in range(s.num_children()):
                        self.sum_w_flat[base + k] = s.parameter_at(k)
                        self.sum_logw_flat[base + k] = _safe_log(s.parameter_at(k))

        self.variables = sorted(root.scope)
        self.max_var = self.variables[len(self.variables) - 1] if self.variables else -1

    cdef void _classify_leaf(self, CircuitNode node, size_t n) except *:
        """Tag a built-in leaf and record its scope/cardinality, or fall back.

        Only the closed set of built-in finite-discrete leaves participates in
        the flattened fast path; any other leaf (custom subclass, non-finite
        leaf) is tagged ``LEAF_FALLBACK`` so the owning graph routes to the
        object/vtable path with identical behavior.
        """
        cdef FiniteDiscreteInputNode leaf
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
            self.leaf_kind[n] = LEAF_FALLBACK
            self.has_fallback = True
            return
        leaf = <FiniteDiscreteInputNode>node
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
