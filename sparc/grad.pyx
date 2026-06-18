# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Differentiable mean log-likelihood (reverse-mode AD over the circuit DAG).

``GradBundle`` is the single gradient container used everywhere in SparC:
``value`` plus ``sum_grads`` / ``cat_grads`` dicts keyed by ``node.id``. Two-
circuit queries simply return a pair of ``GradBundle`` objects.
"""

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.math cimport exp

import numpy as np

from sparc._mathutils cimport sp_logsumexp, sp_safe_log
from sparc.nodes cimport (
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
    """Gradient container.

    Attributes
    ----------
    value : float
        The scalar objective value.
    sum_grads : dict[int, numpy.ndarray]
        ``SumNode.id`` -> gradient w.r.t. that node's ``parameters``.
    cat_grads : dict[int, numpy.ndarray]
        ``CategoricalInputNode.id`` -> gradient w.r.t. its ``probabilities``.

    Gradients are w.r.t. the linear parameters (no simplex projection); project
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
        cdef CategoricalInputNode cat
        cdef int var
        cdef int value
        cdef double p_v
        cdef object arr
        if not isinstance(node, CategoricalInputNode):
            return
        cat = <CategoricalInputNode>node
        var = cat.scope_var_c()
        value = self.evidence.get(var)
        p_v = cat.pmf_at(<size_t>value)
        if p_v <= 0.0:
            return
        arr = grad_arr(self.cat_grads, cat.id, cat.support_size())
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

    cdef tuple solve_dataset(self, CircuitNode root, list dataset):
        cdef Py_ssize_t n = len(dataset)
        cdef Py_ssize_t idx
        cdef double total_ll = 0.0
        cdef double inv_n
        cdef double ll
        cdef GradBundle grads
        if root.scope.size() == 0:
            raise ValueError(
                "root scope is empty; call propagate_scope() on the circuit first"
            )
        if n == 0:
            raise ValueError("dataset must contain at least one datapoint")
        inv_n = 1.0 / <double>n
        for idx in range(n):
            self.evidence = Evidence(dataset[idx])
            self.evidence.require_vars(root.scope)
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


def mean_log_likelihood_and_grad(CircuitNode root, object dataset):
    """Mean log-likelihood of a dataset and its gradient w.r.t. circuit params.

    Parameters
    ----------
    root : CircuitNode
        Circuit root with propagated scope.
    dataset : iterable of dict[int, int]
        Each datapoint is a full ``{var: value}`` assignment over the scope.

    Returns
    -------
    (mean_ll, grads) : tuple[float, GradBundle]
    """
    cdef _LLGradContext ctx = _LLGradContext()
    return ctx.solve_dataset(root, list(dataset))
