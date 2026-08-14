# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Core circuit node types for SparC.

The node layer is built for *extensibility through C-level virtual dispatch*:
``InputNode`` exposes a tiny ``cdef`` vtable (``prob_c`` / ``sample_into_c``),
``FiniteDiscreteInputNode`` adds the discrete-support interface, and
``ContinuousInputNode`` adds density / closed-form query hooks. A circuit is
all-discrete or all-continuous; mixing the two leaf domains is rejected.
"""

from cpython.ref cimport PyObject

from libcpp cimport bool as cpp_bool
from libcpp.random cimport mt19937_64, uniform_real_distribution
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.math cimport NAN, exp, fabs, isfinite, log, sqrt

from sparc._continuous cimport (
    gaussian_density,
    gaussian_esd,
    gaussian_esd_dsigma,
    gaussian_inner_product,
    gaussian_inner_product_grad,
    gaussian_log_density,
    gaussian_log_inner_product,
    gaussian_log_inner_product_grad,
    gaussian_w2sq,
    gaussian_w2sq_grad,
)

import numpy as np
cimport numpy as cnp

cdef double PROB_TOL = 1e-6
cdef double SIGMA_FLOOR = 1e-12


cdef inline double _sigmoid(double x) noexcept:
    cdef double z
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    z = exp(x)
    return z / (1.0 + z)


cdef void validate_non_negative_scope(unordered_set[int]& scope):
    cdef int v
    for v in sorted(scope):
        if v < 0:
            raise ValueError(f"scope indices must be non-negative, got {v}")


cdef void validate_probabilities(const vector[double]& p, cpp_bool normalize_check):
    cdef size_t i
    cdef size_t n = p.size()
    cdef double total = 0.0
    cdef double x
    if n == 0:
        raise ValueError("probability vector must not be empty")
    for i in range(n):
        x = p[i]
        if not isfinite(x) or x < 0.0:
            raise ValueError("probabilities must be finite and non-negative")
        total += x
    if normalize_check and fabs(total - 1.0) > PROB_TOL:
        raise ValueError(f"probabilities must sum to 1, got {total}")


cdef void fill_vector_double(vector[double]& dest, object values) except *:
    cdef object item
    dest.clear()
    for item in values:
        dest.push_back(float(item))


cdef size_t _next_node_id = 0


cdef size_t _alloc_node_id() except *:
    global _next_node_id
    cdef size_t nid = _next_node_id
    _next_node_id += 1
    return nid


cdef void _claim_node_id(size_t nid) noexcept:
    global _next_node_id
    if nid >= _next_node_id:
        _next_node_id = nid + 1


cdef size_t _resolve_node_id(object id) except *:
    if id is None:
        return _alloc_node_id()
    cdef size_t nid = <size_t>int(id)
    _claim_node_id(nid)
    return nid


def alloc_node_id():
    """Allocate a fresh node id (for internal builders)."""
    return int(_alloc_node_id())


def reset_node_id_allocator():
    """Reset the global node-id counter (for tests)."""
    global _next_node_id
    _next_node_id = 0


# --- RNG ----------------------------------------------------------------------

cdef class RandomState:
    """Thin wrapper over a C++ Mersenne-Twister + U(0,1) for fast sampling."""

    def __cinit__(self, unsigned long long seed):
        self.rng = mt19937_64(seed)
        self.has_spare = False
        self.spare = 0.0

    cdef inline double next_double(self) noexcept nogil:
        return self.dist(self.rng)

    cdef double next_normal(self) noexcept nogil:
        cdef double u
        cdef double v
        cdef double s
        cdef double mul
        if self.has_spare:
            self.has_spare = False
            return self.spare
        while True:
            u = 2.0 * self.next_double() - 1.0
            v = 2.0 * self.next_double() - 1.0
            s = u * u + v * v
            if s > 0.0 and s < 1.0:
                break
        mul = sqrt(-2.0 * log(s) / s)
        self.spare = v * mul
        self.has_spare = True
        return u * mul


# --- Evidence -----------------------------------------------------------------

cdef class Evidence:
    def __init__(self, cnp.ndarray row not None):
        if row.ndim != 1:
            raise ValueError("Evidence array must be 1-D")
        cdef Py_ssize_t i
        cdef int val
        self._buf.resize(<size_t>row.shape[0])
        for i in range(row.shape[0]):
            val = int(row[i])
            if val < 0:
                raise ValueError(f"outcome value must be non-negative, got {val}")
            self._buf[<size_t>i] = val

    cdef void init_dense(self, int width) except *:
        if width < 0:
            raise ValueError(f"evidence width must be non-negative, got {width}")
        self._buf.assign(<size_t>width, -1)

    cdef void set_var(self, int var, int value) noexcept:
        if var >= 0 and <size_t>var < self._buf.size():
            self._buf[<size_t>var] = value

    cdef int get(self, int var) except *:
        if var < 0 or <size_t>var >= self._buf.size():
            raise ValueError(f"missing evidence for variable {var}")
        cdef int val = self._buf[<size_t>var]
        if val < 0:
            raise ValueError(f"missing evidence for variable {var}")
        return val

    cdef inline bint has(self, int var) noexcept:
        if var < 0 or <size_t>var >= self._buf.size():
            return False
        return self._buf[<size_t>var] >= 0

    cdef void require_vars(self, unordered_set[int]& scope_vars) except *:
        cdef int v
        for v in sorted(scope_vars):
            self.get(v)

    cdef void validate_value(self, int var, int value, Py_ssize_t cardinality) except *:
        if value < 0 or value >= cardinality:
            raise ValueError(
                f"evidence for variable {var}: outcome {value} out of range "
                f"[0, {cardinality})"
            )


# --- Continuous evidence ------------------------------------------------------

cdef class ContinuousEvidence:
    """Dense float evidence; NaN marks a marginalized (missing) variable."""

    def __init__(self, cnp.ndarray row not None):
        if row.ndim != 1:
            raise ValueError("Evidence array must be 1-D")
        cdef Py_ssize_t i
        cdef double val
        self._buf.resize(<size_t>row.shape[0])
        for i in range(row.shape[0]):
            val = float(row[i])
            self._buf[<size_t>i] = val

    cdef void init_dense(self, int width) except *:
        if width < 0:
            raise ValueError(f"evidence width must be non-negative, got {width}")
        self._buf.assign(<size_t>width, NAN)

    cdef void set_var(self, int var, double value) noexcept:
        if var >= 0 and <size_t>var < self._buf.size():
            self._buf[<size_t>var] = value

    cdef double get(self, int var) except *:
        if var < 0 or <size_t>var >= self._buf.size():
            raise ValueError(f"missing evidence for variable {var}")
        cdef double val = self._buf[<size_t>var]
        if not isfinite(val):
            raise ValueError(f"missing evidence for variable {var}")
        return val

    cdef inline bint has(self, int var) noexcept:
        if var < 0 or <size_t>var >= self._buf.size():
            return False
        return isfinite(self._buf[<size_t>var])


# --- Base node ----------------------------------------------------------------

cdef class CircuitNode:
    def __init__(self, size_t id):
        self.id = id
        self.node_kind = -1
        self.circuit_domain = DOMAIN_UNSET
        self.scope.clear()

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _propagate_scope_impl"
        )

    cpdef void propagate_scope(self) except *:
        cdef unordered_set[size_t] visited
        self._propagate_scope_impl(visited)

    cpdef list scope_as_list(self):
        self._ensure_scope()
        return sorted(self.scope)

    cpdef void set_scope_from_iterable(self, object indices) except *:
        cdef int v
        self.scope.clear()
        for v in indices:
            if v < 0:
                raise ValueError(f"scope indices must be non-negative, got {v}")
            self.scope.insert(<int>v)

    cdef void _ensure_scope(self) except *:
        if self.scope.size() == 0:
            self.propagate_scope()

    def likelihood(self, data, var_to_col=None):
        from sparc.eval import likelihood
        self._ensure_scope()
        return likelihood(self, data, var_to_col)

    def log_likelihood(self, data, var_to_col=None):
        from sparc.eval import log_likelihood
        self._ensure_scope()
        return log_likelihood(self, data, var_to_col)

    def mean_log_likelihood_and_grad(self, dataset, var_to_col=None):
        from sparc.grad import mean_log_likelihood_and_grad
        self._ensure_scope()
        return mean_log_likelihood_and_grad(self, dataset, var_to_col)

    def sample(self, n_samples, seed=None, *, differentiable=False):
        from sparc.eval import sample
        self._ensure_scope()
        return sample(
            self,
            n_samples,
            seed,
            differentiable=differentiable,
        )

    def compile(self):
        from sparc._graph import CompiledCircuit
        self._ensure_scope()
        return CompiledCircuit(self)

    def clone(self):
        from sparc.node_clone import clone_node
        return clone_node(self, {})

    def save(self, path, *, indent=2, encoding="utf-8"):
        from sparc.io.serializer import CircuitSerializer
        CircuitSerializer.save(self, path, indent=indent, encoding=encoding)

    @classmethod
    def load(cls, path, *, encoding="utf-8"):
        from sparc.io.serializer import CircuitSerializer
        return CircuitSerializer.load(path, encoding=encoding)

    def cw_distance(self, other, metric_p=1.0, scale_factor=1.0, metric=None):
        from sparc.queries.cw import cw_distance
        return cw_distance(self, other, metric_p, scale_factor, metric)

    def cw_distance_and_grad(self, other, metric_p=1.0, scale_factor=1.0, metric=None):
        from sparc.queries.cw import cw_distance_and_grad
        return cw_distance_and_grad(self, other, metric_p, scale_factor, metric)

    def gcw_crossterm(
        self,
        other,
        metric_p=1.0,
        scale_factor_1=1.0,
        scale_factor_2=1.0,
        metric1=None,
        metric2=None,
    ):
        from sparc.queries.gcw import gcw_crossterm
        return gcw_crossterm(
            self, other, metric_p, scale_factor_1, scale_factor_2, metric1, metric2
        )

    def gcw_crossterm_and_grad(
        self,
        other,
        metric_p=1.0,
        scale_factor_1=1.0,
        scale_factor_2=1.0,
        metric1=None,
        metric2=None,
    ):
        from sparc.queries.gcw import gcw_crossterm_and_grad
        return gcw_crossterm_and_grad(
            self, other, metric_p, scale_factor_1, scale_factor_2, metric1, metric2
        )

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

    def expected_squared_distance(self, metric_p=1.0, scale_factor=1.0, metric=None):
        from sparc.queries.esd import expected_squared_distance
        return expected_squared_distance(self, metric_p, scale_factor, metric)

    def expected_squared_distance_and_grad(
        self, metric_p=1.0, scale_factor=1.0, metric=None
    ):
        from sparc.queries.esd import expected_squared_distance_and_grad
        return expected_squared_distance_and_grad(self, metric_p, scale_factor, metric)


cdef CircuitNode node_from_ptr(PyObject* obj):
    return <CircuitNode><object>obj


cdef void scope_union_from(unordered_set[int]& dest, unordered_set[int]& src):
    dest.insert(src.begin(), src.end())


cdef void scope_union_from_ptrs(
    unordered_set[int]& dest,
    const vector[PyObject*]& children,
):
    cdef PyObject* raw
    cdef CircuitNode node
    cdef size_t i
    cdef size_t n = children.size()
    for i in range(n):
        raw = children[i]
        if raw != NULL:
            node = node_from_ptr(raw)
            scope_union_from(dest, node.scope)


cdef void fill_children(
    vector[PyObject*]& ptrs,
    list refs,
    object children,
) except *:
    cdef object child
    ptrs.clear()
    refs.clear()
    for child in children:
        if not isinstance(child, CircuitNode):
            raise TypeError("children must be CircuitNode instances")
        refs.append(child)
        ptrs.push_back(<PyObject*>child)


cdef void _set_domain_from_children(InternalNode node) except *:
    cdef size_t i
    cdef size_t n = node.num_children()
    cdef CircuitNode child
    cdef int d
    if n == 0:
        raise ValueError("internal node must have at least one child")
    child = node.child_at(0)
    d = child.circuit_domain
    if d == DOMAIN_UNSET:
        raise ValueError(
            f"child {child.id} has unset circuit domain; leaves must declare "
            "discrete or continuous"
        )
    for i in range(1, n):
        child = node.child_at(i)
        if child.circuit_domain != d:
            raise ValueError(
                "cannot mix discrete and continuous input nodes in one circuit"
            )
    node.circuit_domain = d


# --- Internal nodes -----------------------------------------------------------

cdef class InternalNode(CircuitNode):
    """Shared machinery for nodes that have children (sum, product)."""

    cdef inline size_t num_children(self) noexcept:
        return self._children.size()

    cdef CircuitNode child_at(self, size_t index) except *:
        if index >= self._children.size():
            raise IndexError(f"child index {index} out of range")
        return node_from_ptr(self._children[index])

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        cdef pair[unordered_set[size_t].iterator, cpp_bool] inserted
        cdef size_t i
        cdef size_t n
        cdef CircuitNode child
        inserted = visited.insert(self.id)
        if not inserted.second:
            return
        n = self._children.size()
        for i in range(n):
            child = self.child_at(i)
            child._propagate_scope_impl(visited)
        self.scope.clear()
        scope_union_from_ptrs(self.scope, self._children)
        validate_non_negative_scope(self.scope)

    cpdef list children(self):
        return list(self._child_refs)


cdef class SumNode(InternalNode):
    """Mixture node: weighted sum over child sub-circuits.

    Args:
        children: List of child :class:`CircuitNode` instances.
        parameters: Non-negative mixture weights summing to 1.
        id: Optional unique node identifier (auto-assigned when omitted).
    """

    def __init__(self, object children, object parameters, *, object id=None):
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_SUM
        self._child_refs = []
        if len(children) < 1:
            raise ValueError("SumNode must have at least one child")
        if len(children) != len(parameters):
            raise ValueError(
                f"children and parameters length mismatch: "
                f"{len(children)} vs {len(parameters)}"
            )
        fill_children(self._children, self._child_refs, children)
        _set_domain_from_children(self)
        fill_vector_double(self.parameters, parameters)
        validate_probabilities(self.parameters, True)

    cdef double parameter_at(self, size_t index) except *:
        if index >= self.parameters.size():
            raise IndexError(f"parameter index {index} out of range")
        return self.parameters[index]

    cpdef list parameters_list(self):
        cdef size_t i
        cdef size_t n = self.parameters.size()
        cdef list out = []
        for i in range(n):
            out.append(self.parameters[i])
        return out

    cpdef void set_parameters_list(self, object parameters) except *:
        cdef size_t n_old = self.parameters.size()
        cdef object params_list = list(parameters)
        if len(params_list) != n_old:
            raise ValueError(
                f"parameter length mismatch: expected {n_old}, "
                f"got {len(params_list)}"
            )
        fill_vector_double(self.parameters, params_list)
        validate_probabilities(self.parameters, True)


cdef class ProductNode(InternalNode):
    """Factorization node: product over child sub-circuits with disjoint scopes.

    Args:
        children: List of child :class:`CircuitNode` instances with pairwise
            disjoint scopes.
        id: Optional unique node identifier (auto-assigned when omitted).
    """

    def __init__(self, object children, *, object id=None):
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_PRODUCT
        self._child_refs = []
        if len(children) < 1:
            raise ValueError("ProductNode must have at least one child")
        fill_children(self._children, self._child_refs, children)
        _set_domain_from_children(self)


# --- Leaf nodes ---------------------------------------------------------------

cdef class InputNode(CircuitNode):
    """Base leaf. Subclasses override the two ``cdef`` hooks below."""

    cdef double prob_c(self, Evidence ev) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement prob_c"
        )

    cdef void sample_into_c(self, RandomState rng, int* out) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement sample_into_c"
        )

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        cdef pair[unordered_set[size_t].iterator, cpp_bool] inserted
        inserted = visited.insert(self.id)
        if not inserted.second:
            return
        if self.scope.size() == 0:
            raise ValueError(f"InputNode {self.id} has empty scope")
        validate_non_negative_scope(self.scope)


cdef class FiniteDiscreteInputNode(InputNode):
    """Leaf with finite discrete support over a single variable.

    This is the interface required by Wasserstein (GCW / CW) and expectation
    queries: they only need the PMF and the scope variable.
    """

    cdef size_t support_size(self) noexcept:
        return 0

    cdef double pmf_at(self, size_t index) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement pmf_at"
        )

    cdef int scope_var_c(self) except *:
        cdef int v
        if self.scope.size() != 1:
            raise ValueError(
                f"{type(self).__name__} {self.id} must have scope of size 1"
            )
        for v in sorted(self.scope):
            return v
        raise ValueError(f"{type(self).__name__} {self.id} has empty scope")

    cpdef Py_ssize_t cardinality(self):
        return <Py_ssize_t>self.support_size()

    cdef double prob_c(self, Evidence ev) except *:
        cdef int var = self.scope_var_c()
        if not ev.has(var):
            return 1.0
        cdef int value = ev.get(var)
        cdef Py_ssize_t card = self.cardinality()
        ev.validate_value(var, value, card)
        return self.pmf_at(<size_t>value)

    cdef void sample_into_c(self, RandomState rng, int* out) except *:
        cdef int var = self.scope_var_c()
        cdef size_t n = self.support_size()
        cdef size_t i
        cdef double u = rng.next_double()
        cdef double cum = 0.0
        for i in range(n):
            cum += self.pmf_at(i)
            if u < cum:
                out[var] = <int>i
                return
        out[var] = <int>(n - 1)


cdef class CategoricalInputNode(FiniteDiscreteInputNode):
    """Categorical leaf over a single variable.

    Args:
        scope_var: Variable index (non-negative integer).
        probabilities: PMF over at least two outcomes, summing to 1.
        id: Optional unique node identifier (auto-assigned when omitted).
    """

    def __init__(self, int scope_var, object probabilities, *, object id=None):
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_INPUT
        self.circuit_domain = DOMAIN_DISCRETE
        self.scope.clear()
        self.scope.insert(scope_var)
        fill_vector_double(self.probabilities, probabilities)
        if self.probabilities.size() < 2:
            raise ValueError("categorical distribution must have at least 2 outcomes")
        validate_probabilities(self.probabilities, True)

    cdef inline size_t support_size(self) noexcept:
        return self.probabilities.size()

    cdef double pmf_at(self, size_t index) except *:
        if index >= self.probabilities.size():
            raise IndexError(
                f"outcome index {index} out of range for node {self.id}"
            )
        return self.probabilities[index]

    cpdef list probabilities_list(self):
        cdef size_t i
        cdef size_t n = self.probabilities.size()
        cdef list out = []
        for i in range(n):
            out.append(self.probabilities[i])
        return out

    cpdef void set_probabilities_list(self, object probabilities) except *:
        fill_vector_double(self.probabilities, probabilities)
        if self.probabilities.size() < 2:
            raise ValueError("categorical distribution must have at least 2 outcomes")
        validate_probabilities(self.probabilities, True)


cdef class BernoulliInputNode(FiniteDiscreteInputNode):
    """Binary leaf with success probability ``p`` (support ``{0, 1}``).

    Internally stored as a 2-outcome PMF ``[1 - p, p]`` so it reuses the same
    simplex-projected gradient path as the categorical leaf.
    """

    def __init__(self, int scope_var, double p, *, object id=None):
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        if not isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"bernoulli p must lie in [0, 1], got {p}")
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_INPUT
        self.circuit_domain = DOMAIN_DISCRETE
        self.scope.clear()
        self.scope.insert(scope_var)
        self.probabilities.clear()
        self.probabilities.push_back(1.0 - p)
        self.probabilities.push_back(p)

    cdef inline size_t support_size(self) noexcept:
        return 2

    cdef double pmf_at(self, size_t index) except *:
        if index >= self.probabilities.size():
            raise IndexError(
                f"outcome index {index} out of range for node {self.id}"
            )
        return self.probabilities[index]

    cpdef double p(self):
        return self.probabilities[1]

    cpdef list probabilities_list(self):
        return [self.probabilities[0], self.probabilities[1]]

    cpdef void set_probabilities_list(self, object probabilities) except *:
        cdef list vals = list(probabilities)
        if len(vals) != 2:
            raise ValueError("bernoulli distribution requires exactly 2 outcomes")
        fill_vector_double(self.probabilities, vals)
        validate_probabilities(self.probabilities, True)


cdef class IndicatorInputNode(FiniteDiscreteInputNode):
    """Deterministic leaf placing all mass on a single outcome ``value``.

    Useful as a clamped/observed leaf over a variable with ``num_cats`` states.
    Carries no trainable parameters.
    """

    def __init__(self, int scope_var, int value, object num_cats, *, object id=None):
        cdef Py_ssize_t k = int(num_cats)
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        if k < 2:
            raise ValueError("indicator distribution must have at least 2 outcomes")
        if value < 0 or value >= k:
            raise ValueError(
                f"indicator value {value} out of range [0, {k})"
            )
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_INPUT
        self.circuit_domain = DOMAIN_DISCRETE
        self.scope.clear()
        self.scope.insert(scope_var)
        self.value = value
        self.num_cats = <size_t>k

    cdef inline size_t support_size(self) noexcept:
        return self.num_cats

    cdef double pmf_at(self, size_t index) except *:
        if index >= self.num_cats:
            raise IndexError(
                f"outcome index {index} out of range for node {self.id}"
            )
        if <int>index == self.value:
            return 1.0
        return 0.0

    cpdef int value_at(self):
        return self.value

    cpdef Py_ssize_t num_categories(self):
        return <Py_ssize_t>self.num_cats


cdef class LiteralInputNode(FiniteDiscreteInputNode):
    """Deterministic boolean leaf (support ``{0, 1}``) clamped to ``value``."""

    def __init__(self, int scope_var, object value, *, object id=None):
        cdef int v = 1 if value else 0
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_INPUT
        self.circuit_domain = DOMAIN_DISCRETE
        self.scope.clear()
        self.scope.insert(scope_var)
        self.value = v

    cdef inline size_t support_size(self) noexcept:
        return 2

    cdef double pmf_at(self, size_t index) except *:
        if index >= 2:
            raise IndexError(
                f"outcome index {index} out of range for node {self.id}"
            )
        if <int>index == self.value:
            return 1.0
        return 0.0

    cpdef int value_at(self):
        return self.value


cdef class DiscreteLogisticInputNode(FiniteDiscreteInputNode):
    """Logistic distribution discretized over integer bins ``0 .. num_cats-1``.

    The PMF of bin ``k`` is the logistic CDF mass over ``[k - 0.5, k + 0.5)``
    with the two boundary bins absorbing the lower and upper tails, so the bins
    always sum to exactly one. Parameters are the location ``mu`` and scale
    ``s``; they describe a continuous shape sampled onto a finite grid and are
    treated as fixed by the simplex optimizer (the leaf remains fully usable for
    likelihood, sampling, and all transport / expectation queries).
    """

    def __init__(
        self, int scope_var, double mu, double s, object num_cats, *, object id=None
    ):
        cdef Py_ssize_t k = int(num_cats)
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        if k < 2:
            raise ValueError("discrete logistic must have at least 2 outcomes")
        if not isfinite(mu):
            raise ValueError("discrete logistic mu must be finite")
        if not isfinite(s) or s <= 0.0:
            raise ValueError(f"discrete logistic s must be positive, got {s}")
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_INPUT
        self.circuit_domain = DOMAIN_DISCRETE
        self.scope.clear()
        self.scope.insert(scope_var)
        self.mu = mu
        self.s = s
        self.num_cats = <size_t>k

    cdef inline size_t support_size(self) noexcept:
        return self.num_cats

    cdef double pmf_at(self, size_t index) except *:
        cdef double lo
        cdef double hi
        if index >= self.num_cats:
            raise IndexError(
                f"outcome index {index} out of range for node {self.id}"
            )
        if index == 0:
            hi = (0.5 - self.mu) / self.s
            return _sigmoid(hi)
        if index == self.num_cats - 1:
            lo = (<double>index - 0.5 - self.mu) / self.s
            return 1.0 - _sigmoid(lo)
        lo = (<double>index - 0.5 - self.mu) / self.s
        hi = (<double>index + 0.5 - self.mu) / self.s
        return _sigmoid(hi) - _sigmoid(lo)

    cpdef double mu_value(self):
        return self.mu

    cpdef double s_value(self):
        return self.s

    cpdef Py_ssize_t num_categories(self):
        return <Py_ssize_t>self.num_cats


cdef class ContinuousInputNode(InputNode):
    """Base continuous leaf. Subclasses override density / query hooks."""

    cdef int scope_var_c(self) except *:
        cdef int v
        if self.scope.size() != 1:
            raise ValueError(
                f"{type(self).__name__} {self.id} must have scope of size 1"
            )
        for v in sorted(self.scope):
            return v
        raise ValueError(f"{type(self).__name__} {self.id} has empty scope")

    cdef size_t n_params(self) noexcept:
        return 0

    cdef void params_into(self, double* out) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement params_into"
        )

    cdef void set_params_from(self, const double* src, size_t n) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement set_params_from"
        )

    cdef double log_density_c(self, ContinuousEvidence ev) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement log_density_c"
        )

    cdef double density_c(self, ContinuousEvidence ev) except *:
        cdef double lp = self.log_density_c(ev)
        if not isfinite(lp):
            return 0.0
        return exp(lp)

    cdef void sample_into_continuous(self, RandomState rng, double* out) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement sample_into_continuous"
        )

    cdef void log_density_backward_c(
        self, ContinuousEvidence ev, double g, double* g_self
    ) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement log_density_backward_c"
        )

    cdef double cw_w2sq_c(self, ContinuousInputNode other, double scale) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement cw_w2sq_c"
        )

    cdef void cw_w2sq_backward_c(
        self, ContinuousInputNode other, double scale, double g,
        double* g_self, double* g_other,
    ) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement cw_w2sq_backward_c"
        )

    cdef double inner_product_c(self, ContinuousInputNode other) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement inner_product_c"
        )

    cdef void inner_product_backward_c(
        self, ContinuousInputNode other, double g,
        double* g_self, double* g_other,
    ) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement inner_product_backward_c"
        )

    cdef double log_inner_product_c(self, ContinuousInputNode other) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement log_inner_product_c"
        )

    cdef void log_inner_product_backward_c(
        self, ContinuousInputNode other, double g,
        double* g_self, double* g_other,
    ) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement log_inner_product_backward_c"
        )

    cdef double esd_c(self, double scale) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement esd_c"
        )

    cdef void esd_backward_c(self, double scale, double g, double* g_self) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement esd_backward_c"
        )

    cpdef list parameters_list(self):
        cdef size_t n = self.n_params()
        cdef size_t i
        cdef vector[double] buf
        cdef list out = []
        buf.resize(n)
        if n > 0:
            self.params_into(buf.data())
        for i in range(n):
            out.append(buf[i])
        return out

    cpdef void set_params_list(self, object params) except *:
        cdef vector[double] buf
        fill_vector_double(buf, params)
        self.set_params_from(buf.data(), buf.size())


cdef class GaussianInputNode(ContinuousInputNode):
    """Univariate Gaussian leaf :math:`N(\\mu, \\sigma^2)`.

    Args:
        scope_var: Variable index (non-negative integer).
        mu: Location.
        sigma: Positive standard deviation.
        id: Optional unique node identifier (auto-assigned when omitted).
    """

    def __init__(self, int scope_var, double mu=0.0, double sigma=1.0, *, object id=None):
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        if not isfinite(mu):
            raise ValueError("gaussian mu must be finite")
        if not isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"gaussian sigma must be positive, got {sigma}")
        cdef size_t node_id = _resolve_node_id(id)
        CircuitNode.__init__(self, node_id)
        self.node_kind = NODE_INPUT
        self.circuit_domain = DOMAIN_CONTINUOUS
        self.scope.clear()
        self.scope.insert(scope_var)
        self.mu = mu
        self.sigma = sigma

    cdef inline size_t n_params(self) noexcept:
        return 2

    cdef void params_into(self, double* out) except *:
        out[0] = self.mu
        out[1] = self.sigma

    cdef void set_params_from(self, const double* src, size_t n) except *:
        if n != 2:
            raise ValueError(f"gaussian expects 2 parameters, got {n}")
        if not isfinite(src[0]):
            raise ValueError("gaussian mu must be finite")
        if not isfinite(src[1]) or src[1] <= 0.0:
            raise ValueError(f"gaussian sigma must be positive, got {src[1]}")
        self.mu = src[0]
        self.sigma = src[1]

    cdef double log_density_c(self, ContinuousEvidence ev) except *:
        cdef int var = self.scope_var_c()
        if not ev.has(var):
            return 0.0
        return gaussian_log_density(ev.get(var), self.mu, self.sigma)

    cdef double density_c(self, ContinuousEvidence ev) except *:
        cdef int var = self.scope_var_c()
        if not ev.has(var):
            return 1.0
        return gaussian_density(ev.get(var), self.mu, self.sigma)

    cdef void sample_into_continuous(self, RandomState rng, double* out) except *:
        cdef int var = self.scope_var_c()
        out[var] = self.mu + self.sigma * rng.next_normal()

    cdef void log_density_backward_c(
        self, ContinuousEvidence ev, double g, double* g_self
    ) except *:
        cdef int var = self.scope_var_c()
        cdef double x
        cdef double z
        if not ev.has(var):
            g_self[0] = 0.0
            g_self[1] = 0.0
            return
        x = ev.get(var)
        z = x - self.mu
        g_self[0] = g * z / (self.sigma * self.sigma)
        g_self[1] = g * (-1.0 / self.sigma + z * z / (self.sigma * self.sigma * self.sigma))

    cdef double cw_w2sq_c(self, ContinuousInputNode other, double scale) except *:
        cdef GaussianInputNode g
        if not isinstance(other, GaussianInputNode):
            raise TypeError(
                "incompatible continuous leaves: "
                f"{type(self).__name__} vs {type(other).__name__}"
            )
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        g = <GaussianInputNode>other
        return gaussian_w2sq(self.mu, self.sigma, g.mu, g.sigma, scale)

    cdef void cw_w2sq_backward_c(
        self, ContinuousInputNode other, double scale, double g,
        double* g_self, double* g_other,
    ) except *:
        cdef GaussianInputNode o
        cdef double dmu1, ds1, dmu2, ds2
        if not isinstance(other, GaussianInputNode):
            raise TypeError(
                "incompatible continuous leaves: "
                f"{type(self).__name__} vs {type(other).__name__}"
            )
        o = <GaussianInputNode>other
        gaussian_w2sq_grad(
            self.mu, self.sigma, o.mu, o.sigma, scale, g,
            &dmu1, &ds1, &dmu2, &ds2,
        )
        g_self[0] = dmu1
        g_self[1] = ds1
        g_other[0] = dmu2
        g_other[1] = ds2

    cdef double inner_product_c(self, ContinuousInputNode other) except *:
        cdef GaussianInputNode g
        if not isinstance(other, GaussianInputNode):
            raise TypeError(
                "incompatible continuous leaves: "
                f"{type(self).__name__} vs {type(other).__name__}"
            )
        g = <GaussianInputNode>other
        return gaussian_inner_product(self.mu, self.sigma, g.mu, g.sigma)

    cdef void inner_product_backward_c(
        self, ContinuousInputNode other, double g,
        double* g_self, double* g_other,
    ) except *:
        cdef GaussianInputNode o
        cdef double dmu1, ds1, dmu2, ds2
        if not isinstance(other, GaussianInputNode):
            raise TypeError(
                "incompatible continuous leaves: "
                f"{type(self).__name__} vs {type(other).__name__}"
            )
        o = <GaussianInputNode>other
        gaussian_inner_product_grad(
            self.mu, self.sigma, o.mu, o.sigma, g,
            &dmu1, &ds1, &dmu2, &ds2,
        )
        g_self[0] = dmu1
        g_self[1] = ds1
        g_other[0] = dmu2
        g_other[1] = ds2

    cdef double log_inner_product_c(self, ContinuousInputNode other) except *:
        cdef GaussianInputNode g
        if not isinstance(other, GaussianInputNode):
            raise TypeError(
                "incompatible continuous leaves: "
                f"{type(self).__name__} vs {type(other).__name__}"
            )
        g = <GaussianInputNode>other
        return gaussian_log_inner_product(self.mu, self.sigma, g.mu, g.sigma)

    cdef void log_inner_product_backward_c(
        self, ContinuousInputNode other, double g,
        double* g_self, double* g_other,
    ) except *:
        cdef GaussianInputNode o
        cdef double dmu1, ds1, dmu2, ds2
        if not isinstance(other, GaussianInputNode):
            raise TypeError(
                "incompatible continuous leaves: "
                f"{type(self).__name__} vs {type(other).__name__}"
            )
        o = <GaussianInputNode>other
        gaussian_log_inner_product_grad(
            self.mu, self.sigma, o.mu, o.sigma, g,
            &dmu1, &ds1, &dmu2, &ds2,
        )
        g_self[0] = dmu1
        g_self[1] = ds1
        g_other[0] = dmu2
        g_other[1] = ds2

    cdef double esd_c(self, double scale) except *:
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        return gaussian_esd(self.sigma, scale)

    cdef void esd_backward_c(self, double scale, double g, double* g_self) except *:
        g_self[0] = 0.0
        g_self[1] = gaussian_esd_dsigma(self.sigma, scale, g)

    cpdef double mu_value(self):
        return self.mu

    cpdef double sigma_value(self):
        return self.sigma

    cpdef void set_mu(self, double mu) except *:
        if not isfinite(mu):
            raise ValueError("gaussian mu must be finite")
        self.mu = mu

    cpdef void set_sigma(self, double sigma) except *:
        if not isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"gaussian sigma must be positive, got {sigma}")
        self.sigma = sigma

