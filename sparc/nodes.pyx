# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Core circuit node types for SparC.

The node layer is built for *extensibility through C-level virtual dispatch*:
``InputNode`` exposes a tiny ``cdef`` vtable (``prob_c`` / ``sample_into_c``) and
``FiniteDiscreteInputNode`` extends it with the discrete-support interface that
Wasserstein and expectation queries need (``support_size`` / ``pmf_at`` /
``scope_var_c``). A new leaf distribution is added by subclassing one of those
and overriding a couple of ``cdef`` methods -- no query, eval, or builder code
needs to change.
"""

from cpython.ref cimport PyObject

from libcpp cimport bool as cpp_bool
from libcpp.random cimport mt19937_64, uniform_real_distribution
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.math cimport exp, fabs, isfinite

cdef double PROB_TOL = 1e-6


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


# --- RNG ----------------------------------------------------------------------

cdef class RandomState:
    """Thin wrapper over a C++ Mersenne-Twister + U(0,1) for fast sampling."""

    def __cinit__(self, unsigned long long seed):
        self.rng = mt19937_64(seed)

    cdef inline double next_double(self) noexcept:
        return self.dist(self.rng)


# --- Evidence -----------------------------------------------------------------

cdef class Evidence:
    def __init__(self, object assignment=None):
        self._values.clear()
        if assignment is not None:
            self.update_from_mapping(assignment)

    cpdef void update_from_mapping(self, object assignment) except *:
        cdef object key
        cdef object value
        cdef int var
        cdef int outcome
        self._values.clear()
        for key, value in assignment.items():
            var = int(key)
            outcome = int(value)
            if var < 0:
                raise ValueError(f"variable index must be non-negative, got {var}")
            if outcome < 0:
                raise ValueError(f"outcome value must be non-negative, got {outcome}")
            self._values[var] = outcome

    cdef int get(self, int var) except *:
        if self._values.find(var) == self._values.end():
            raise ValueError(f"missing evidence for variable {var}")
        return self._values[var]

    cdef inline bint has(self, int var) noexcept:
        return self._values.find(var) != self._values.end()

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


# --- Base node ----------------------------------------------------------------

cdef class CircuitNode:
    def __init__(self, size_t id):
        self.id = id
        self.node_kind = -1
        self.scope.clear()

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _propagate_scope_impl"
        )

    cpdef void propagate_scope(self) except *:
        cdef unordered_set[size_t] visited
        self._propagate_scope_impl(visited)

    cpdef list scope_as_list(self):
        return sorted(self.scope)

    cpdef void set_scope_from_iterable(self, object indices) except *:
        cdef int v
        self.scope.clear()
        for v in indices:
            if v < 0:
                raise ValueError(f"scope indices must be non-negative, got {v}")
            self.scope.insert(<int>v)


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
    def __init__(self, size_t id, object children, object parameters):
        CircuitNode.__init__(self, id)
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
    def __init__(self, size_t id, object children):
        CircuitNode.__init__(self, id)
        self.node_kind = NODE_PRODUCT
        self._child_refs = []
        if len(children) < 1:
            raise ValueError("ProductNode must have at least one child")
        fill_children(self._children, self._child_refs, children)


# --- Leaf nodes ---------------------------------------------------------------

cdef class InputNode(CircuitNode):
    """Base leaf. Subclasses override the two ``cdef`` hooks below."""

    cdef double prob_c(self, Evidence ev) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement prob_c"
        )

    cdef void sample_into_c(self, RandomState rng, dict out) except *:
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
        cdef int value = ev.get(var)
        cdef Py_ssize_t card = self.cardinality()
        ev.validate_value(var, value, card)
        return self.pmf_at(<size_t>value)

    cdef void sample_into_c(self, RandomState rng, dict out) except *:
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
    def __init__(self, size_t id, int scope_var, object probabilities):
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        CircuitNode.__init__(self, id)
        self.node_kind = NODE_INPUT
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

    def __init__(self, size_t id, int scope_var, double p):
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        if not isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"bernoulli p must lie in [0, 1], got {p}")
        CircuitNode.__init__(self, id)
        self.node_kind = NODE_INPUT
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

    def __init__(self, size_t id, int scope_var, int value, object num_cats):
        cdef Py_ssize_t k = int(num_cats)
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        if k < 2:
            raise ValueError("indicator distribution must have at least 2 outcomes")
        if value < 0 or value >= k:
            raise ValueError(
                f"indicator value {value} out of range [0, {k})"
            )
        CircuitNode.__init__(self, id)
        self.node_kind = NODE_INPUT
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

    def __init__(self, size_t id, int scope_var, object value):
        cdef int v = 1 if value else 0
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        CircuitNode.__init__(self, id)
        self.node_kind = NODE_INPUT
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

    def __init__(self, size_t id, int scope_var, double mu, double s, object num_cats):
        cdef Py_ssize_t k = int(num_cats)
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        if k < 2:
            raise ValueError("discrete logistic must have at least 2 outcomes")
        if not isfinite(mu):
            raise ValueError("discrete logistic mu must be finite")
        if not isfinite(s) or s <= 0.0:
            raise ValueError(f"discrete logistic s must be positive, got {s}")
        CircuitNode.__init__(self, id)
        self.node_kind = NODE_INPUT
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
