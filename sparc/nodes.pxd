from cpython.ref cimport PyObject
from libcpp.random cimport mt19937_64, uniform_real_distribution
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

# Node-kind tags. All leaf node types share NODE_INPUT; queries that need to
# inspect a leaf's distribution go through the FiniteDiscreteInputNode or
# ContinuousInputNode vtable.
cdef enum NodeKind:
    NODE_SUM = 0
    NODE_PRODUCT = 1
    NODE_INPUT = 2

# A circuit is all-discrete or all-continuous (no mixed input nodes).
cdef enum CircuitDomain:
    DOMAIN_UNSET = 0
    DOMAIN_DISCRETE = 1
    DOMAIN_CONTINUOUS = 2

cdef class RandomState:
    cdef mt19937_64 rng
    cdef uniform_real_distribution[double] dist
    cdef double spare
    cdef bint has_spare
    cdef double next_double(self) noexcept nogil
    cdef double next_normal(self) noexcept nogil

cdef class Evidence:
    cdef vector[int] _buf

    cdef int get(self, int var) except *
    cdef bint has(self, int var) noexcept
    cdef void require_vars(self, unordered_set[int]& scope_vars) except *
    cdef void validate_value(self, int var, int value, Py_ssize_t cardinality) except *
    cdef void init_dense(self, int width) except *
    cdef void set_var(self, int var, int value) noexcept

cdef class ContinuousEvidence:
    cdef vector[double] _buf

    cdef double get(self, int var) except *
    cdef bint has(self, int var) noexcept
    cdef void init_dense(self, int width) except *
    cdef void set_var(self, int var, double value) noexcept

cdef class CircuitNode:
    cdef readonly size_t id
    cdef readonly int node_kind
    cdef readonly int circuit_domain
    cdef unordered_set[int] scope

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *
    cdef void _ensure_scope(self) except *
    cpdef void propagate_scope(self) except *
    cpdef list scope_as_list(self)
    cpdef void set_scope_from_iterable(self, object indices) except *

cdef CircuitNode node_from_ptr(PyObject* obj)

cdef class InternalNode(CircuitNode):
    cdef vector[PyObject*] _children
    cdef list _child_refs

    cdef size_t num_children(self) noexcept
    cdef CircuitNode child_at(self, size_t index) except *
    cpdef list children(self)

cdef class SumNode(InternalNode):
    cdef vector[double] parameters

    cdef double parameter_at(self, size_t index) except *
    cpdef list parameters_list(self)
    cpdef void set_parameters_list(self, object parameters) except *

cdef class ProductNode(InternalNode):
    pass

# --- Extensible leaf interface -------------------------------------------------
# A new leaf type subclasses InputNode and overrides prob_c / sample_into_c.
# If it has finite discrete support and should participate in OT / expectation
# queries, it subclasses FiniteDiscreteInputNode and also overrides
# support_size / pmf_at / scope_var_c.

cdef class InputNode(CircuitNode):
    cdef double prob_c(self, Evidence ev) except *
    cdef void sample_into_c(self, RandomState rng, int* out) except *

cdef class FiniteDiscreteInputNode(InputNode):
    cdef size_t support_size(self) noexcept
    cdef double pmf_at(self, size_t index) except *
    cdef int scope_var_c(self) except *
    cpdef Py_ssize_t cardinality(self)

cdef class CategoricalInputNode(FiniteDiscreteInputNode):
    cdef vector[double] probabilities

    cpdef list probabilities_list(self)
    cpdef void set_probabilities_list(self, object probabilities) except *

cdef class BernoulliInputNode(FiniteDiscreteInputNode):
    cdef vector[double] probabilities

    cpdef double p(self)
    cpdef list probabilities_list(self)
    cpdef void set_probabilities_list(self, object probabilities) except *

cdef class IndicatorInputNode(FiniteDiscreteInputNode):
    cdef int value
    cdef size_t num_cats

    cpdef int value_at(self)
    cpdef Py_ssize_t num_categories(self)

cdef class LiteralInputNode(FiniteDiscreteInputNode):
    cdef int value

    cpdef int value_at(self)

cdef class DiscreteLogisticInputNode(FiniteDiscreteInputNode):
    cdef double mu
    cdef double s
    cdef size_t num_cats

    cpdef double mu_value(self)
    cpdef double s_value(self)
    cpdef Py_ssize_t num_categories(self)

cdef class ContinuousInputNode(InputNode):
    cdef int scope_var_c(self) except *
    cdef size_t n_params(self) noexcept
    cdef void params_into(self, double* out) except *
    cdef void set_params_from(self, const double* src, size_t n) except *
    cdef double log_density_c(self, ContinuousEvidence ev) except *
    cdef double density_c(self, ContinuousEvidence ev) except *
    cdef void sample_into_continuous(self, RandomState rng, double* out) except *
    cdef void log_density_backward_c(
        self, ContinuousEvidence ev, double g, double* g_self
    ) except *
    cdef double cw_w2sq_c(self, ContinuousInputNode other, double scale) except *
    cdef void cw_w2sq_backward_c(
        self, ContinuousInputNode other, double scale, double g,
        double* g_self, double* g_other,
    ) except *
    cdef double inner_product_c(self, ContinuousInputNode other) except *
    cdef void inner_product_backward_c(
        self, ContinuousInputNode other, double g,
        double* g_self, double* g_other,
    ) except *
    cdef double log_inner_product_c(self, ContinuousInputNode other) except *
    cdef void log_inner_product_backward_c(
        self, ContinuousInputNode other, double g,
        double* g_self, double* g_other,
    ) except *
    cdef double esd_c(self, double scale) except *
    cdef void esd_backward_c(self, double scale, double g, double* g_self) except *
    cpdef list parameters_list(self)
    cpdef void set_params_list(self, object params) except *

cdef class GaussianInputNode(ContinuousInputNode):
    cdef double mu
    cdef double sigma

    cpdef double mu_value(self)
    cpdef double sigma_value(self)
    cpdef void set_mu(self, double mu) except *
    cpdef void set_sigma(self, double sigma) except *
