from libc.stdint cimport uint64_t
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp

from sparc.nodes cimport CircuitNode, RandomState

# Leaf-kind tags for specialized nogil paths. Unknown FiniteDiscreteInputNode
# subclasses use LEAF_GENERIC (PMF materialized via pmf_at at compile time).
cdef enum LeafKind:
    LEAF_CATEGORICAL = 0
    LEAF_BERNOULLI = 1
    LEAF_INDICATOR = 2
    LEAF_LITERAL = 3
    LEAF_DISCRETE_LOGISTIC = 4
    LEAF_GENERIC = 5


cdef class CompiledCircuit:
    # --- topology (post-order; children appear before parents) ---------------
    cdef vector[int] kinds
    cdef vector[size_t] child_off
    cdef vector[size_t] children_flat
    cdef vector[double] sum_w_flat
    cdef vector[double] sum_logw_flat

    # --- leaf descriptors (valid where kinds[n] == NODE_INPUT) ---------------
    cdef vector[int] leaf_kind
    cdef vector[int] leaf_var
    cdef vector[int] leaf_card
    cdef vector[char] leaf_trainable
    cdef vector[size_t] leaf_pmf_off
    cdef vector[double] leaf_pmf_flat
    cdef vector[double] leaf_logpmf_flat

    # --- scope metadata for flat product-child matching ----------------------
    cdef vector[uint64_t] scope_sig
    cdef vector[int] scope_size
    cdef vector[size_t] scope_vars_off
    cdef vector[int] scope_vars_flat

    # --- identity ------------------------------------------------------------
    cdef vector[size_t] node_ids
    cdef list node_objs
    cdef size_t n_nodes
    cdef size_t root_index
    cdef int max_var
    cdef readonly list variables
    cdef object _metric_pools

    cdef void _build(self, CircuitNode root) except *
    cdef void _classify_leaf(self, CircuitNode node, size_t n) except *
    cdef void _fill_scope(self, CircuitNode node, size_t n) except *
    cdef void _postorder(self, CircuitNode node, dict index_of, list order) except *
    cdef void _refresh_leaf_pmfs(self) except *
    cdef void _refresh_sum_weights(self) except *
    cdef void _score(
        self,
        const int[:, ::1] data,
        const vector[int]& leaf_col,
        double[::1] out,
        bint log_space,
    ) except *
    cdef object _likelihood_impl(
        self, object data, object var_to_col, bint log_space
    )


cdef int _max_var_from_scope(unordered_set[int]& scope) noexcept

cdef cnp.ndarray _coerce_data_array(object data, bint allow_1d) except *

cdef void _leaf_column_map(
    CompiledCircuit g,
    object var_to_col,
    Py_ssize_t n_cols,
    vector[int]& leaf_col,
) except *

cdef void _build_evidence_vector(
    CompiledCircuit g, cnp.ndarray row, vector[int]& ev
) except *

cdef void _validate_batch_data(
    CompiledCircuit g,
    const int[:, ::1] data,
    const vector[int]& leaf_col,
) except *

cdef void _flat_eval_batch(
    CompiledCircuit g,
    const int[:, ::1] data,
    const vector[int]& leaf_col,
    Py_ssize_t n_rows,
    bint log_space,
    double[:, ::1] val,
    double[::1] out,
) noexcept nogil

cdef double _flat_eval(
    CompiledCircuit g, const int* ev, bint log_space, double[::1] val
) noexcept nogil

cdef void _flat_sample_node(
    CompiledCircuit g, size_t n, RandomState rng, int* out
) noexcept nogil


cdef void match_prod_children_flat(
    CompiledCircuit g0,
    size_t n0,
    CompiledCircuit g1,
    size_t n1,
    vector[int]& row_ind,
    vector[int]& col_ind,
    str query_name,
) except *


cdef double sp_graph_sigmoid(double x) noexcept nogil
cdef double graph_safe_log(double x) noexcept nogil
