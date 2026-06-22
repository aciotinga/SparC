from libcpp.unordered_set cimport unordered_set

from sparc.nodes cimport CircuitNode, Evidence

cdef Evidence _evidence_from_row(
    const int[:] row,
    Py_ssize_t n_cols,
    int max_var,
    unordered_set[int]& scope,
    object var_to_col,
) except *

cpdef object likelihood(CircuitNode root, object data, object var_to_col=*) except *
cpdef object log_likelihood(CircuitNode root, object data, object var_to_col=*) except *
cpdef object sample(CircuitNode root, Py_ssize_t n_samples, object seed=*)
