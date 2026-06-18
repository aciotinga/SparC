from sparc.nodes cimport CircuitNode

cpdef double likelihood(CircuitNode root, object assignment) except *
cpdef double log_likelihood(CircuitNode root, object assignment) except *
cpdef list sample(CircuitNode root, Py_ssize_t n_samples, object seed=*)
