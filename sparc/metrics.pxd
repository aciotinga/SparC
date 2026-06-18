from libcpp.vector cimport vector

cdef class GroundMetric:
    cdef void pairwise(self, int scope_var, size_t n, vector[double]& out) except *
    cdef void cross(
        self, int var_a, int var_b, size_t n, size_t m, vector[double]& out
    ) except *

cdef class PNormMetric(GroundMetric):
    cdef double p
    cdef double scale
