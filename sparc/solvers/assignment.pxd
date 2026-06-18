from libcpp.vector cimport vector

cdef void assignment_min(
    const vector[double]& cost,
    size_t n,
    size_t m,
    vector[int]& row_ind,
    vector[int]& col_ind,
) except *
