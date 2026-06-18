from libcpp.vector cimport vector

cdef int transport_with_duals(
    const vector[double]& cost,
    const vector[double]& supply,
    const vector[double]& demand,
    size_t n,
    size_t m,
    vector[double]& plan_out,
    vector[double]& u_out,
    vector[double]& v_out,
) except -1 nogil
