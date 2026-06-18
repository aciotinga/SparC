from libcpp.vector cimport vector

cdef double PROB_EPS

cdef size_t nw_plan(
    const vector[double]& p,
    const vector[double]& q,
    size_t n,
    size_t m,
    vector[int]& rows_out,
    vector[int]& cols_out,
    vector[double]& vals_out,
    vector[int]& modes_out,
) noexcept nogil

cdef double nw_run(
    const vector[double]& p,
    const vector[double]& q,
    const vector[double]& d_p,
    const vector[double]& d_q,
    size_t n,
    size_t m,
    vector[int]& rows_out,
    vector[int]& cols_out,
    vector[double]& vals_out,
    vector[int]& modes_out,
) noexcept nogil

cdef void nw_backward_marginals(
    const vector[int]& rows,
    const vector[int]& cols,
    const vector[int]& modes,
    const vector[double]& G,
    size_t n,
    size_t m,
    vector[double]& adj_p_out,
    vector[double]& adj_q_out,
) noexcept nogil
