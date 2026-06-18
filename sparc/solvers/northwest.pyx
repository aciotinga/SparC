# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Northwest-corner monotone coupling and its reverse-mode marginal subgradient.

Used by the GCW / CW leaf couplings, where the ground space is the integer line
and the monotone (comonotone) plan is optimal for separable convex ground costs.
"""

from libcpp.vector cimport vector

cdef double PROB_EPS = 1e-8


cdef inline double _dist_at(const vector[double]& mat, size_t n, size_t i, size_t j) noexcept nogil:
    return mat[i * n + j]


cdef size_t nw_plan(
    const vector[double]& p,
    const vector[double]& q,
    size_t n,
    size_t m,
    vector[int]& rows_out,
    vector[int]& cols_out,
    vector[double]& vals_out,
    vector[int]& modes_out,
) noexcept nogil:
    """Northwest-corner monotone coupling plan.

    ``modes_out[a]`` is 0 if ``p_rem`` was strictly smaller than ``q_rem`` at
    step ``a`` (P binding) and 1 otherwise (Q binding or tie).
    """
    cdef size_t max_entries = n + m - 1
    rows_out.resize(max_entries)
    cols_out.resize(max_entries)
    vals_out.resize(max_entries)
    modes_out.resize(max_entries)

    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t idx = 0
    cdef double p_rem = p[0]
    cdef double q_rem = q[0]
    cdef double flow
    cdef int mode

    while i < n and j < m:
        if p_rem < q_rem:
            flow = p_rem
            mode = 0
        else:
            flow = q_rem
            mode = 1

        rows_out[idx] = <int>i
        cols_out[idx] = <int>j
        vals_out[idx] = flow
        modes_out[idx] = mode
        idx += 1

        p_rem -= flow
        q_rem -= flow

        if p_rem < PROB_EPS:
            i += 1
            if i < n:
                p_rem = p[i]
        if q_rem < PROB_EPS:
            j += 1
            if j < m:
                q_rem = q[j]

    rows_out.resize(idx)
    cols_out.resize(idx)
    vals_out.resize(idx)
    modes_out.resize(idx)
    return idx


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
) noexcept nogil:
    """Northwest-corner plan plus the GCW bilinear cross-term aggregation."""
    cdef size_t idx
    cdef size_t a
    cdef size_t b
    cdef int i_a
    cdef int j_a
    cdef double v_a
    cdef double cross_term = 0.0

    idx = nw_plan(p, q, n, m, rows_out, cols_out, vals_out, modes_out)

    for a in range(idx):
        i_a = rows_out[a]
        j_a = cols_out[a]
        v_a = vals_out[a]
        cross_term += v_a * v_a * _dist_at(d_p, n, i_a, i_a) * _dist_at(d_q, m, j_a, j_a)
        for b in range(a):
            cross_term += (
                2.0 * v_a * vals_out[b]
                * _dist_at(d_p, n, i_a, rows_out[b])
                * _dist_at(d_q, m, cols_out[b], j_a)
            )
    return cross_term


cdef void nw_backward_marginals(
    const vector[int]& rows,
    const vector[int]& cols,
    const vector[int]& modes,
    const vector[double]& G,
    size_t n,
    size_t m,
    vector[double]& adj_p_out,
    vector[double]& adj_q_out,
) noexcept nogil:
    """Reverse-mode subgradient of the NW plan w.r.t. its marginals.

    Given ``G_a = dL/dw_a``, accumulate ``dL/dp`` and ``dL/dq`` using the local
    affine structure of the NW plan on its fixed support.
    """
    adj_p_out.assign(n, 0.0)
    adj_q_out.assign(m, 0.0)
    cdef vector[double] G_work = G
    cdef size_t num_steps = rows.size()
    cdef ssize_t a
    cdef size_t a_prime
    cdef int i_a
    cdef int j_a
    cdef double g_a

    if num_steps == 0:
        return

    for a in range(<ssize_t>num_steps - 1, -1, -1):
        g_a = G_work[<size_t>a]
        if g_a == 0.0:
            continue
        i_a = rows[<size_t>a]
        j_a = cols[<size_t>a]
        if modes[<size_t>a] == 0:
            adj_p_out[i_a] += g_a
            for a_prime in range(<size_t>a):
                if rows[a_prime] == i_a:
                    G_work[a_prime] -= g_a
        else:
            adj_q_out[j_a] += g_a
            for a_prime in range(<size_t>a):
                if cols[a_prime] == j_a:
                    G_work[a_prime] -= g_a
