# Header-only numerical helpers shared across SparC Cython modules.
from libcpp.vector cimport vector
from libc.math cimport exp, INFINITY, isfinite, log

cdef enum:
    SP_MAX_SUM_FANIN = 64


cdef inline double SP_NEG_INF() noexcept nogil:
    return -INFINITY


cdef inline double sp_safe_log(double x) noexcept nogil:
    if x > 0.0:
        return log(x)
    return -INFINITY


cdef inline double sp_logsumexp_ptr(const double* log_vals, size_t n) noexcept nogil:
    """Numerically stable log(sum(exp(log_vals))) from a C array."""
    cdef size_t i
    cdef double max_log
    cdef double sum_exp = 0.0
    cdef double x
    if n == 0:
        return -INFINITY
    max_log = log_vals[0]
    for i in range(1, n):
        if log_vals[i] > max_log:
            max_log = log_vals[i]
    if not isfinite(max_log) or max_log == -INFINITY:
        return -INFINITY
    for i in range(n):
        x = log_vals[i]
        if isfinite(x) and x > -INFINITY:
            sum_exp += exp(x - max_log)
    if sum_exp <= 0.0:
        return -INFINITY
    return max_log + log(sum_exp)


cdef inline void sp_logsumexp_batch(
    double* out,
    Py_ssize_t n_rows,
    const double* terms,
    size_t n_terms,
) noexcept nogil:
    """Per batch lane r, logsumexp over n_terms pre-gathered terms."""
    cdef Py_ssize_t r
    for r in range(n_rows):
        out[r] = sp_logsumexp_ptr(&terms[r * n_terms], n_terms)


cdef inline double sp_logsumexp(const vector[double]& log_vals) noexcept nogil:
    """Numerically stable log(sum(exp(log_vals)))."""
    return sp_logsumexp_ptr(log_vals.data(), log_vals.size())
