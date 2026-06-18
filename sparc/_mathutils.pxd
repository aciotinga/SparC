# Header-only numerical helpers shared across SparC Cython modules.
from libcpp.vector cimport vector
from libc.math cimport exp, INFINITY, isfinite, log


cdef inline double SP_NEG_INF() noexcept nogil:
    return -INFINITY


cdef inline double sp_safe_log(double x) noexcept nogil:
    if x > 0.0:
        return log(x)
    return -INFINITY


cdef inline double sp_logsumexp(const vector[double]& log_vals) noexcept nogil:
    """Numerically stable log(sum(exp(log_vals)))."""
    cdef size_t n = log_vals.size()
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
