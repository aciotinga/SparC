# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Pluggable ground metrics for Wasserstein (CW/GCW) queries.

A ground metric defines the cost of moving mass between two outcome indices of
a finite-discrete leaf. Custom metrics subclass :class:`GroundMetric` and fill
the ``pairwise`` (n x n, same support) and ``cross`` (n x m, two supports)
cost matrices. The default :class:`PNormMetric` uses ``|i - j|**p / scale``
on the integer line.
"""

from libcpp.vector cimport vector
from libc.math cimport fabs, pow


cdef class GroundMetric:
    """Abstract ground metric over integer outcome indices."""

    cdef void pairwise(self, int scope_var, size_t n, vector[double]& out) except *:
        raise NotImplementedError

    cdef void cross(
        self, int var_a, int var_b, size_t n, size_t m, vector[double]& out
    ) except *:
        raise NotImplementedError


cdef class PNormMetric(GroundMetric):
    r"""Ground metric :math:`d(i, j) = |i - j|^p / \mathrm{scale}` on the integer line.

    Args:
        p: Exponent in the distance (typically ``1.0`` or ``2.0``).
        scale: Positive scale divisor applied to every cost.
    """

    def __init__(self, double p=1.0, double scale=1.0):
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        self.p = p
        self.scale = scale

    cdef void pairwise(self, int scope_var, size_t n, vector[double]& out) except *:
        cdef size_t i
        cdef size_t j
        out.resize(n * n)
        for i in range(n):
            for j in range(n):
                out[i * n + j] = pow(fabs(<double>i - <double>j), self.p) / self.scale

    cdef void cross(
        self, int var_a, int var_b, size_t n, size_t m, vector[double]& out
    ) except *:
        cdef size_t i
        cdef size_t j
        out.resize(n * m)
        for i in range(n):
            for j in range(m):
                out[i * m + j] = pow(fabs(<double>i - <double>j), self.p) / self.scale
