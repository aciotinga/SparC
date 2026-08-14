# Header-only continuous-leaf formulas (1-D Gaussian). nogil, shared by
# object-graph and compiled paths.
from libc.math cimport INFINITY, exp, log, sqrt, M_PI

cdef inline double gaussian_log_density(
    double x, double mu, double sigma
) noexcept nogil:
    cdef double z
    if sigma <= 0.0:
        return -INFINITY
    z = (x - mu) / sigma
    return -0.5 * log(2.0 * M_PI) - log(sigma) - 0.5 * z * z


cdef inline double gaussian_density(
    double x, double mu, double sigma
) noexcept nogil:
    cdef double lp = gaussian_log_density(x, mu, sigma)
    if lp == -INFINITY:
        return 0.0
    return exp(lp)


cdef inline double gaussian_w2sq(
    double mu1, double s1, double mu2, double s2, double scale
) noexcept nogil:
    cdef double dmu = mu1 - mu2
    cdef double ds = s1 - s2
    return (dmu * dmu + ds * ds) / scale


cdef inline void gaussian_w2sq_grad(
    double mu1, double s1, double mu2, double s2, double scale, double g,
    double* dmu1, double* ds1, double* dmu2, double* ds2,
) noexcept nogil:
    cdef double inv = g * 2.0 / scale
    dmu1[0] = inv * (mu1 - mu2)
    ds1[0] = inv * (s1 - s2)
    dmu2[0] = inv * (mu2 - mu1)
    ds2[0] = inv * (s2 - s1)


cdef inline double gaussian_log_inner_product(
    double mu1, double s1, double mu2, double s2
) noexcept nogil:
    cdef double v = s1 * s1 + s2 * s2
    cdef double delta
    if v <= 0.0:
        return -INFINITY
    delta = mu1 - mu2
    return -0.5 * log(2.0 * M_PI * v) - 0.5 * delta * delta / v


cdef inline double gaussian_inner_product(
    double mu1, double s1, double mu2, double s2
) noexcept nogil:
    cdef double lp = gaussian_log_inner_product(mu1, s1, mu2, s2)
    if lp == -INFINITY:
        return 0.0
    return exp(lp)


cdef inline void gaussian_inner_product_grad(
    double mu1, double s1, double mu2, double s2, double g,
    double* dmu1, double* ds1, double* dmu2, double* ds2,
) noexcept nogil:
    cdef double v = s1 * s1 + s2 * s2
    cdef double delta = mu1 - mu2
    cdef double ip
    cdef double dlog_dv
    if v <= 0.0:
        dmu1[0] = 0.0
        ds1[0] = 0.0
        dmu2[0] = 0.0
        ds2[0] = 0.0
        return
    ip = exp(-0.5 * log(2.0 * M_PI * v) - 0.5 * delta * delta / v)
    dlog_dv = -0.5 / v + 0.5 * delta * delta / (v * v)
    dmu1[0] = g * ip * (-delta / v)
    dmu2[0] = g * ip * (delta / v)
    ds1[0] = g * ip * dlog_dv * 2.0 * s1
    ds2[0] = g * ip * dlog_dv * 2.0 * s2


cdef inline void gaussian_log_inner_product_grad(
    double mu1, double s1, double mu2, double s2, double g,
    double* dmu1, double* ds1, double* dmu2, double* ds2,
) noexcept nogil:
    cdef double v = s1 * s1 + s2 * s2
    cdef double delta = mu1 - mu2
    cdef double dlog_dv
    if v <= 0.0:
        dmu1[0] = 0.0
        ds1[0] = 0.0
        dmu2[0] = 0.0
        ds2[0] = 0.0
        return
    dlog_dv = -0.5 / v + 0.5 * delta * delta / (v * v)
    dmu1[0] = g * (-delta / v)
    dmu2[0] = g * (delta / v)
    ds1[0] = g * dlog_dv * 2.0 * s1
    ds2[0] = g * dlog_dv * 2.0 * s2


cdef inline double gaussian_esd(double sigma, double scale) noexcept nogil:
    return 2.0 * sigma * sigma / scale


cdef inline double gaussian_esd_dsigma(
    double sigma, double scale, double g
) noexcept nogil:
    return g * 4.0 * sigma / scale
