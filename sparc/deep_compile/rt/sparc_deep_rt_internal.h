#ifndef SPARC_DEEP_RT_INTERNAL_H
#define SPARC_DEEP_RT_INTERNAL_H

#include "sparc_deep_rt.h"

#include <math.h>

#ifndef SPARC_NEG_INF
#define SPARC_NEG_INF (-INFINITY)
#endif

static inline double sparc_ws_at(
    const double* workspace, int32_t node, int32_t ws_stride, int32_t lr
) {
    return workspace[(size_t)node * (size_t)ws_stride + (size_t)lr];
}

static inline void sparc_ws_set(
    double* workspace, int32_t node, int32_t ws_stride, int32_t lr, double v
) {
    workspace[(size_t)node * (size_t)ws_stride + (size_t)lr] = v;
}

static inline double sparc_logsumexp_terms(const double* terms, int n) {
    int i;
    double m, s, t;
    if (n <= 0) {
        return SPARC_NEG_INF;
    }
    if (n == 1) {
        return terms[0];
    }
    m = terms[0];
    for (i = 1; i < n; ++i) {
        if (terms[i] > m) {
            m = terms[i];
        }
    }
    if (!isfinite(m) || m == SPARC_NEG_INF) {
        return SPARC_NEG_INF;
    }
    s = 0.0;
    for (i = 0; i < n; ++i) {
        t = terms[i];
        if (isfinite(t) && t > SPARC_NEG_INF) {
            s += exp(t - m);
        }
    }
    if (s <= 0.0) {
        return SPARC_NEG_INF;
    }
    return m + log(s);
}

void sparc_eval_tile_scalar(
    int log_space,
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t r0,
    int32_t rn,
    int32_t ws_stride,
    const SparcOp* ops,
    int32_t n_ops,
    double* workspace
);

#endif
