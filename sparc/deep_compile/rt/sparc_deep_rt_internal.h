#ifndef SPARC_DEEP_RT_INTERNAL_H
#define SPARC_DEEP_RT_INTERNAL_H

#include "sparc_deep_rt.h"

#include <math.h>

#ifndef SPARC_NEG_INF
#define SPARC_NEG_INF (-INFINITY)
#endif

#define SPARC_MAX_STACK_FANIN 64

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

static inline double sparc_logsumexp_csr(
    const double* tape,
    int32_t sum_w_base,
    int32_t start,
    int32_t stop,
    const int16_t* children_flat,
    const double* workspace,
    int32_t ws_stride,
    int32_t lr
) {
    int32_t i;
    int32_t nf = stop - start;
    double terms[SPARC_MAX_STACK_FANIN];
    double max_log;
    double sum_exp;
    double term;

    if (nf <= 0) {
        return SPARC_NEG_INF;
    }
    if (nf <= SPARC_MAX_STACK_FANIN) {
        for (i = start; i < stop; ++i) {
            terms[i - start] = tape[sum_w_base + i]
                + sparc_ws_at(workspace, children_flat[i], ws_stride, lr);
        }
        return sparc_logsumexp_terms(terms, nf);
    }
    max_log = SPARC_NEG_INF;
    for (i = start; i < stop; ++i) {
        term = tape[sum_w_base + i]
            + sparc_ws_at(workspace, children_flat[i], ws_stride, lr);
        if (term > max_log) {
            max_log = term;
        }
    }
    if (!isfinite(max_log) || max_log == SPARC_NEG_INF) {
        return SPARC_NEG_INF;
    }
    sum_exp = 0.0;
    for (i = start; i < stop; ++i) {
        term = tape[sum_w_base + i]
            + sparc_ws_at(workspace, children_flat[i], ws_stride, lr);
        if (isfinite(term) && term > SPARC_NEG_INF) {
            sum_exp += exp(term - max_log);
        }
    }
    if (sum_exp <= 0.0) {
        return SPARC_NEG_INF;
    }
    return max_log + log(sum_exp);
}

static inline void sparc_prod_row_csr(
    int log_space,
    int32_t node,
    const int16_t* child_off,
    const int16_t* children_flat,
    double* workspace,
    int32_t ws_stride,
    int32_t lr
) {
    int32_t start = child_off[node];
    int32_t stop = child_off[node + 1];
    int32_t i;

    if (start >= stop) {
        sparc_ws_set(workspace, node, ws_stride, lr, log_space ? 0.0 : 1.0);
        return;
    }
    if (stop - start == 1) {
        sparc_ws_set(
            workspace, node, ws_stride, lr,
            sparc_ws_at(workspace, children_flat[start], ws_stride, lr)
        );
        return;
    }
    if (log_space) {
        double acc = sparc_ws_at(workspace, children_flat[start], ws_stride, lr);
        for (i = start + 1; i < stop; ++i) {
            acc += sparc_ws_at(workspace, children_flat[i], ws_stride, lr);
        }
        sparc_ws_set(workspace, node, ws_stride, lr, acc);
    } else {
        double acc = sparc_ws_at(workspace, children_flat[start], ws_stride, lr);
        for (i = start + 1; i < stop; ++i) {
            acc *= sparc_ws_at(workspace, children_flat[i], ws_stride, lr);
        }
        sparc_ws_set(workspace, node, ws_stride, lr, acc);
    }
}

static inline void sparc_sum_row_csr(
    int log_space,
    const double* tape,
    int32_t node,
    int32_t sum_w_base,
    const int16_t* child_off,
    const int16_t* children_flat,
    double* workspace,
    int32_t ws_stride,
    int32_t lr
) {
    int32_t start = child_off[node];
    int32_t stop = child_off[node + 1];
    int32_t i;
    double acc;

    if (start >= stop) {
        sparc_ws_set(workspace, node, ws_stride, lr, log_space ? SPARC_NEG_INF : 0.0);
        return;
    }
    if (stop - start == 1) {
        double c = sparc_ws_at(workspace, children_flat[start], ws_stride, lr);
        if (log_space) {
            sparc_ws_set(workspace, node, ws_stride, lr, tape[sum_w_base + start] + c);
        } else {
            sparc_ws_set(workspace, node, ws_stride, lr, tape[sum_w_base + start] * c);
        }
        return;
    }
    if (log_space) {
        sparc_ws_set(
            workspace, node, ws_stride, lr,
            sparc_logsumexp_csr(
                tape, sum_w_base, start, stop, children_flat, workspace, ws_stride, lr
            )
        );
        return;
    }
    acc = tape[sum_w_base + start]
        * sparc_ws_at(workspace, children_flat[start], ws_stride, lr);
    for (i = start + 1; i < stop; ++i) {
        acc += tape[sum_w_base + i]
            * sparc_ws_at(workspace, children_flat[i], ws_stride, lr);
    }
    sparc_ws_set(workspace, node, ws_stride, lr, acc);
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
    const int16_t* child_off,
    const int16_t* children_flat,
    int32_t sum_w_base,
    double* workspace
);

#endif
