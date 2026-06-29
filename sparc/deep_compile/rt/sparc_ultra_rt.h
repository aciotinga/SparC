#ifndef SPARC_ULTRA_RT_H
#define SPARC_ULTRA_RT_H

#include <stdint.h>
#include <math.h>

#ifdef _MSC_VER
#define SPARC_ULTRA_EXPORT __declspec(dllexport)
#else
#define SPARC_ULTRA_EXPORT
#endif

#ifndef SPARC_NEG_INF
#define SPARC_NEG_INF (-INFINITY)
#endif

SPARC_ULTRA_EXPORT void sparc_ultra_leaf_bin(
    int log_space,
    int32_t node,
    int32_t var,
    int32_t pmf_off,
    const double* tape,
    const int32_t* data,
    int32_t data_stride,
    const int32_t* col_for_var,
    int32_t r0,
    int32_t rn,
    int32_t ws_stride,
    double* workspace
);

SPARC_ULTRA_EXPORT void sparc_ultra_leaf_tbl(
    int log_space,
    int32_t node,
    int32_t var,
    int32_t pmf_off,
    int32_t card,
    const double* tape,
    const int32_t* data,
    int32_t data_stride,
    const int32_t* col_for_var,
    int32_t r0,
    int32_t rn,
    int32_t ws_stride,
    double* workspace
);

SPARC_ULTRA_EXPORT void sparc_ultra_product(
    int log_space,
    int32_t node,
    int32_t r0,
    int32_t rn,
    int32_t ws_stride,
    double* workspace,
    int32_t n_children,
    const int32_t* children
);

SPARC_ULTRA_EXPORT void sparc_ultra_sum(
    int log_space,
    int32_t node,
    const double* tape,
    int32_t r0,
    int32_t rn,
    int32_t ws_stride,
    double* workspace,
    int32_t n_children,
    const int32_t* w_idx,
    const int32_t* children
);

SPARC_ULTRA_EXPORT void sparc_ultra_ws_copy(
    int32_t dst_node,
    int32_t src_node,
    int32_t r0,
    int32_t rn,
    int32_t ws_stride,
    double* workspace
);

#endif
