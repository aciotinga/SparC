#ifndef SPARC_DEEP_RT_H
#define SPARC_DEEP_RT_H

#include <stdint.h>

#ifdef _MSC_VER
#define SPARC_RT_EXPORT __declspec(dllexport)
#else
#define SPARC_RT_EXPORT
#endif

enum SparcOpKind {
    SPARC_OP_LEAF_BIN = 0,
    SPARC_OP_LEAF_TBL = 1,
    SPARC_OP_PROD_LIN = 2,
    SPARC_OP_PROD_LOG = 3,
    SPARC_OP_SUM_LIN = 4,
    SPARC_OP_SUM_LOG = 5,
};

#define SPARC_MAX_FANIN 4

typedef struct SparcOp {
    int16_t kind;
    int16_t node;
    int16_t leaf_idx;
    int16_t pmf_off;
    int16_t leaf_card;
    int16_t n_children;
    int16_t children[SPARC_MAX_FANIN];
    int16_t widx[SPARC_MAX_FANIN];
} SparcOp;

typedef void (*sparc_eval_batch_fn)(
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t n_rows,
    const SparcOp* ops,
    int32_t n_ops,
    int32_t n_nodes,
    int32_t root_index,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel
);

typedef struct SparcDispatch {
    sparc_eval_batch_fn eval_lin_batch;
    sparc_eval_batch_fn eval_log_batch;
    const char* isa_name;
} SparcDispatch;

SPARC_RT_EXPORT void sparc_init_dispatch(void);
SPARC_RT_EXPORT const SparcDispatch* sparc_dispatch(void);
SPARC_RT_EXPORT const char* sparc_active_isa_name(void);
SPARC_RT_EXPORT void sparc_force_isa(const char* name);

SPARC_RT_EXPORT int32_t sparc_workspace_doubles(
    int32_t n_nodes, int32_t tile, int32_t parallel
);

SPARC_RT_EXPORT void sparc_eval_lin_batch_scalar(
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t n_rows,
    const SparcOp* ops,
    int32_t n_ops,
    int32_t n_nodes,
    int32_t root_index,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel
);

SPARC_RT_EXPORT void sparc_eval_log_batch_scalar(
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t n_rows,
    const SparcOp* ops,
    int32_t n_ops,
    int32_t n_nodes,
    int32_t root_index,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel
);

SPARC_RT_EXPORT void sparc_eval_lin_batch_avx2(
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t n_rows,
    const SparcOp* ops,
    int32_t n_ops,
    int32_t n_nodes,
    int32_t root_index,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel
);

SPARC_RT_EXPORT void sparc_eval_log_batch_avx2(
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t n_rows,
    const SparcOp* ops,
    int32_t n_ops,
    int32_t n_nodes,
    int32_t root_index,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel
);

SPARC_RT_EXPORT void sparc_eval_lin_batch_avx512(
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t n_rows,
    const SparcOp* ops,
    int32_t n_ops,
    int32_t n_nodes,
    int32_t root_index,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel
);

SPARC_RT_EXPORT void sparc_eval_log_batch_avx512(
    const double* tape,
    const int32_t* leaf_ev,
    int32_t leaf_ev_stride,
    int32_t n_rows,
    const SparcOp* ops,
    int32_t n_ops,
    int32_t n_nodes,
    int32_t root_index,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel
);

#endif
