# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""nogil helpers for deep-compiled batch inference."""

from libc.stdint cimport int32_t

import numpy as np
cimport numpy as cnp

cnp.import_array()

ctypedef void (*sparc_batch_fn)(
    double* tape,
    const int32_t* data,
    int32_t data_stride,
    const int32_t* col_for_var,
    int32_t n_rows,
    double* workspace,
    double* out,
    int32_t tile,
    int32_t parallel,
) noexcept nogil


cdef void _eval_batch_nogil(
    sparc_batch_fn fn,
    double[::1] tape,
    const int32_t[:, ::1] data,
    const int32_t[::1] col_for_var,
    double[::1] workspace,
    double[::1] out,
    int32_t tile,
    int32_t parallel,
) noexcept nogil:
    cdef int32_t n_rows = <int32_t>data.shape[0]
    cdef int32_t data_stride = <int32_t>data.shape[1]
    fn(
        &tape[0],
        &data[0, 0],
        data_stride,
        &col_for_var[0],
        n_rows,
        &workspace[0],
        &out[0],
        tile,
        parallel,
    )


def eval_likelihood_batch(
    size_t fn_ptr,
    cnp.ndarray tape,
    cnp.ndarray data,
    cnp.ndarray col_for_var,
    cnp.ndarray workspace,
    cnp.ndarray out,
    int tile,
    int parallel,
):
    """Call native sparc_likelihood_batch under nogil."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tape1d = np.ascontiguousarray(tape, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] data2d = np.ascontiguousarray(data, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] col_arr = np.ascontiguousarray(col_for_var, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ws_arr = np.ascontiguousarray(workspace, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_arr = np.ascontiguousarray(out, dtype=np.float64)
    cdef sparc_batch_fn fn = <sparc_batch_fn>fn_ptr
    cdef double[::1] tape_view = tape1d
    cdef const int32_t[:, ::1] data_view = data2d
    cdef const int32_t[::1] col_view = col_arr
    cdef double[::1] ws_view = ws_arr
    cdef double[::1] out_view = out_arr
    with nogil:
        _eval_batch_nogil(
            fn,
            tape_view,
            data_view,
            col_view,
            ws_view,
            out_view,
            <int32_t>tile,
            <int32_t>parallel,
        )
    return out_arr


def eval_log_likelihood_batch(
    size_t fn_ptr,
    cnp.ndarray tape,
    cnp.ndarray data,
    cnp.ndarray col_for_var,
    cnp.ndarray workspace,
    cnp.ndarray out,
    int tile,
    int parallel,
):
    """Call native sparc_log_likelihood_batch under nogil."""
    return eval_likelihood_batch(
        fn_ptr, tape, data, col_for_var, workspace, out, tile, parallel
    )


cdef void _fill_leaf_ev_nogil(
    const int32_t[:, ::1] data,
    const int32_t[::1] col_for_var,
    const int32_t[::1] leaf_vars,
    int32_t[:, ::1] leaf_ev,
) noexcept nogil:
    cdef Py_ssize_t i, r, n_rows, n_leaf, var, col
    n_rows = data.shape[0]
    n_leaf = leaf_vars.shape[0]
    for i in range(n_leaf):
        var = leaf_vars[i]
        col = col_for_var[var]
        for r in range(n_rows):
            leaf_ev[i, r] = data[r, col]


def fill_leaf_ev(
    cnp.ndarray data,
    cnp.ndarray col_for_var,
    cnp.ndarray leaf_vars,
    cnp.ndarray leaf_ev,
):
    """Fill (n_leaf, n_rows) leaf evidence from batch data under nogil."""
    cdef cnp.ndarray[cnp.int32_t, ndim=2] data2d = np.ascontiguousarray(data, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] col_arr = np.ascontiguousarray(col_for_var, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] leaf_arr = np.ascontiguousarray(leaf_vars, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] out_arr = np.ascontiguousarray(leaf_ev, dtype=np.int32)
    cdef const int32_t[:, ::1] data_view = data2d
    cdef const int32_t[::1] col_view = col_arr
    cdef const int32_t[::1] leaf_view = leaf_arr
    cdef int32_t[:, ::1] leaf_ev_view = out_arr
    with nogil:
        _fill_leaf_ev_nogil(data_view, col_view, leaf_view, leaf_ev_view)
    return out_arr
