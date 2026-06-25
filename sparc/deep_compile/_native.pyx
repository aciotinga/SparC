# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""nogil helpers for deep-compiled batch inference."""

from libc.stdint cimport int32_t

import numpy as np
cimport numpy as cnp

cnp.import_array()


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
