# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Rectangular linear-sum assignment (Kuhn-Munkres / Hungarian), O(k^3).

Matches the smaller side fully and minimizes total cost -- the same semantics
as ``scipy.optimize.linear_sum_assignment``. The GCW product-vs-product handler
negates its score matrix to turn the max-weight matching into a min-cost one.
"""

from libcpp.vector cimport vector

cdef double _INF = 1e18


cdef void _hungarian_rect(
    const vector[double]& cost,
    size_t n,
    size_t m,
    vector[int]& row_to_col,
) except *:
    """Min-cost assignment for an n x m matrix with n <= m (1-indexed inside).

    Fills ``row_to_col`` (size n) with the matched column for each row.
    """
    cdef vector[double] u
    cdef vector[double] v
    cdef vector[int] p
    cdef vector[int] way
    cdef vector[double] minv
    cdef vector[char] used
    u.assign(n + 1, 0.0)
    v.assign(m + 1, 0.0)
    p.assign(m + 1, 0)
    way.assign(m + 1, 0)

    cdef size_t i
    cdef size_t j
    cdef int i0
    cdef int j0
    cdef int j1
    cdef double delta
    cdef double cur
    for i in range(1, n + 1):
        p[0] = <int>i
        j0 = 0
        minv.assign(m + 1, _INF)
        used.assign(m + 1, 0)
        while True:
            used[j0] = 1
            i0 = p[j0]
            delta = _INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = <int>j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col.assign(n, -1)
    for j in range(1, m + 1):
        if p[j] != 0:
            row_to_col[p[j] - 1] = <int>(j - 1)


cdef void assignment_min(
    const vector[double]& cost,
    size_t n,
    size_t m,
    vector[int]& row_ind,
    vector[int]& col_ind,
) except *:
    row_ind.clear()
    col_ind.clear()
    if n == 0 or m == 0:
        return
    cdef vector[int] matched
    cdef vector[double] cost_t
    cdef size_t i
    cdef size_t j
    if n <= m:
        _hungarian_rect(cost, n, m, matched)
        for i in range(n):
            row_ind.push_back(<int>i)
            col_ind.push_back(matched[i])
    else:
        cost_t.resize(n * m)
        for i in range(n):
            for j in range(m):
                cost_t[j * n + i] = cost[i * m + j]
        _hungarian_rect(cost_t, m, n, matched)
        for j in range(m):
            row_ind.push_back(matched[j])
            col_ind.push_back(<int>j)


def solve_assignment(object cost2d):
    """Return (row_ind, col_ind) minimizing total assignment cost."""
    import numpy as np

    arr = np.ascontiguousarray(cost2d, dtype=np.float64)
    cdef size_t n = arr.shape[0]
    cdef size_t m = arr.shape[1]
    cdef vector[double] cost
    cdef vector[int] row_ind
    cdef vector[int] col_ind
    cdef size_t i
    cdef size_t j
    cost.resize(n * m)
    for i in range(n):
        for j in range(m):
            cost[i * m + j] = arr[i, j]
    assignment_min(cost, n, m, row_ind, col_ind)
    ri = np.array([row_ind[i] for i in range(row_ind.size())], dtype=np.int64)
    ci = np.array([col_ind[i] for i in range(col_ind.size())], dtype=np.int64)
    return ri, ci
