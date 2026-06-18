# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Balanced transportation problem via the transportation (MODI) simplex.

Solves ``min <cost, x>`` s.t. row sums == supply, col sums == demand, x >= 0,
returning the optimal plan and the dual potentials ``(u, v)`` with
``u_i + v_j == cost_ij`` on the optimal basis. The duals are the shadow prices
``d(obj)/d(supply_i)`` and ``d(obj)/d(demand_j)`` used by the sum×sum
subgradient in CW/GCW queries.

The right-hand side is perturbed by tiny distinct amounts to guarantee a
non-degenerate basic feasible solution (so the simplex strictly improves each
pivot and terminates); the duals depend only on costs + optimal basis, and the
returned plan is recomputed exactly on that basis with the true supply/demand.
"""

from libcpp.vector cimport vector
from libc.math cimport fabs

cdef double _TOL = 1e-9


cdef int _find_initial_basis(
    const vector[double]& sup,
    const vector[double]& dem,
    size_t n,
    size_t m,
    vector[char]& is_basic,
    vector[double]& x,
) except -1:
    """Northwest-corner basic feasible solution, completed to a spanning tree."""
    cdef vector[double] s = sup
    cdef vector[double] d = dem
    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t count = 0
    cdef double f
    is_basic.assign(n * m, 0)
    x.assign(n * m, 0.0)

    while True:
        f = s[i] if s[i] < d[j] else d[j]
        x[i * m + j] = f
        is_basic[i * m + j] = 1
        count += 1
        s[i] -= f
        d[j] -= f
        if i == n - 1 and j == m - 1:
            break
        if i < n - 1 and (s[i] <= _TOL or j == m - 1):
            i += 1
        else:
            j += 1

    # Complete to a spanning tree (n + m - 1 independent cells) via union-find,
    # adding zero-flow basic cells across disconnected components.
    cdef size_t total = n + m
    cdef vector[size_t] parent
    parent.resize(total)
    cdef size_t k
    for k in range(total):
        parent[k] = k
    cdef size_t cell
    cdef size_t ri
    cdef size_t cj
    for cell in range(n * m):
        if is_basic[cell]:
            _uf_union(parent, cell // m, n + (cell % m))
    for ri in range(n):
        for cj in range(m):
            if count >= n + m - 1:
                return 0
            if not is_basic[ri * m + cj]:
                if _uf_find(parent, ri) != _uf_find(parent, n + cj):
                    is_basic[ri * m + cj] = 1
                    _uf_union(parent, ri, n + cj)
                    count += 1
    return 0


cdef size_t _uf_find(vector[size_t]& parent, size_t a) noexcept nogil:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


cdef void _uf_union(vector[size_t]& parent, size_t a, size_t b) noexcept nogil:
    cdef size_t ra = _uf_find(parent, a)
    cdef size_t rb = _uf_find(parent, b)
    if ra != rb:
        parent[ra] = rb


cdef void _compute_potentials(
    const vector[double]& cost,
    const vector[char]& is_basic,
    size_t n,
    size_t m,
    vector[double]& u,
    vector[double]& v,
) noexcept:
    """Solve u_i + v_j = c_ij over the basis spanning tree (gauge u_0 = 0)."""
    cdef vector[char] u_known
    cdef vector[char] v_known
    u.assign(n, 0.0)
    v.assign(m, 0.0)
    u_known.assign(n, 0)
    v_known.assign(m, 0)
    u[0] = 0.0
    u_known[0] = 1

    cdef bint progress = True
    cdef size_t i
    cdef size_t j
    while progress:
        progress = False
        for i in range(n):
            for j in range(m):
                if not is_basic[i * m + j]:
                    continue
                if u_known[i] and not v_known[j]:
                    v[j] = cost[i * m + j] - u[i]
                    v_known[j] = 1
                    progress = True
                elif v_known[j] and not u_known[i]:
                    u[i] = cost[i * m + j] - v[j]
                    u_known[i] = 1
                    progress = True


cdef bint _find_loop(
    const vector[char]& is_basic,
    size_t n,
    size_t m,
    size_t enter_i,
    size_t enter_j,
    vector[size_t]& loop_cells,
) except *:
    """Find the unique cycle created by adding cell (enter_i, enter_j) to the
    basis tree. Returns cells in cyclic order starting with the entering cell.
    """
    cdef size_t total = n + m
    cdef size_t start = n + enter_j      # column node
    cdef size_t target = enter_i         # row node
    cdef vector[ssize_t] parent_node
    cdef vector[ssize_t] parent_cell
    parent_node.assign(total, -1)
    parent_cell.assign(total, -1)
    cdef vector[char] visited
    visited.assign(total, 0)
    cdef vector[size_t] stack
    stack.push_back(start)
    visited[start] = 1
    cdef size_t node
    cdef size_t i
    cdef size_t j
    cdef size_t nbr
    cdef bint found = False
    while not stack.empty():
        node = stack.back()
        stack.pop_back()
        if node == target:
            found = True
            break
        if node < n:
            i = node
            for j in range(m):
                if is_basic[i * m + j]:
                    nbr = n + j
                    if not visited[nbr]:
                        visited[nbr] = 1
                        parent_node[nbr] = <ssize_t>node
                        parent_cell[nbr] = <ssize_t>(i * m + j)
                        stack.push_back(nbr)
        else:
            j = node - n
            for i in range(n):
                if is_basic[i * m + j]:
                    nbr = i
                    if not visited[nbr]:
                        visited[nbr] = 1
                        parent_node[nbr] = <ssize_t>node
                        parent_cell[nbr] = <ssize_t>(i * m + j)
                        stack.push_back(nbr)
    if not found:
        return False
    loop_cells.clear()
    loop_cells.push_back(enter_i * m + enter_j)
    cdef ssize_t cur = <ssize_t>target
    while cur != <ssize_t>start:
        loop_cells.push_back(<size_t>parent_cell[cur])
        cur = parent_node[cur]
    return True


cdef int transport_with_duals(
    const vector[double]& cost,
    const vector[double]& supply,
    const vector[double]& demand,
    size_t n,
    size_t m,
    vector[double]& plan_out,
    vector[double]& u_out,
    vector[double]& v_out,
) except -1:
    if n == 0 or m == 0:
        raise ValueError("transport: empty marginals")
    cdef double s_sum = 0.0
    cdef double d_sum = 0.0
    cdef size_t i
    cdef size_t j
    for i in range(n):
        s_sum += supply[i]
    for j in range(m):
        d_sum += demand[j]
    if fabs(s_sum - d_sum) > 1e-6:
        raise ValueError(
            f"transport: unbalanced marginals (supply={s_sum}, demand={d_sum})"
        )

    # Perturb rhs with tiny distinct amounts for non-degeneracy.
    cdef double eps = 1e-9
    cdef vector[double] sup
    cdef vector[double] dem
    sup.resize(n)
    dem.resize(m)
    cdef double added = 0.0
    for i in range(n):
        sup[i] = supply[i] + eps * <double>(i + 1)
        added += eps * <double>(i + 1)
    for j in range(m):
        dem[j] = demand[j]
    dem[m - 1] += added

    cdef vector[char] is_basic
    cdef vector[double] x
    _find_initial_basis(sup, dem, n, m, is_basic, x)

    cdef vector[double] u
    cdef vector[double] v
    cdef vector[size_t] loop_cells
    cdef size_t enter
    cdef size_t enter_i
    cdef size_t enter_j
    cdef double best_rc
    cdef double rc
    cdef size_t k
    cdef size_t cell
    cdef double theta
    cdef size_t leave_pos
    cdef double flow
    cdef size_t max_iter = 50 * (n + m) + 1000
    cdef size_t it

    for it in range(max_iter):
        _compute_potentials(cost, is_basic, n, m, u, v)
        # Find entering cell with most negative reduced cost.
        best_rc = -_TOL
        enter = <size_t>(-1)
        for i in range(n):
            for j in range(m):
                if is_basic[i * m + j]:
                    continue
                rc = cost[i * m + j] - u[i] - v[j]
                if rc < best_rc:
                    best_rc = rc
                    enter = i * m + j
        if enter == <size_t>(-1):
            break  # optimal
        enter_i = enter // m
        enter_j = enter % m
        if not _find_loop(is_basic, n, m, enter_i, enter_j, loop_cells):
            raise RuntimeError("transport: failed to find pivot loop")
        # Odd positions (1, 3, ...) are the minus cells.
        theta = -1.0
        leave_pos = 0
        for k in range(1, loop_cells.size(), 2):
            flow = x[loop_cells[k]]
            if theta < 0.0 or flow < theta:
                theta = flow
                leave_pos = k
        # Pivot: +theta on even positions, -theta on odd positions.
        for k in range(loop_cells.size()):
            cell = loop_cells[k]
            if k % 2 == 0:
                x[cell] += theta
            else:
                x[cell] -= theta
        is_basic[enter] = 1
        is_basic[loop_cells[leave_pos]] = 0

    _compute_potentials(cost, is_basic, n, m, u, v)
    u_out = u
    v_out = v

    # Recompute the exact plan on the optimal basis with TRUE supply/demand.
    _solve_tree_flows(is_basic, supply, demand, n, m, plan_out)
    return 0


cdef void _solve_tree_flows(
    const vector[char]& is_basic,
    const vector[double]& supply,
    const vector[double]& demand,
    size_t n,
    size_t m,
    vector[double]& plan_out,
) except *:
    """Exact flows on the basis spanning tree via leaf elimination."""
    plan_out.assign(n * m, 0.0)
    cdef vector[double] res_sup = supply
    cdef vector[double] res_dem = demand
    cdef vector[int] row_deg
    cdef vector[int] col_deg
    row_deg.assign(n, 0)
    col_deg.assign(m, 0)
    cdef vector[char] done
    done.assign(n * m, 0)
    cdef size_t i
    cdef size_t j
    cdef size_t cell
    cdef size_t assigned = 0
    cdef size_t n_basis = 0
    for cell in range(n * m):
        if is_basic[cell]:
            row_deg[cell // m] += 1
            col_deg[cell % m] += 1
            n_basis += 1

    cdef bint progress = True
    cdef double flow
    while assigned < n_basis and progress:
        progress = False
        for cell in range(n * m):
            if not is_basic[cell] or done[cell]:
                continue
            i = cell // m
            j = cell % m
            if row_deg[i] == 1:
                flow = res_sup[i]
                plan_out[cell] = flow
                res_sup[i] -= flow
                res_dem[j] -= flow
                row_deg[i] -= 1
                col_deg[j] -= 1
                done[cell] = 1
                assigned += 1
                progress = True
            elif col_deg[j] == 1:
                flow = res_dem[j]
                plan_out[cell] = flow
                res_sup[i] -= flow
                res_dem[j] -= flow
                row_deg[i] -= 1
                col_deg[j] -= 1
                done[cell] = 1
                assigned += 1
                progress = True


# --- Python wrapper for testing ----------------------------------------------

def solve_transport(object cost2d, object supply, object demand):
    """Return (plan, u, v) for a balanced transportation problem.

    Parameters are array-likes; ``cost2d`` is shape (n, m).
    """
    import numpy as np

    cost_arr = np.ascontiguousarray(cost2d, dtype=np.float64)
    sup_arr = np.ascontiguousarray(supply, dtype=np.float64)
    dem_arr = np.ascontiguousarray(demand, dtype=np.float64)
    cdef size_t n = cost_arr.shape[0]
    cdef size_t m = cost_arr.shape[1]
    cdef vector[double] cost
    cdef vector[double] sup
    cdef vector[double] dem
    cdef vector[double] plan
    cdef vector[double] u
    cdef vector[double] v
    cdef size_t i
    cdef size_t j
    cost.resize(n * m)
    for i in range(n):
        for j in range(m):
            cost[i * m + j] = cost_arr[i, j]
    sup.resize(n)
    for i in range(n):
        sup[i] = sup_arr[i]
    dem.resize(m)
    for j in range(m):
        dem[j] = dem_arr[j]
    transport_with_duals(cost, sup, dem, n, m, plan, u, v)
    plan_np = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            plan_np[i, j] = plan[i * m + j]
    u_np = np.array([u[i] for i in range(n)], dtype=np.float64)
    v_np = np.array([v[j] for j in range(m)], dtype=np.float64)
    return plan_np, u_np, v_np
