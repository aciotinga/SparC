# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Circuit-Wasserstein W_p^p between two structurally compatible PCs.

Returns the additive ``W_p^p`` objective from the CW recursion; the ``p``-th
root is the CW distance. Leaf couplings use the northwest-corner plan on the
integer line; sum-vs-sum couplings solve a transportation LP (built-in network
simplex by default) with duals for the marginal subgradient. Gradients are
returned for ``circuit2`` only.
"""

from libcpp.vector cimport vector

from sparc.grad cimport GradBundle
from sparc.metrics cimport GroundMetric, PNormMetric
from sparc.nodes cimport (
    CircuitNode,
    FiniteDiscreteInputNode,
    NODE_INPUT,
    NODE_PRODUCT,
    NODE_SUM,
    ProductNode,
    SumNode,
)
from sparc.queries._engine cimport (
    CoupleContext,
    NO_TAPE_IDX,
    TapeEntry,
    match_prod_children,
)
from sparc.solvers.northwest cimport nw_backward_marginals, nw_plan
from sparc.solvers.transport cimport transport_with_duals


cdef CircuitNode _unwrap(object circuit):
    from sparc.circuit import Circuit
    if isinstance(circuit, Circuit):
        return <CircuitNode>(<object>circuit).root
    if isinstance(circuit, CircuitNode):
        return <CircuitNode>circuit
    raise TypeError("expected a Circuit or CircuitNode")


cdef void _leaf_pmf(FiniteDiscreteInputNode leaf, vector[double]& out) except *:
    cdef size_t n = leaf.support_size()
    cdef size_t k
    out.resize(n)
    for k in range(n):
        out[k] = leaf.pmf_at(k)


cdef class _CWLeaf(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef vector[int] rows
    cdef vector[int] cols
    cdef vector[int] modes
    cdef vector[double] d_cross  # n x m

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef size_t num = self.rows.size()
        cdef size_t a
        cdef int i_a
        cdef int j_a
        cdef vector[double] G
        cdef vector[double] adj_p
        cdef vector[double] adj_q
        cdef object arr
        cdef size_t k
        G.resize(num)
        for a in range(num):
            i_a = self.rows[a]
            j_a = self.cols[a]
            G[a] = g * self.d_cross[<size_t>i_a * self.m + <size_t>j_a]
        nw_backward_marginals(self.rows, self.cols, self.modes, G, self.n, self.m, adj_p, adj_q)
        if self.side_Q == 1:
            arr = c.cat_grad_arr(1, self.Q, self.m)
            for k in range(self.m):
                arr[k] += adj_q[k]
        if self.side_P == 1:
            arr = c.cat_grad_arr(1, self.P, self.n)
            for k in range(self.n):
                arr[k] += adj_p[k]


cdef class _CWSumSum(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef vector[double] w
    cdef vector[double] V
    cdef vector[double] pi
    cdef vector[double] rho
    cdef vector[size_t] child_idx

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef size_t i
        cdef size_t j
        cdef size_t cid
        cdef object arr
        for i in range(self.n):
            for j in range(self.m):
                cid = self.child_idx[i * self.m + j]
                if cid != NO_TAPE_IDX:
                    c.tape_adjoints[cid] += g * self.w[i * self.m + j]
        if self.side_P == 1:
            arr = c.sum_grad_arr(1, self.P, self.n)
            for i in range(self.n):
                arr[i] += g * self.pi[i]
        if self.side_Q == 1:
            arr = c.sum_grad_arr(1, self.Q, self.m)
            for j in range(self.m):
                arr[j] += g * self.rho[j]


cdef class _CWProdProd(TapeEntry):
    cdef size_t m
    cdef vector[int] row_ind
    cdef vector[int] col_ind
    cdef vector[size_t] child_idx

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef size_t k_idx
        cdef size_t cid
        cdef int r
        cdef size_t num = self.row_ind.size()
        for k_idx in range(num):
            r = self.row_ind[k_idx]
            cid = self.child_idx[<size_t>r * self.m + <size_t>self.col_ind[k_idx]]
            if cid != NO_TAPE_IDX:
                c.tape_adjoints[cid] += g


cdef class CWContext(CoupleContext):
    cdef GroundMetric metric

    cdef double couple_value(self, CircuitNode P, CircuitNode Q, int sP, int sQ) except *:
        cdef double cached
        cdef double res
        if self.memo_get(P, Q, &cached):
            return cached
        if P.node_kind == NODE_INPUT and Q.node_kind == NODE_INPUT:
            res = self._leaf(<FiniteDiscreteInputNode>P, <FiniteDiscreteInputNode>Q, sP, sQ)
        elif P.node_kind == NODE_SUM and Q.node_kind == NODE_SUM:
            res = self._sum_sum(<SumNode>P, <SumNode>Q, sP, sQ)
        elif P.node_kind == NODE_PRODUCT and Q.node_kind == NODE_PRODUCT:
            res = self._prod_prod(<ProductNode>P, <ProductNode>Q, sP, sQ)
        else:
            raise ValueError(
                f"CW incompatible: cannot couple {type(P).__name__} with "
                f"{type(Q).__name__}"
            )
        self.memo_put(P, Q, res)
        return res

    cdef double _leaf(self, FiniteDiscreteInputNode P, FiniteDiscreteInputNode Q, int sP, int sQ) except *:
        cdef size_t n = P.support_size()
        cdef size_t m = Q.support_size()
        cdef vector[double] p_pmf
        cdef vector[double] q_pmf
        cdef vector[double] d_cross
        cdef vector[int] rows
        cdef vector[int] cols
        cdef vector[double] vals
        cdef vector[int] modes
        cdef size_t num
        cdef size_t a
        cdef double value = 0.0
        cdef _CWLeaf entry
        _leaf_pmf(P, p_pmf)
        _leaf_pmf(Q, q_pmf)
        self.metric.cross(P.scope_var_c(), Q.scope_var_c(), n, m, d_cross)
        num = nw_plan(p_pmf, q_pmf, n, m, rows, cols, vals, modes)
        for a in range(num):
            value += vals[a] * d_cross[<size_t>rows[a] * m + <size_t>cols[a]]
        if self.recording:
            entry = _CWLeaf()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.rows = rows
            entry.cols = cols
            entry.modes = modes
            entry.d_cross = d_cross
            self.append_tape(entry, P, Q)
        return value

    cdef double _sum_sum(self, SumNode P, SumNode Q, int sP, int sQ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef CircuitNode pc
        cdef CircuitNode qc
        cdef vector[double] V
        cdef vector[size_t] child_idx
        cdef vector[double] cost
        cdef vector[double] theta
        cdef vector[double] phi
        cdef vector[double] plan
        cdef vector[double] u
        cdef vector[double] v
        cdef double cross_term = 0.0
        cdef double w
        cdef _CWSumSum entry
        V.resize(n * m)
        if self.recording:
            child_idx.resize(n * m)
        for i in range(n):
            pc = P.child_at(i)
            for j in range(m):
                qc = Q.child_at(j)
                V[i * m + j] = self.couple_value(pc, qc, sP, sQ)
                if self.recording:
                    child_idx[i * m + j] = self.lookup_pair_tape_idx(pc, qc)
        cost = V  # CW minimizes <V, w>
        theta.resize(n)
        phi.resize(m)
        for i in range(n):
            theta[i] = P.parameter_at(i)
        for j in range(m):
            phi[j] = Q.parameter_at(j)
        transport_with_duals(cost, theta, phi, n, m, plan, u, v)
        for i in range(n):
            for j in range(m):
                w = plan[i * m + j]
                if w > 0.0:
                    cross_term += w * V[i * m + j]
        if self.recording:
            entry = _CWSumSum()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.w = plan
            entry.V = V
            entry.pi = u
            entry.rho = v
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return cross_term

    cdef double _prod_prod(self, ProductNode P, ProductNode Q, int sP, int sQ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef vector[int] row_ind
        cdef vector[int] col_ind
        cdef size_t i
        cdef int q_idx
        cdef CircuitNode pc
        cdef CircuitNode qc
        cdef vector[size_t] child_idx
        cdef double total = 0.0
        cdef _CWProdProd entry
        cdef size_t t
        match_prod_children(P, Q, row_ind, col_ind, "CW")
        if self.recording:
            child_idx.resize(n * m)
            for t in range(n * m):
                child_idx[t] = NO_TAPE_IDX
        for i in range(n):
            q_idx = col_ind[i]
            pc = P.child_at(i)
            qc = Q.child_at(<size_t>q_idx)
            total += self.couple_value(pc, qc, sP, sQ)
            if self.recording:
                child_idx[i * m + <size_t>q_idx] = self.lookup_pair_tape_idx(pc, qc)
        if self.recording:
            entry = _CWProdProd()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.m = m
            entry.row_ind = row_ind
            entry.col_ind = col_ind
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return total


cpdef double cw_distance(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor=1.0,
    object metric=None,
) except *:
    """Compute the Circuit-Wasserstein ``W_p^p`` objective between two PCs."""
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef CWContext ctx = CWContext()
    ctx.reset_base()
    ctx.metric = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    return ctx.couple_value(r1, r2, 0, 1)


cpdef tuple cw_distance_and_grad(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor=1.0,
    object metric=None,
):
    """Compute CW ``W_p^p`` and subgradients w.r.t. ``circuit2``.

    Returns ``(value, grads)`` with ``grads`` a :class:`~sparc.grad.GradBundle`
    for ``circuit2`` nodes only.
    """
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef CWContext ctx = CWContext()
    cdef double value
    cdef size_t root_idx
    cdef GradBundle grads
    ctx.reset_base()
    ctx.metric = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    ctx.recording = True
    try:
        value = ctx.couple_value(r1, r2, 0, 1)
        root_idx = ctx.lookup_pair_tape_idx(r1, r2)
        if root_idx == NO_TAPE_IDX:
            raise RuntimeError("internal: root pair has no tape entry")
        ctx.tape_adjoints[root_idx] = 1.0
        ctx.run_backward()
    finally:
        ctx.recording = False
    grads = GradBundle()
    grads.value = value
    grads.sum_grads = ctx.sum_grads1
    grads.cat_grads = ctx.cat_grads1
    return (value, grads)
