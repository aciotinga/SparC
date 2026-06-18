# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Circuit-Wasserstein W_p^p between two structurally compatible PCs.

Returns the additive ``W_p^p`` objective from the CW recursion; the ``p``-th
root is the CW distance. Leaf couplings use the northwest-corner plan on the
integer line; sum-vs-sum couplings solve a transportation LP (built-in network
simplex by default) with duals for the marginal subgradient. Gradients are
returned for ``circuit2`` only.
"""

from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

from sparc._graph cimport CompiledGraph
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
    pair_key,
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


# === Flattened nogil tape + pools (gradients for circuit2 only) ===============

cdef enum CWEntryKind:
    CK_LEAF = 0
    CK_SUMSUM = 1
    CK_PRODPROD = 2


cdef struct CWEntry:
    int kind
    size_t pQ
    size_t n
    size_t m
    vector[int] rows
    vector[int] cols
    vector[int] modes
    vector[double] d_cross
    vector[double] w
    vector[double] V
    vector[double] rho
    vector[size_t] child_idx
    vector[int] row_ind
    vector[int] col_ind


cdef void _cw_backward_core(
    CWEntry* ee, double* adj, size_t n_entries,
    const size_t* lpoff1, const size_t* coff1,
    double* cat1, double* sum1,
) noexcept nogil:
    cdef ssize_t t
    cdef CWEntry* e
    cdef double g
    cdef size_t i
    cdef size_t j
    cdef size_t a
    cdef size_t cid
    cdef size_t kidx
    cdef size_t num
    cdef size_t r
    cdef size_t n
    cdef size_t m
    cdef size_t qoff
    cdef vector[double] G
    cdef vector[double] adj_p
    cdef vector[double] adj_q
    for t in range(<ssize_t>n_entries - 1, -1, -1):
        g = adj[<size_t>t]
        if g == 0.0:
            continue
        e = &ee[<size_t>t]
        n = e.n
        m = e.m
        if e.kind == CK_LEAF:
            num = e.rows.size()
            G.resize(num)
            for a in range(num):
                G[a] = g * e.d_cross[<size_t>e.rows[a] * m + <size_t>e.cols[a]]
            nw_backward_marginals(e.rows, e.cols, e.modes, G, n, m, adj_p, adj_q)
            qoff = lpoff1[e.pQ]
            for j in range(m):
                cat1[qoff + j] += adj_q[j]
        elif e.kind == CK_SUMSUM:
            for i in range(n):
                for j in range(m):
                    cid = e.child_idx[i * m + j]
                    if cid != NO_TAPE_IDX:
                        adj[cid] += g * e.w[i * m + j]
            qoff = coff1[e.pQ]
            for j in range(m):
                sum1[qoff + j] += g * e.rho[j]
        else:  # CK_PRODPROD
            num = e.row_ind.size()
            for kidx in range(num):
                r = <size_t>e.row_ind[kidx]
                cid = e.child_idx[r * m + <size_t>e.col_ind[kidx]]
                if cid != NO_TAPE_IDX:
                    adj[cid] += g


cdef class _FlatCWContext(CoupleContext):
    cdef GroundMetric metric
    cdef CompiledGraph g1
    cdef unordered_map[size_t, size_t] pos1
    cdef vector[CWEntry] etape
    cdef vector[double] cat1
    cdef vector[double] sum1

    cdef void _setup(self, CompiledGraph graph1) except *:
        self.reset_base()
        self.recording = True
        self.g1 = graph1
        cdef size_t pos
        self.pos1.clear()
        for pos in range(self.g1.n_nodes):
            self.pos1[self.g1.node_ids[pos]] = pos
        self.cat1.assign(self.g1.leaf_pmf_flat.size(), 0.0)
        self.sum1.assign(self.g1.children_flat.size(), 0.0)
        self.etape.clear()

    cdef size_t _eappend(self, CircuitNode P, CircuitNode Q) except *:
        cdef size_t idx = self.etape.size()
        self.etape.push_back(CWEntry())
        self.tape_adjoints.push_back(0.0)
        self.pair_to_tape[pair_key(P, Q)] = idx
        return idx

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
        cdef size_t idx
        _leaf_pmf(P, p_pmf)
        _leaf_pmf(Q, q_pmf)
        self.metric.cross(P.scope_var_c(), Q.scope_var_c(), n, m, d_cross)
        num = nw_plan(p_pmf, q_pmf, n, m, rows, cols, vals, modes)
        for a in range(num):
            value += vals[a] * d_cross[<size_t>rows[a] * m + <size_t>cols[a]]
        idx = self._eappend(P, Q)
        self.etape[idx].kind = CK_LEAF
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].m = m
        self.etape[idx].rows = rows
        self.etape[idx].cols = cols
        self.etape[idx].modes = modes
        self.etape[idx].d_cross = d_cross
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
        cdef size_t idx
        V.resize(n * m)
        child_idx.resize(n * m)
        for i in range(n):
            pc = P.child_at(i)
            for j in range(m):
                qc = Q.child_at(j)
                V[i * m + j] = self.couple_value(pc, qc, sP, sQ)
                child_idx[i * m + j] = self.lookup_pair_tape_idx(pc, qc)
        cost = V
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
        idx = self._eappend(P, Q)
        self.etape[idx].kind = CK_SUMSUM
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].m = m
        self.etape[idx].w = plan
        self.etape[idx].rho = v
        self.etape[idx].child_idx = child_idx
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
        cdef size_t t
        cdef size_t idx
        match_prod_children(P, Q, row_ind, col_ind, "CW")
        child_idx.resize(n * m)
        for t in range(n * m):
            child_idx[t] = NO_TAPE_IDX
        for i in range(n):
            q_idx = col_ind[i]
            pc = P.child_at(i)
            qc = Q.child_at(<size_t>q_idx)
            total += self.couple_value(pc, qc, sP, sQ)
            child_idx[i * m + <size_t>q_idx] = self.lookup_pair_tape_idx(pc, qc)
        idx = self._eappend(P, Q)
        self.etape[idx].kind = CK_PRODPROD
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].m = m
        self.etape[idx].row_ind = row_ind
        self.etape[idx].col_ind = col_ind
        self.etape[idx].child_idx = child_idx
        return total

    cdef tuple solve_with_grad(self, CircuitNode r1, CircuitNode r2, CompiledGraph graph1):
        self._setup(graph1)
        cdef double value = self.couple_value(r1, r2, 0, 1)
        cdef size_t root_idx = self.lookup_pair_tape_idx(r1, r2)
        if root_idx == NO_TAPE_IDX:
            raise RuntimeError("internal: root pair has no tape entry")
        self.tape_adjoints[root_idx] = 1.0
        with nogil:
            _cw_backward_core(
                self.etape.data(), self.tape_adjoints.data(), self.etape.size(),
                self.g1.leaf_pmf_off.data(), self.g1.child_off.data(),
                self.cat1.data(), self.sum1.data(),
            )
        cdef GradBundle grads = GradBundle()
        grads.value = value
        grads.sum_grads = self._grads_for(True)
        grads.cat_grads = self._grads_for(False)
        return (value, grads)

    cdef dict _grads_for(self, bint sums):
        cdef dict out = {}
        cdef size_t nn
        cdef size_t start
        cdef size_t stop
        cdef size_t off
        cdef size_t k
        cdef int card
        cdef object arr
        for nn in range(self.g1.n_nodes):
            if sums:
                if self.g1.kinds[nn] != NODE_SUM:
                    continue
                start = self.g1.child_off[nn]
                stop = self.g1.child_off[nn + 1]
                arr = np.empty(stop - start, dtype=np.float64)
                for k in range(start, stop):
                    arr[k - start] = self.sum1[k]
                out[self.g1.node_ids[nn]] = arr
            else:
                if self.g1.kinds[nn] != NODE_INPUT:
                    continue
                off = self.g1.leaf_pmf_off[nn]
                card = self.g1.leaf_card[nn]
                arr = np.empty(card, dtype=np.float64)
                for k in range(<size_t>card):
                    arr[k] = self.cat1[off + k]
                out[self.g1.node_ids[nn]] = arr
        return out


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
    cdef GroundMetric m = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    cdef CompiledGraph g0 = CompiledGraph()
    cdef CompiledGraph g1 = CompiledGraph()
    g0.build(r1)
    g1.build(r2)
    cdef CWContext ctx
    cdef double value
    cdef size_t root_idx
    cdef GradBundle grads
    cdef _FlatCWContext flat
    if g0.has_fallback or g1.has_fallback:
        ctx = CWContext()
        ctx.reset_base()
        ctx.metric = m
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
    flat = _FlatCWContext()
    flat.metric = m
    return flat.solve_with_grad(r1, r2, g1)
