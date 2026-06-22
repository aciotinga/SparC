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

from sparc._graph cimport CompiledCircuit, match_prod_children_flat
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
    if isinstance(circuit, CompiledCircuit):
        raise TypeError(
            "expected a Circuit or CircuitNode for object-graph queries; "
            "use CompiledCircuit methods for the fast path"
        )
    if isinstance(circuit, Circuit):
        return <CircuitNode>(<object>circuit).root
    if isinstance(circuit, CircuitNode):
        return <CircuitNode>circuit
    raise TypeError("expected a Circuit or CircuitNode")


cdef CircuitNode _compiled_root(CompiledCircuit g):
    return <CircuitNode>g.node_objs[g.root_index]


cdef void _check_pair_types(object c1, object c2) except *:
    from sparc.circuit import Circuit
    cdef bint cc1 = isinstance(c1, CompiledCircuit)
    cdef bint cc2 = isinstance(c2, CompiledCircuit)
    if cc1 != cc2:
        raise TypeError(
            "pairwise queries require both operands to be the same kind: "
            "either both Circuit/CircuitNode or both CompiledCircuit"
        )


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
    size_t p0
    size_t pQ
    size_t n
    size_t m
    vector[int] rows
    vector[int] cols
    vector[int] modes
    vector[double] d_cross
    vector[double] w
    vector[double] nw_vals
    vector[double] rho
    vector[size_t] child_idx
    vector[int] row_ind
    vector[int] col_ind


cdef inline uint64_t _flat_pair_key(size_t i0, size_t i1) noexcept:
    return (<uint64_t>i0 << 32) | <uint64_t>i1


cdef void _cw_forward_core(
    CWEntry* ee,
    double* vals,
    size_t n_entries,
    const size_t* lpoff0,
    const double* lpmf0,
    const size_t* lpoff1,
    const double* lpmf1,
    const size_t* coff0,
    const double* sw0,
    const size_t* coff1,
    const double* sw1,
) noexcept nogil:
    cdef size_t t
    cdef CWEntry* e
    cdef size_t n
    cdef size_t m
    cdef size_t i
    cdef size_t j
    cdef size_t a
    cdef size_t num
    cdef size_t cid
    cdef size_t kidx
    cdef size_t r
    cdef double value
    cdef double cross_term
    cdef double w
    cdef vector[double] p_pmf
    cdef vector[double] q_pmf
    cdef vector[double] V
    cdef vector[double] cost
    cdef vector[double] theta
    cdef vector[double] phi
    cdef vector[double] plan
    cdef vector[double] u
    cdef vector[double] v
    for t in range(n_entries):
        e = &ee[t]
        n = e.n
        m = e.m
        if e.kind == CK_LEAF:
            p_pmf.resize(n)
            q_pmf.resize(m)
            for i in range(n):
                p_pmf[i] = lpmf0[lpoff0[e.p0] + i]
            for j in range(m):
                q_pmf[j] = lpmf1[lpoff1[e.pQ] + j]
            num = nw_plan(
                p_pmf, q_pmf, n, m, e.rows, e.cols, e.nw_vals, e.modes
            )
            value = 0.0
            for a in range(num):
                value += e.nw_vals[a] * e.d_cross[
                    <size_t>e.rows[a] * m + <size_t>e.cols[a]
                ]
            vals[t] = value
        elif e.kind == CK_SUMSUM:
            V.resize(n * m)
            for i in range(n):
                for j in range(m):
                    cid = e.child_idx[i * m + j]
                    if cid != NO_TAPE_IDX:
                        V[i * m + j] = vals[cid]
                    else:
                        V[i * m + j] = 0.0
            cost = V
            theta.resize(n)
            phi.resize(m)
            for i in range(n):
                theta[i] = sw0[coff0[e.p0] + i]
            for j in range(m):
                phi[j] = sw1[coff1[e.pQ] + j]
            transport_with_duals(cost, theta, phi, n, m, plan, u, v)
            cross_term = 0.0
            for i in range(n):
                for j in range(m):
                    w = plan[i * m + j]
                    if w > 0.0:
                        cross_term += w * V[i * m + j]
            e.w = plan
            e.rho = v
            vals[t] = cross_term
        else:  # CK_PRODPROD
            value = 0.0
            num = e.row_ind.size()
            for kidx in range(num):
                r = <size_t>e.row_ind[kidx]
                cid = e.child_idx[r * m + <size_t>e.col_ind[kidx]]
                if cid != NO_TAPE_IDX:
                    value += vals[cid]
            vals[t] = value


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
    cdef CompiledCircuit g0
    cdef CompiledCircuit g1
    cdef unordered_map[uint64_t, size_t] flat_pair_to_tape
    cdef vector[CWEntry] etape
    cdef vector[double] tape_vals
    cdef vector[double] cat1
    cdef vector[double] sum1

    cdef void _reset_flat(self) except *:
        self.reset_base()
        self.g0 = None
        self.g1 = None
        self.flat_pair_to_tape.clear()
        self.etape.clear()
        self.tape_vals.clear()
        self.cat1.clear()
        self.sum1.clear()

    cdef size_t _record_pair(self, size_t i0, size_t i1) except *:
        cdef uint64_t key = _flat_pair_key(i0, i1)
        if self.flat_pair_to_tape.count(key):
            return self.flat_pair_to_tape[key]

        cdef CompiledCircuit g0 = self.g0
        cdef CompiledCircuit g1 = self.g1
        cdef int k0 = g0.kinds[i0]
        cdef int k1 = g1.kinds[i1]
        cdef size_t idx
        cdef size_t n
        cdef size_t m
        cdef size_t i
        cdef size_t j
        cdef size_t qidx
        cdef size_t start0
        cdef size_t stop0
        cdef size_t start1
        cdef size_t stop1
        cdef size_t ci0
        cdef size_t ci1
        cdef vector[int] row_ind
        cdef vector[int] col_ind
        cdef vector[size_t] child_idx
        cdef vector[double] d_cross

        if k0 == NODE_INPUT and k1 == NODE_INPUT:
            n = <size_t>g0.leaf_card[i0]
            m = <size_t>g1.leaf_card[i1]
            idx = self.etape.size()
            self.etape.push_back(CWEntry())
            self.tape_adjoints.push_back(0.0)
            self.flat_pair_to_tape[key] = idx
            self.etape[idx].kind = CK_LEAF
            self.etape[idx].p0 = i0
            self.etape[idx].pQ = i1
            self.etape[idx].n = n
            self.etape[idx].m = m
            self.metric.cross(g0.leaf_var[i0], g1.leaf_var[i1], n, m, d_cross)
            self.etape[idx].d_cross = d_cross
            return idx
        elif k0 == NODE_SUM and k1 == NODE_SUM:
            start0 = g0.child_off[i0]
            stop0 = g0.child_off[i0 + 1]
            n = stop0 - start0
            start1 = g1.child_off[i1]
            stop1 = g1.child_off[i1 + 1]
            m = stop1 - start1
            child_idx.resize(n * m)
            for i in range(n):
                for j in range(m):
                    ci0 = g0.children_flat[start0 + i]
                    ci1 = g1.children_flat[start1 + j]
                    child_idx[i * m + j] = self._record_pair(ci0, ci1)
            idx = self.etape.size()
            self.etape.push_back(CWEntry())
            self.tape_adjoints.push_back(0.0)
            self.flat_pair_to_tape[key] = idx
            self.etape[idx].kind = CK_SUMSUM
            self.etape[idx].p0 = i0
            self.etape[idx].pQ = i1
            self.etape[idx].n = n
            self.etape[idx].m = m
            self.etape[idx].child_idx = child_idx
            return idx
        elif k0 == NODE_PRODUCT and k1 == NODE_PRODUCT:
            start0 = g0.child_off[i0]
            stop0 = g0.child_off[i0 + 1]
            n = stop0 - start0
            start1 = g1.child_off[i1]
            stop1 = g1.child_off[i1 + 1]
            m = stop1 - start1
            match_prod_children_flat(g0, i0, g1, i1, row_ind, col_ind, "CW")
            child_idx.resize(n * m)
            for t in range(n * m):
                child_idx[t] = NO_TAPE_IDX
            for i in range(n):
                qidx = <size_t>col_ind[i]
                ci0 = g0.children_flat[start0 + i]
                ci1 = g1.children_flat[start1 + qidx]
                child_idx[i * m + qidx] = self._record_pair(ci0, ci1)
            idx = self.etape.size()
            self.etape.push_back(CWEntry())
            self.tape_adjoints.push_back(0.0)
            self.flat_pair_to_tape[key] = idx
            self.etape[idx].kind = CK_PRODPROD
            self.etape[idx].p0 = i0
            self.etape[idx].pQ = i1
            self.etape[idx].n = n
            self.etape[idx].m = m
            self.etape[idx].row_ind = row_ind
            self.etape[idx].col_ind = col_ind
            self.etape[idx].child_idx = child_idx
            return idx
        else:
            raise ValueError(
                f"CW incompatible: cannot couple node kinds {k0} with {k1}"
            )

    cdef double _solve_flat(self, CompiledCircuit c0, CompiledCircuit c1) except *:
        self._reset_flat()
        self.g0 = c0
        self.g1 = c1
        self.cat1.assign(self.g1.leaf_pmf_flat.size(), 0.0)
        self.sum1.assign(self.g1.children_flat.size(), 0.0)
        self._record_pair(self.g0.root_index, self.g1.root_index)
        self.tape_vals.assign(self.etape.size(), 0.0)
        with nogil:
            _cw_forward_core(
                self.etape.data(), self.tape_vals.data(), self.etape.size(),
                self.g0.leaf_pmf_off.data(), self.g0.leaf_pmf_flat.data(),
                self.g1.leaf_pmf_off.data(), self.g1.leaf_pmf_flat.data(),
                self.g0.child_off.data(), self.g0.sum_w_flat.data(),
                self.g1.child_off.data(), self.g1.sum_w_flat.data(),
            )
        return self.tape_vals[self.etape.size() - 1]

    cdef double solve_value(self, CompiledCircuit c0, CompiledCircuit c1):
        return self._solve_flat(c0, c1)

    cdef tuple solve_with_grad(self, CompiledCircuit c0, CompiledCircuit c1):
        cdef double value = self._solve_flat(c0, c1)
        cdef size_t root_idx = self.etape.size() - 1
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
    r"""Compute the Circuit-Wasserstein :math:`W_p^p` objective between two PCs.

    Returns the additive :math:`W_p^p` value from the CW recursion; the
    :math:`p`-th root is the CW distance. Circuits must be structurally
    compatible (matching scopes and decompositions).

    Args:
        circuit1: First circuit (:class:`~sparc.circuit.Circuit` or root node).
        circuit2: Second circuit.
        metric_p: Exponent for the default :class:`PNormMetric` when ``metric``
            is omitted.
        scale_factor: Scale for the default metric.
        metric: Optional :class:`GroundMetric` instance.

    Returns:
        The :math:`W_p^p` objective value.
    """
    cdef GroundMetric m
    cdef CompiledCircuit c1
    cdef CompiledCircuit c2
    cdef _FlatCWContext flat
    cdef CircuitNode r1
    cdef CircuitNode r2
    cdef CWContext ctx
    _check_pair_types(circuit1, circuit2)
    m = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    if isinstance(circuit1, CompiledCircuit):
        c1 = circuit1
        c2 = circuit2
        flat = _FlatCWContext()
        flat.metric = m
        return flat.solve_value(c1, c2)
    r1 = _unwrap(circuit1)
    r2 = _unwrap(circuit2)
    ctx = CWContext()
    ctx.reset_base()
    ctx.metric = m
    return ctx.couple_value(r1, r2, 0, 1)


cpdef tuple cw_distance_and_grad(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor=1.0,
    object metric=None,
):
    r"""Compute CW :math:`W_p^p` and subgradients w.r.t. ``circuit2``.

    Args:
        circuit1: First circuit (:class:`~sparc.circuit.Circuit` or root node).
        circuit2: Second circuit (receives gradients).
        metric_p: Exponent for the default :class:`PNormMetric`.
        scale_factor: Scale for the default metric.
        metric: Optional :class:`GroundMetric` instance.

    Returns:
        ``(value, grads)`` where ``grads`` is a :class:`~sparc.grad.GradBundle`
        over ``circuit2`` nodes only.
    """
    cdef GroundMetric m
    cdef CompiledCircuit c1
    cdef CompiledCircuit c2
    cdef _FlatCWContext flat
    cdef CircuitNode r1
    cdef CircuitNode r2
    cdef CWContext ctx
    cdef double value
    cdef size_t root_idx
    cdef GradBundle grads
    _check_pair_types(circuit1, circuit2)
    m = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    if isinstance(circuit1, CompiledCircuit):
        c1 = circuit1
        c2 = circuit2
        flat = _FlatCWContext()
        flat.metric = m
        return flat.solve_with_grad(c1, c2)
    r1 = _unwrap(circuit1)
    r2 = _unwrap(circuit2)
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
