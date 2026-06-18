# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Gromov-Circuit-Wasserstein cross-term between two probabilistic circuits.

Forward recursion couples node pairs across all type combinations (leaf, sum,
product) using the northwest-corner plan at leaves, a transportation LP (with
duals) at sum-vs-sum, a Hungarian assignment at product-vs-product, and an
argmax / max-of-two at the mixed cases. A pre-pass computes per-node expected
distances for both circuits. Gradients (w.r.t. ``circuit2``) are exact
subgradients: discrete choices from the forward solve are frozen and reverse-
mode AD is applied to the remaining smooth structure, including a second
top-down pass over the expected-distance recursion.
"""

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from sparc.grad cimport GradBundle
from sparc.metrics cimport GroundMetric, PNormMetric
from sparc.nodes cimport (
    CategoricalInputNode,
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
    obj_id,
)
from sparc.solvers.assignment cimport assignment_min
from sparc.solvers.northwest cimport (
    nw_backward_marginals,
    nw_plan,
    nw_run,
)
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


# === Tape entries =============================================================

cdef class _GCWLeaf(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef int p_scope
    cdef int q_scope
    cdef vector[int] rows
    cdef vector[int] cols
    cdef vector[double] vals
    cdef vector[int] modes

    cdef void backward(self, object ctx, double g) except *:
        cdef GCWContext c = <GCWContext>ctx
        cdef size_t num = self.rows.size()
        cdef vector[double]* d_p = c.get_dist(self.side_P, self.p_scope, self.n)
        cdef vector[double]* d_q = c.get_dist(self.side_Q, self.q_scope, self.m)
        cdef vector[double] G
        cdef vector[double] adj_p
        cdef vector[double] adj_q
        cdef size_t a
        cdef size_t b
        cdef int i_a
        cdef int j_a
        cdef double sum_b
        cdef object arr
        cdef size_t k
        G.resize(num)
        for a in range(num):
            i_a = self.rows[a]
            j_a = self.cols[a]
            sum_b = 0.0
            for b in range(num):
                sum_b += (
                    self.vals[b]
                    * deref(d_p)[<size_t>i_a * self.n + <size_t>self.rows[b]]
                    * deref(d_q)[<size_t>j_a * self.m + <size_t>self.cols[b]]
                )
            G[a] = 2.0 * sum_b * g
        nw_backward_marginals(self.rows, self.cols, self.modes, G, self.n, self.m, adj_p, adj_q)
        if self.side_P == 1:
            arr = c.cat_grad_arr(1, self.P, self.n)
            for k in range(self.n):
                arr[k] += adj_p[k]
        if self.side_Q == 1:
            arr = c.cat_grad_arr(1, self.Q, self.m)
            for k in range(self.m):
                arr[k] += adj_q[k]


cdef class _GCWSumSum(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef vector[double] w
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
                arr[i] += -g * self.pi[i]
        if self.side_Q == 1:
            arr = c.sum_grad_arr(1, self.Q, self.m)
            for j in range(self.m):
                arr[j] += -g * self.rho[j]


cdef class _GCWProdProd(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef vector[double] d_p
    cdef vector[double] d_q
    cdef list p_children
    cdef list q_children
    cdef vector[int] row_ind
    cdef vector[int] col_ind
    cdef vector[size_t] child_idx

    cdef void backward(self, object ctx, double g) except *:
        cdef GCWContext c = <GCWContext>ctx
        cdef size_t i
        cdef size_t j
        cdef size_t k_idx
        cdef size_t cid
        cdef CircuitNode child
        cdef double sum_d_p = 0.0
        cdef double sum_d_q = 0.0
        cdef size_t num = self.row_ind.size()
        cdef vector[int] matched_q_for_p
        cdef vector[int] matched_p_for_q
        cdef int r
        cdef int col
        matched_q_for_p.assign(self.n, -1)
        matched_p_for_q.assign(self.m, -1)
        for k_idx in range(num):
            r = self.row_ind[k_idx]
            col = self.col_ind[k_idx]
            matched_q_for_p[r] = col
            matched_p_for_q[col] = r
            cid = self.child_idx[<size_t>r * self.m + <size_t>col]
            if cid != NO_TAPE_IDX:
                c.tape_adjoints[cid] += g
        for j in range(self.m):
            sum_d_q += self.d_q[j]
        for i in range(self.n):
            sum_d_p += self.d_p[i]
        if self.side_P == 1:
            for i in range(self.n):
                child = <CircuitNode>self.p_children[i]
                if matched_q_for_p[i] >= 0:
                    c.ed_adj_2[obj_id(child)] += g * (sum_d_q - self.d_q[<size_t>matched_q_for_p[i]])
                else:
                    c.ed_adj_2[obj_id(child)] += g * sum_d_q
        if self.side_Q == 1:
            for j in range(self.m):
                child = <CircuitNode>self.q_children[j]
                if matched_p_for_q[j] >= 0:
                    c.ed_adj_2[obj_id(child)] += g * (sum_d_p - self.d_p[<size_t>matched_p_for_q[j]])
                else:
                    c.ed_adj_2[obj_id(child)] += g * sum_d_p


cdef class _GCWSumOther(TapeEntry):
    cdef size_t nc
    cdef vector[double] theta
    cdef vector[double] V
    cdef vector[size_t] child_idx

    cdef void backward(self, object ctx, double g) except *:
        _bw_sum_other(
            <GCWContext>ctx, self.theta, self.V, self.child_idx, self.nc,
            self.P, self.side_P, g,
        )


cdef class _GCWProdOther(TapeEntry):
    cdef size_t nc
    cdef vector[double] d_p
    cdef double d_q
    cdef size_t best_idx
    cdef vector[size_t] child_idx

    cdef void backward(self, object ctx, double g) except *:
        _bw_prod_other(
            <GCWContext>ctx, <ProductNode>self.P, self.Q, self.d_p, self.d_q,
            self.best_idx, self.child_idx, self.nc, self.side_P, self.side_Q, g,
        )


cdef class _GCWMaxSumProd(TapeEntry):
    cdef int max_winner
    # sum-other branch (sum node is "P")
    cdef CircuitNode sum_node
    cdef int sum_side
    cdef size_t so_nc
    cdef vector[double] so_theta
    cdef vector[double] so_V
    cdef vector[size_t] so_child_idx
    # prod-other branch (prod node is "P")
    cdef ProductNode prod_node
    cdef int prod_side
    cdef size_t po_nc
    cdef vector[double] po_d_p
    cdef double po_d_q
    cdef size_t po_best_idx
    cdef vector[size_t] po_child_idx

    cdef void backward(self, object ctx, double g) except *:
        cdef GCWContext c = <GCWContext>ctx
        if self.max_winner == 0:
            _bw_sum_other(
                c, self.so_theta, self.so_V, self.so_child_idx, self.so_nc,
                self.sum_node, self.sum_side, g,
            )
        else:
            _bw_prod_other(
                c, self.prod_node, self.sum_node, self.po_d_p, self.po_d_q,
                self.po_best_idx, self.po_child_idx, self.po_nc,
                self.prod_side, self.sum_side, g,
            )


cdef void _bw_sum_other(
    GCWContext c,
    vector[double]& theta,
    vector[double]& V,
    vector[size_t]& child_idx,
    size_t nc,
    CircuitNode P_sum,
    int side_sum,
    double g,
) except *:
    cdef size_t i
    cdef size_t cid
    cdef object arr
    for i in range(nc):
        cid = child_idx[i]
        if cid != NO_TAPE_IDX:
            c.tape_adjoints[cid] += g * theta[i]
    if side_sum == 1:
        arr = c.sum_grad_arr(1, P_sum, nc)
        for i in range(nc):
            arr[i] += g * V[i]


cdef void _bw_prod_other(
    GCWContext c,
    ProductNode P_prod,
    CircuitNode Q_other,
    vector[double]& d_p,
    double d_q,
    size_t best_idx,
    vector[size_t]& child_idx,
    size_t nc,
    int side_prod,
    int side_other,
    double g,
) except *:
    cdef size_t i
    cdef size_t cid
    cdef CircuitNode p_child
    cdef double sum_others_d_p = 0.0
    cid = child_idx[best_idx]
    if cid != NO_TAPE_IDX:
        c.tape_adjoints[cid] += g
    for i in range(nc):
        if i != best_idx:
            sum_others_d_p += d_p[i]
    if side_prod == 1:
        for i in range(nc):
            if i != best_idx:
                p_child = P_prod.child_at(i)
                c.ed_adj_2[obj_id(p_child)] += g * d_q
    if side_other == 1:
        c.ed_adj_2[obj_id(Q_other)] += g * sum_others_d_p


# === Context ==================================================================

cdef class GCWContext(CoupleContext):
    cdef GroundMetric metric0
    cdef GroundMetric metric1
    cdef unordered_map[uint64_t, vector[double]] dist_cache0
    cdef unordered_map[uint64_t, vector[double]] dist_cache1
    cdef unordered_map[size_t, double] d_1
    cdef unordered_map[size_t, double] d_2
    cdef unordered_map[size_t, double] ed_adj_2
    cdef list d_2_order
    cdef int _mat_next
    cdef int _mat_offset
    cdef dict _mat_memo
    cdef dict _mat_embed_memo

    def __cinit__(self):
        self.d_2_order = []

    cdef vector[double]* get_dist(self, int side, int scope_var, size_t n) except *:
        cdef uint64_t key = (<uint64_t>scope_var << 32) | <uint64_t>n
        if side == 0:
            if self.dist_cache0.find(key) == self.dist_cache0.end():
                self.metric0.pairwise(scope_var, n, self.dist_cache0[key])
            return &self.dist_cache0[key]
        if self.dist_cache1.find(key) == self.dist_cache1.end():
            self.metric1.pairwise(scope_var, n, self.dist_cache1[key])
        return &self.dist_cache1[key]

    cdef double d_lookup(self, int side, CircuitNode node) except *:
        cdef size_t nid = obj_id(node)
        cdef unordered_map[size_t, double].iterator it
        if side == 0:
            it = self.d_1.find(nid)
            if it == self.d_1.end():
                raise KeyError(f"expected distance missing on side 0 for node {node.id}")
        else:
            it = self.d_2.find(nid)
            if it == self.d_2.end():
                raise KeyError(f"expected distance missing on side 1 for node {node.id}")
        return deref(it).second

    cdef double _ed_node(
        self, CircuitNode node, int side, unordered_map[size_t, double]* cache, list order
    ) except *:
        cdef size_t nid = obj_id(node)
        cdef unordered_map[size_t, double].iterator it = deref(cache).find(nid)
        if it != deref(cache).end():
            return deref(it).second
        cdef double total = 0.0
        cdef size_t i
        cdef size_t nc
        cdef size_t n_out
        cdef size_t a
        cdef size_t b
        cdef double pa
        cdef CircuitNode child
        cdef vector[double]* d_ptr
        cdef FiniteDiscreteInputNode leaf
        cdef SumNode s
        cdef ProductNode prod
        if node.node_kind == NODE_INPUT:
            leaf = <FiniteDiscreteInputNode>node
            n_out = leaf.support_size()
            d_ptr = self.get_dist(side, leaf.scope_var_c(), n_out)
            for a in range(n_out):
                pa = leaf.pmf_at(a)
                for b in range(n_out):
                    total += pa * deref(d_ptr)[a * n_out + b] * leaf.pmf_at(b)
        elif node.node_kind == NODE_SUM:
            s = <SumNode>node
            nc = s.num_children()
            for i in range(nc):
                total += s.parameter_at(i) * self._ed_node(s.child_at(i), side, cache, order)
        elif node.node_kind == NODE_PRODUCT:
            prod = <ProductNode>node
            nc = prod.num_children()
            for i in range(nc):
                total += self._ed_node(prod.child_at(i), side, cache, order)
        else:
            raise TypeError(f"unsupported node type: {type(node).__name__}")
        deref(cache)[nid] = total
        if order is not None:
            order.append(node)
        return total

    cdef double couple_value(self, CircuitNode P, CircuitNode Q, int sP, int sQ) except *:
        cdef double cached
        cdef double res
        cdef int pk
        cdef int qk
        cdef ProductNode Pp
        cdef ProductNode Qp
        if self.memo_get(P, Q, &cached):
            return cached
        pk = P.node_kind
        qk = Q.node_kind
        if pk == NODE_INPUT and qk == NODE_INPUT:
            res = self._leaf(<FiniteDiscreteInputNode>P, <FiniteDiscreteInputNode>Q, sP, sQ)
        elif pk == NODE_SUM and qk == NODE_SUM:
            res = self._sum_sum(<SumNode>P, <SumNode>Q, sP, sQ)
        elif pk == NODE_PRODUCT and qk == NODE_PRODUCT:
            Pp = <ProductNode>P
            Qp = <ProductNode>Q
            if Pp.num_children() < Qp.num_children():
                res = self._prod_prod(Qp, Pp, sQ, sP)
            else:
                res = self._prod_prod(Pp, Qp, sP, sQ)
        elif pk == NODE_PRODUCT and qk == NODE_SUM:
            res = self._max_sum_prod(<SumNode>Q, <ProductNode>P, sQ, sP)
        elif pk == NODE_SUM and qk == NODE_PRODUCT:
            res = self._max_sum_prod(<SumNode>P, <ProductNode>Q, sP, sQ)
        elif pk == NODE_SUM and qk == NODE_INPUT:
            res = self._sum_other(<SumNode>P, Q, sP, sQ)
        elif pk == NODE_INPUT and qk == NODE_SUM:
            res = self._sum_other(<SumNode>Q, P, sQ, sP)
        elif pk == NODE_INPUT and qk == NODE_PRODUCT:
            res = self._prod_other(<ProductNode>Q, P, sQ, sP)
        elif pk == NODE_PRODUCT and qk == NODE_INPUT:
            res = self._prod_other(<ProductNode>P, Q, sP, sQ)
        else:
            raise NotImplementedError("GCW coupling not implemented for this pair")
        self.memo_put(P, Q, res)
        return res

    cdef double _leaf(self, FiniteDiscreteInputNode P, FiniteDiscreteInputNode Q, int sP, int sQ) except *:
        cdef size_t n = P.support_size()
        cdef size_t m = Q.support_size()
        cdef int p_scope = P.scope_var_c()
        cdef int q_scope = Q.scope_var_c()
        cdef vector[double]* d_p = self.get_dist(sP, p_scope, n)
        cdef vector[double]* d_q = self.get_dist(sQ, q_scope, m)
        cdef vector[double] p_pmf
        cdef vector[double] q_pmf
        cdef vector[int] rows
        cdef vector[int] cols
        cdef vector[double] vals
        cdef vector[int] modes
        cdef double value
        cdef _GCWLeaf entry
        _leaf_pmf(P, p_pmf)
        _leaf_pmf(Q, q_pmf)
        value = nw_run(p_pmf, q_pmf, deref(d_p), deref(d_q), n, m, rows, cols, vals, modes)
        if self.recording:
            entry = _GCWLeaf()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.p_scope = p_scope
            entry.q_scope = q_scope
            entry.rows = rows
            entry.cols = cols
            entry.vals = vals
            entry.modes = modes
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
        cdef _GCWSumSum entry
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
        cost.resize(n * m)
        for i in range(n):
            for j in range(m):
                cost[i * m + j] = -V[i * m + j]  # GCW maximizes <V, w>
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
            entry = _GCWSumSum()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.w = plan
            entry.pi = u
            entry.rho = v
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return cross_term

    cdef double _prod_prod(self, ProductNode P, ProductNode Q, int sP, int sQ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef CircuitNode pc
        cdef CircuitNode qc
        cdef double c_cost
        cdef double ed_val
        cdef double base_cost = 0.0
        cdef double gw_cost
        cdef vector[double] pairwise
        cdef vector[double] cost
        cdef vector[double] d_p
        cdef vector[double] d_q
        cdef vector[int] row_ind
        cdef vector[int] col_ind
        cdef vector[size_t] child_idx
        cdef size_t k
        cdef size_t num
        cdef _GCWProdProd entry
        cdef list p_children = []
        cdef list q_children = []
        pairwise.resize(n * m)
        cost.resize(n * m)
        d_p.resize(n)
        d_q.resize(m)
        if self.recording:
            child_idx.resize(n * m)
        for i in range(n):
            p_children.append(P.child_at(i))
        for j in range(m):
            q_children.append(Q.child_at(j))
        for j in range(m):
            d_q[j] = self.d_lookup(sQ, <CircuitNode>q_children[j])
        for i in range(n):
            d_p[i] = self.d_lookup(sP, <CircuitNode>p_children[i])
        for i in range(n):
            pc = <CircuitNode>p_children[i]
            for j in range(m):
                qc = <CircuitNode>q_children[j]
                c_cost = self.couple_value(pc, qc, sP, sQ)
                ed_val = d_p[i] * d_q[j]
                pairwise[i * m + j] = c_cost - ed_val
                cost[i * m + j] = ed_val - c_cost  # minimize -> maximize pairwise
                base_cost += ed_val
                if self.recording:
                    child_idx[i * m + j] = self.lookup_pair_tape_idx(pc, qc)
        assignment_min(cost, n, m, row_ind, col_ind)
        gw_cost = base_cost
        num = row_ind.size()
        for k in range(num):
            gw_cost += pairwise[<size_t>row_ind[k] * m + <size_t>col_ind[k]]
        if self.recording:
            entry = _GCWProdProd()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.d_p = d_p
            entry.d_q = d_q
            entry.p_children = p_children
            entry.q_children = q_children
            entry.row_ind = row_ind
            entry.col_ind = col_ind
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return gw_cost

    cdef double _sum_other(self, SumNode P, CircuitNode Q, int sP, int sQ) except *:
        cdef size_t nc = P.num_children()
        cdef size_t i
        cdef CircuitNode pc
        cdef double total = 0.0
        cdef double v_i
        cdef vector[double] V
        cdef vector[double] theta
        cdef vector[size_t] child_idx
        cdef _GCWSumOther entry
        V.resize(nc)
        theta.resize(nc)
        if self.recording:
            child_idx.resize(nc)
        for i in range(nc):
            pc = P.child_at(i)
            v_i = self.couple_value(pc, Q, sP, sQ)
            V[i] = v_i
            theta[i] = P.parameter_at(i)
            total += theta[i] * v_i
            if self.recording:
                child_idx[i] = self.lookup_pair_tape_idx(pc, Q)
        if self.recording:
            entry = _GCWSumOther()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.nc = nc
            entry.theta = theta
            entry.V = V
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return total

    cdef double _prod_other(self, ProductNode P, CircuitNode Q, int sP, int sQ) except *:
        cdef size_t nc = P.num_children()
        cdef size_t i
        cdef size_t best_idx = 0
        cdef CircuitNode pc
        cdef vector[double] V
        cdef vector[double] d_p
        cdef double d_q = self.d_lookup(sQ, Q)
        cdef double total_cost
        cdef double best_val = -1e300
        cdef double adjusted
        cdef vector[size_t] child_idx
        cdef _GCWProdOther entry
        V.resize(nc)
        d_p.resize(nc)
        if self.recording:
            child_idx.resize(nc)
        for i in range(nc):
            pc = P.child_at(i)
            V[i] = self.couple_value(pc, Q, sP, sQ)
            d_p[i] = self.d_lookup(sP, pc)
            if self.recording:
                child_idx[i] = self.lookup_pair_tape_idx(pc, Q)
        for i in range(nc):
            adjusted = V[i] - d_p[i] * d_q
            if adjusted > best_val:
                best_val = adjusted
                best_idx = i
        total_cost = V[best_idx]
        for i in range(nc):
            if i != best_idx:
                total_cost += d_p[i] * d_q
        if self.recording:
            entry = _GCWProdOther()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.nc = nc
            entry.d_p = d_p
            entry.d_q = d_q
            entry.best_idx = best_idx
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return total_cost

    cdef double _max_sum_prod(self, SumNode P_sum, ProductNode Q_prod, int sP_sum, int sQ_prod) except *:
        cdef size_t nc_sum = P_sum.num_children()
        cdef size_t nc_prod = Q_prod.num_children()
        cdef size_t i
        cdef size_t best_idx = 0
        cdef CircuitNode sum_child
        cdef CircuitNode prod_child
        cdef vector[double] so_V
        cdef vector[double] so_theta
        cdef vector[size_t] so_idx
        cdef vector[double] po_V
        cdef vector[double] po_d_p
        cdef vector[size_t] po_idx
        cdef double d_q_sum
        cdef double v_i
        cdef double adjusted
        cdef double best_val = -1e300
        cdef double res1 = 0.0
        cdef double res2
        cdef int max_winner
        cdef _GCWMaxSumProd entry
        so_V.resize(nc_sum)
        so_theta.resize(nc_sum)
        if self.recording:
            so_idx.resize(nc_sum)
        for i in range(nc_sum):
            sum_child = P_sum.child_at(i)
            v_i = self.couple_value(sum_child, Q_prod, sP_sum, sQ_prod)
            so_V[i] = v_i
            so_theta[i] = P_sum.parameter_at(i)
            res1 += so_theta[i] * v_i
            if self.recording:
                so_idx[i] = self.lookup_pair_tape_idx(sum_child, Q_prod)
        po_V.resize(nc_prod)
        po_d_p.resize(nc_prod)
        if self.recording:
            po_idx.resize(nc_prod)
        d_q_sum = self.d_lookup(sP_sum, P_sum)
        for i in range(nc_prod):
            prod_child = Q_prod.child_at(i)
            po_V[i] = self.couple_value(prod_child, P_sum, sQ_prod, sP_sum)
            po_d_p[i] = self.d_lookup(sQ_prod, prod_child)
            if self.recording:
                po_idx[i] = self.lookup_pair_tape_idx(prod_child, P_sum)
        for i in range(nc_prod):
            adjusted = po_V[i] - po_d_p[i] * d_q_sum
            if adjusted > best_val:
                best_val = adjusted
                best_idx = i
        res2 = po_V[best_idx]
        for i in range(nc_prod):
            if i != best_idx:
                res2 += po_d_p[i] * d_q_sum
        max_winner = 0 if res1 >= res2 else 1
        if self.recording:
            entry = _GCWMaxSumProd()
            entry.P = P_sum
            entry.Q = Q_prod
            entry.side_P = sP_sum
            entry.side_Q = sQ_prod
            entry.max_winner = max_winner
            entry.sum_node = P_sum
            entry.sum_side = sP_sum
            entry.so_nc = nc_sum
            entry.so_theta = so_theta
            entry.so_V = so_V
            entry.so_child_idx = so_idx
            entry.prod_node = Q_prod
            entry.prod_side = sQ_prod
            entry.po_nc = nc_prod
            entry.po_d_p = po_d_p
            entry.po_d_q = d_q_sum
            entry.po_best_idx = best_idx
            entry.po_child_idx = po_idx
            self.append_tape(entry, P_sum, Q_prod)
        return res1 if res1 >= res2 else res2

    cdef void _ed_backward(self) except *:
        cdef ssize_t k
        cdef CircuitNode node
        cdef size_t nid
        cdef double adj
        cdef unordered_map[size_t, double].iterator it
        cdef size_t i
        cdef size_t nc
        cdef size_t n_out
        cdef size_t kk
        cdef size_t j
        cdef CircuitNode child
        cdef size_t cid
        cdef object arr
        cdef vector[double]* d_ptr
        cdef double accum
        cdef double child_E
        cdef FiniteDiscreteInputNode leaf
        cdef SumNode s
        cdef ProductNode prod
        cdef unordered_map[size_t, double].iterator d_it
        for k in range(<ssize_t>len(self.d_2_order) - 1, -1, -1):
            node = <CircuitNode>self.d_2_order[k]
            nid = obj_id(node)
            it = self.ed_adj_2.find(nid)
            if it == self.ed_adj_2.end():
                continue
            adj = deref(it).second
            if adj == 0.0:
                continue
            if node.node_kind == NODE_INPUT:
                leaf = <FiniteDiscreteInputNode>node
                n_out = leaf.support_size()
                d_ptr = self.get_dist(1, leaf.scope_var_c(), n_out)
                arr = self.cat_grad_arr(1, leaf, n_out)
                for kk in range(n_out):
                    accum = 0.0
                    for j in range(n_out):
                        accum += deref(d_ptr)[kk * n_out + j] * leaf.pmf_at(j)
                    arr[kk] += 2.0 * adj * accum
            elif node.node_kind == NODE_SUM:
                s = <SumNode>node
                nc = s.num_children()
                arr = self.sum_grad_arr(1, s, nc)
                for i in range(nc):
                    child = s.child_at(i)
                    cid = obj_id(child)
                    d_it = self.d_2.find(cid)
                    child_E = 0.0 if d_it == self.d_2.end() else deref(d_it).second
                    arr[i] += adj * child_E
                    self.ed_adj_2[cid] = self.ed_adj_2[cid] + adj * s.parameter_at(i)
            elif node.node_kind == NODE_PRODUCT:
                prod = <ProductNode>node
                nc = prod.num_children()
                for i in range(nc):
                    child = prod.child_at(i)
                    self.ed_adj_2[obj_id(child)] = self.ed_adj_2[obj_id(child)] + adj

    cdef void reset_gcw(self):
        self.reset_base()
        self.d_1.clear()
        self.d_2.clear()
        self.dist_cache0.clear()
        self.dist_cache1.clear()
        self.ed_adj_2.clear()
        self.d_2_order = []

    cdef double solve(self, CircuitNode c1, CircuitNode c2) except *:
        self.reset_gcw()
        self._ed_node(c1, 0, &self.d_1, None)
        self._ed_node(c2, 1, &self.d_2, self.d_2_order)
        return self.couple_value(c1, c2, 0, 1)

    # === Coupling-circuit materialization ====================================
    # Builds a PC over vars(c1) U (vars(c2) + offset) whose ancestral sampling
    # draws a joint (x, y) pair distributed according to the GCW coupling. The
    # structure mirrors the forward solve: NW plan -> leaf mixture, transport
    # plan -> sum mixture, Hungarian matching -> product over matched pairs
    # (+ marginal embeddings of unmatched children), argmax -> product with the
    # winning couple plus the losing children's marginals.

    cdef int _alloc(self) except *:
        cdef int nid = self._mat_next
        self._mat_next += 1
        return nid

    cdef object _det_cat(self, int var, size_t outcome, size_t n):
        cdef list probs = [0.0] * <Py_ssize_t>n
        probs[<Py_ssize_t>outcome] = 1.0
        return CategoricalInputNode(self._alloc(), var, probs)

    cdef int _var_of(self, FiniteDiscreteInputNode leaf, int side) except *:
        return leaf.scope_var_c() + (self._mat_offset if side == 1 else 0)

    cdef object _embed(self, CircuitNode node, int side):
        """Clone a single circuit's subtree into the coupling space, offsetting
        circuit2 (``side == 1``) variables so the two namespaces stay disjoint."""
        cdef uint64_t key = (<uint64_t>obj_id(node) << 1) | <uint64_t>side
        cdef object cached = self._mat_embed_memo.get(key)
        cdef FiniteDiscreteInputNode leaf
        cdef SumNode s
        cdef ProductNode prod
        cdef list children
        cdef list probs
        cdef size_t i
        cdef size_t nc
        cdef size_t k
        if cached is not None:
            return cached
        cdef object out
        if node.node_kind == NODE_INPUT:
            leaf = <FiniteDiscreteInputNode>node
            nc = leaf.support_size()
            probs = [leaf.pmf_at(k) for k in range(nc)]
            out = CategoricalInputNode(self._alloc(), self._var_of(leaf, side), probs)
        elif node.node_kind == NODE_SUM:
            s = <SumNode>node
            nc = s.num_children()
            children = [self._embed(s.child_at(i), side) for i in range(nc)]
            out = SumNode(self._alloc(), children,
                          [s.parameter_at(i) for i in range(nc)])
        elif node.node_kind == NODE_PRODUCT:
            prod = <ProductNode>node
            nc = prod.num_children()
            children = [self._embed(prod.child_at(i), side) for i in range(nc)]
            out = ProductNode(self._alloc(), children)
        else:
            raise TypeError(f"cannot embed node type {type(node).__name__}")
        self._mat_embed_memo[key] = out
        return out

    cdef object _materialize(self, CircuitNode P, CircuitNode Q, int sP, int sQ):
        cdef uint64_t key = (<uint64_t>obj_id(P) << 32) | <uint64_t>obj_id(Q)
        cdef object cached = self._mat_memo.get(key)
        if cached is not None:
            return cached
        cdef int pk = P.node_kind
        cdef int qk = Q.node_kind
        cdef object out
        cdef ProductNode Pp
        cdef ProductNode Qp
        if pk == NODE_INPUT and qk == NODE_INPUT:
            out = self._mat_leaf(<FiniteDiscreteInputNode>P, <FiniteDiscreteInputNode>Q, sP, sQ)
        elif pk == NODE_SUM and qk == NODE_SUM:
            out = self._mat_sum_sum(<SumNode>P, <SumNode>Q, sP, sQ)
        elif pk == NODE_PRODUCT and qk == NODE_PRODUCT:
            Pp = <ProductNode>P
            Qp = <ProductNode>Q
            if Pp.num_children() < Qp.num_children():
                out = self._mat_prod_prod(Qp, Pp, sQ, sP)
            else:
                out = self._mat_prod_prod(Pp, Qp, sP, sQ)
        elif pk == NODE_PRODUCT and qk == NODE_SUM:
            out = self._mat_max_sum_prod(<SumNode>Q, <ProductNode>P, sQ, sP)
        elif pk == NODE_SUM and qk == NODE_PRODUCT:
            out = self._mat_max_sum_prod(<SumNode>P, <ProductNode>Q, sP, sQ)
        elif pk == NODE_SUM and qk == NODE_INPUT:
            out = self._mat_sum_other(<SumNode>P, Q, sP, sQ)
        elif pk == NODE_INPUT and qk == NODE_SUM:
            out = self._mat_sum_other(<SumNode>Q, P, sQ, sP)
        elif pk == NODE_INPUT and qk == NODE_PRODUCT:
            out = self._mat_prod_other(<ProductNode>Q, P, sQ, sP)
        elif pk == NODE_PRODUCT and qk == NODE_INPUT:
            out = self._mat_prod_other(<ProductNode>P, Q, sP, sQ)
        else:
            raise NotImplementedError("GCW materialization not implemented for this pair")
        self._mat_memo[key] = out
        return out

    cdef object _mat_leaf(self, FiniteDiscreteInputNode P, FiniteDiscreteInputNode Q, int sP, int sQ):
        cdef size_t n = P.support_size()
        cdef size_t m = Q.support_size()
        cdef int p_var = self._var_of(P, sP)
        cdef int q_var = self._var_of(Q, sQ)
        cdef vector[double] p_pmf
        cdef vector[double] q_pmf
        cdef vector[int] rows
        cdef vector[int] cols
        cdef vector[double] vals
        cdef vector[int] modes
        cdef size_t a
        cdef size_t num
        cdef double total = 0.0
        cdef list children = []
        cdef list weights = []
        _leaf_pmf(P, p_pmf)
        _leaf_pmf(Q, q_pmf)
        nw_plan(p_pmf, q_pmf, n, m, rows, cols, vals, modes)
        num = rows.size()
        for a in range(num):
            if vals[a] > 1e-15:
                children.append(ProductNode(self._alloc(), [
                    self._det_cat(p_var, <size_t>rows[a], n),
                    self._det_cat(q_var, <size_t>cols[a], m),
                ]))
                weights.append(vals[a])
                total += vals[a]
        weights = [w / total for w in weights]
        return SumNode(self._alloc(), children, weights)

    cdef object _mat_sum_sum(self, SumNode P, SumNode Q, int sP, int sQ):
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef vector[double] V
        cdef vector[double] cost
        cdef vector[double] theta
        cdef vector[double] phi
        cdef vector[double] plan
        cdef vector[double] u
        cdef vector[double] v
        cdef double w
        cdef double total = 0.0
        cdef list children = []
        cdef list weights = []
        V.resize(n * m)
        for i in range(n):
            for j in range(m):
                V[i * m + j] = self.couple_value(P.child_at(i), Q.child_at(j), sP, sQ)
        cost.resize(n * m)
        for i in range(n):
            for j in range(m):
                cost[i * m + j] = -V[i * m + j]
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
                if w > 1e-15:
                    children.append(self._materialize(P.child_at(i), Q.child_at(j), sP, sQ))
                    weights.append(w)
                    total += w
        weights = [x / total for x in weights]
        return SumNode(self._alloc(), children, weights)

    cdef object _mat_prod_prod(self, ProductNode P, ProductNode Q, int sP, int sQ):
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef size_t k
        cdef double c_cost
        cdef double ed_val
        cdef vector[double] cost
        cdef vector[int] row_ind
        cdef vector[int] col_ind
        cdef list children = []
        cdef list matched_p = [False] * <Py_ssize_t>n
        cdef int r
        cdef int col
        cost.resize(n * m)
        for i in range(n):
            for j in range(m):
                c_cost = self.couple_value(P.child_at(i), Q.child_at(j), sP, sQ)
                ed_val = self.d_lookup(sP, P.child_at(i)) * self.d_lookup(sQ, Q.child_at(j))
                cost[i * m + j] = ed_val - c_cost
        assignment_min(cost, n, m, row_ind, col_ind)
        for k in range(row_ind.size()):
            r = row_ind[k]
            col = col_ind[k]
            matched_p[r] = True
            children.append(self._materialize(P.child_at(<size_t>r), Q.child_at(<size_t>col), sP, sQ))
        # P has >= arity (swapped); embed any unmatched P children as marginals
        for i in range(n):
            if not matched_p[i]:
                children.append(self._embed(P.child_at(i), sP))
        return ProductNode(self._alloc(), children)

    cdef object _mat_sum_other(self, SumNode P, CircuitNode Q, int sP, int sQ):
        cdef size_t nc = P.num_children()
        cdef size_t i
        cdef list children = []
        cdef list weights = []
        for i in range(nc):
            children.append(self._materialize(P.child_at(i), Q, sP, sQ))
            weights.append(P.parameter_at(i))
        return SumNode(self._alloc(), children, weights)

    cdef object _mat_prod_other(self, ProductNode P, CircuitNode Q, int sP, int sQ):
        cdef size_t nc = P.num_children()
        cdef size_t i
        cdef size_t best_idx = 0
        cdef double d_q = self.d_lookup(sQ, Q)
        cdef double best_val = -1e300
        cdef double adjusted
        cdef list children = []
        for i in range(nc):
            adjusted = self.couple_value(P.child_at(i), Q, sP, sQ) - self.d_lookup(sP, P.child_at(i)) * d_q
            if adjusted > best_val:
                best_val = adjusted
                best_idx = i
        children.append(self._materialize(P.child_at(best_idx), Q, sP, sQ))
        for i in range(nc):
            if i != best_idx:
                children.append(self._embed(P.child_at(i), sP))
        return ProductNode(self._alloc(), children)

    cdef object _mat_max_sum_prod(self, SumNode P_sum, ProductNode Q_prod, int sP_sum, int sQ_prod):
        cdef size_t nc_sum = P_sum.num_children()
        cdef size_t nc_prod = Q_prod.num_children()
        cdef size_t i
        cdef size_t best_idx = 0
        cdef double d_q_sum = self.d_lookup(sP_sum, P_sum)
        cdef double res1 = 0.0
        cdef double res2
        cdef double best_val = -1e300
        cdef double adjusted
        for i in range(nc_sum):
            res1 += P_sum.parameter_at(i) * self.couple_value(P_sum.child_at(i), Q_prod, sP_sum, sQ_prod)
        for i in range(nc_prod):
            adjusted = self.couple_value(Q_prod.child_at(i), P_sum, sQ_prod, sP_sum) \
                - self.d_lookup(sQ_prod, Q_prod.child_at(i)) * d_q_sum
            if adjusted > best_val:
                best_val = adjusted
                best_idx = i
        res2 = self.couple_value(Q_prod.child_at(best_idx), P_sum, sQ_prod, sP_sum)
        for i in range(nc_prod):
            if i != best_idx:
                res2 += self.d_lookup(sQ_prod, Q_prod.child_at(i)) * d_q_sum
        if res1 >= res2:
            return self._mat_sum_other(P_sum, Q_prod, sP_sum, sQ_prod)
        return self._mat_prod_other(Q_prod, P_sum, sQ_prod, sP_sum)

    cdef object materialize(self, CircuitNode c1, CircuitNode c2):
        c1.propagate_scope()
        c2.propagate_scope()
        cdef list vars1 = c1.scope_as_list()
        cdef int off = 0
        cdef int vv
        for vv in vars1:
            if vv + 1 > off:
                off = vv + 1
        self._mat_offset = off
        self._mat_next = 0
        self._mat_memo = {}
        self._mat_embed_memo = {}
        self.reset_gcw()
        self._ed_node(c1, 0, &self.d_1, None)
        self._ed_node(c2, 1, &self.d_2, self.d_2_order)
        self.couple_value(c1, c2, 0, 1)
        return self._materialize(c1, c2, 0, 1)


cpdef double gcw_crossterm(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor_1=1.0,
    double scale_factor_2=1.0,
    object metric1=None,
    object metric2=None,
) except *:
    """Compute the GCW cross-term between two probabilistic circuits."""
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef GCWContext ctx = GCWContext()
    ctx.metric0 = metric1 if metric1 is not None else PNormMetric(metric_p, scale_factor_1)
    ctx.metric1 = metric2 if metric2 is not None else PNormMetric(metric_p, scale_factor_2)
    return ctx.solve(r1, r2)


cpdef object gcw_coupling_circuit(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor_1=1.0,
    double scale_factor_2=1.0,
    object metric1=None,
    object metric2=None,
):
    """Materialize the GCW coupling as a probabilistic circuit.

    Returns a :class:`~sparc.circuit.Circuit` over ``vars(circuit1)`` together
    with ``vars(circuit2)`` shifted by ``max(vars(circuit1)) + 1`` (so the two
    variable namespaces are disjoint). Ancestral sampling from the returned
    circuit draws a joint ``(x, y)`` pair distributed according to the coupling
    induced by the GCW solve. Its structure mirrors the recursion: leaf NW plans
    become outcome-pair mixtures, sum-sum transport plans become weighted
    mixtures over coupled children, product-product Hungarian matchings become
    products over matched pairs (unmatched children embedded as marginals), and
    the mixed sum/product argmax becomes a product of the winning couple with
    the losing children's marginals.
    """
    from sparc.circuit import Circuit
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef GCWContext ctx = GCWContext()
    ctx.metric0 = metric1 if metric1 is not None else PNormMetric(metric_p, scale_factor_1)
    ctx.metric1 = metric2 if metric2 is not None else PNormMetric(metric_p, scale_factor_2)
    cdef CircuitNode root = <CircuitNode>ctx.materialize(r1, r2)
    root.propagate_scope()
    return Circuit(root)


cpdef tuple gcw_crossterm_and_grad(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor_1=1.0,
    double scale_factor_2=1.0,
    object metric1=None,
    object metric2=None,
):
    """Compute the GCW cross-term and its subgradients w.r.t. ``circuit2``.

    Returns ``(value, grads)`` with ``grads`` a :class:`~sparc.grad.GradBundle`
    over ``circuit2`` nodes.
    """
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef GCWContext ctx = GCWContext()
    cdef double value
    cdef size_t root_idx
    cdef GradBundle grads
    ctx.metric0 = metric1 if metric1 is not None else PNormMetric(metric_p, scale_factor_1)
    ctx.metric1 = metric2 if metric2 is not None else PNormMetric(metric_p, scale_factor_2)
    ctx.recording = True
    try:
        value = ctx.solve(r1, r2)
        root_idx = ctx.lookup_pair_tape_idx(r1, r2)
        if root_idx == NO_TAPE_IDX:
            raise RuntimeError("internal: root pair has no tape entry")
        ctx.tape_adjoints[root_idx] = 1.0
        ctx.run_backward()
        ctx._ed_backward()
    finally:
        ctx.recording = False
    grads = GradBundle()
    grads.value = value
    grads.sum_grads = ctx.sum_grads1
    grads.cat_grads = ctx.cat_grads1
    return (value, grads)
