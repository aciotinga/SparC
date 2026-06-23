# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Tractable inner-product queries between two compatible PCs.

``exp_query``     : E_Q[P(X)] = sum_x P(x) Q(x)
``log_exp_query`` : log(E_Q[P(X)]) with log-sum-exp stabilization

Both return exact reverse-mode gradients for *both* circuits. Built on the
shared :class:`~sparc.queries._engine.CoupleContext`.
"""

from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.math cimport exp, INFINITY, isfinite, log

import numpy as np

from sparc._graph cimport CompiledCircuit, match_prod_children_flat
from sparc._mathutils cimport sp_safe_log
from sparc.grad cimport GradBundle
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
    cdef bint cc1 = isinstance(c1, CompiledCircuit)
    cdef bint cc2 = isinstance(c2, CompiledCircuit)
    if cc1 != cc2:
        raise TypeError(
            "pairwise queries require both operands to be the same kind: "
            "either both Circuit/CircuitNode or both CompiledCircuit"
        )


cdef void _check_leaf_compat(
    FiniteDiscreteInputNode P, FiniteDiscreteInputNode Q
) except *:
    if P.scope_var_c() != Q.scope_var_c():
        raise ValueError(
            "expectation incompatible: leaf nodes have different scopes"
        )
    if P.support_size() != Q.support_size():
        raise ValueError(
            "expectation incompatible: leaf nodes have different cardinalities "
            f"({P.support_size()} vs {Q.support_size()})"
        )


# === exp_query: linear E_Q[P] =================================================

cdef class _ExpLeaf(TapeEntry):
    cdef size_t n

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef FiniteDiscreteInputNode P = <FiniteDiscreteInputNode>self.P
        cdef FiniteDiscreteInputNode Q = <FiniteDiscreteInputNode>self.Q
        cdef size_t k
        cdef object gp = c.cat_grad_arr(0, P, self.n)
        cdef object gq = c.cat_grad_arr(1, Q, self.n)
        for k in range(self.n):
            gp[k] += g * Q.pmf_at(k)
            gq[k] += g * P.pmf_at(k)


cdef class _ExpSumSum(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef vector[double] V
    cdef vector[double] theta
    cdef vector[double] phi
    cdef vector[size_t] child_idx

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef size_t i
        cdef size_t j
        cdef size_t cid
        cdef double s
        cdef object gp
        cdef object gq
        for i in range(self.n):
            for j in range(self.m):
                cid = self.child_idx[i * self.m + j]
                if cid != NO_TAPE_IDX:
                    c.tape_adjoints[cid] += g * self.theta[i] * self.phi[j]
        gp = c.sum_grad_arr(0, self.P, self.n)
        for i in range(self.n):
            s = 0.0
            for j in range(self.m):
                s += self.phi[j] * self.V[i * self.m + j]
            gp[i] += g * s
        gq = c.sum_grad_arr(1, self.Q, self.m)
        for j in range(self.m):
            s = 0.0
            for i in range(self.n):
                s += self.theta[i] * self.V[i * self.m + j]
            gq[j] += g * s


cdef class _ExpProdProd(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef vector[double] child_vals
    cdef vector[int] row_ind
    cdef vector[int] col_ind
    cdef vector[size_t] child_idx

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef size_t i
        cdef size_t k_idx
        cdef size_t cid
        cdef int r
        cdef double factor
        cdef vector[double] prefix
        cdef vector[double] suffix
        cdef size_t num = self.row_ind.size()
        if self.n == 0:
            return
        prefix.resize(self.n)
        suffix.resize(self.n)
        prefix[0] = 1.0
        for i in range(1, self.n):
            prefix[i] = prefix[i - 1] * self.child_vals[i - 1]
        suffix[self.n - 1] = 1.0
        for i in range(<ssize_t>self.n - 1, 0, -1):
            suffix[<size_t>i - 1] = suffix[<size_t>i] * self.child_vals[<size_t>i]
        for k_idx in range(num):
            r = self.row_ind[k_idx]
            factor = g * prefix[<size_t>r] * suffix[<size_t>r]
            cid = self.child_idx[<size_t>r * self.m + <size_t>self.col_ind[k_idx]]
            if cid != NO_TAPE_IDX:
                c.tape_adjoints[cid] += factor


cdef class ExpectationContext(CoupleContext):
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
                f"expectation incompatible: cannot couple {type(P).__name__} "
                f"with {type(Q).__name__}"
            )
        self.memo_put(P, Q, res)
        return res

    cdef double _leaf(self, FiniteDiscreteInputNode P, FiniteDiscreteInputNode Q, int sP, int sQ) except *:
        _check_leaf_compat(P, Q)
        cdef size_t n = P.support_size()
        cdef size_t k
        cdef double total = 0.0
        cdef _ExpLeaf entry
        for k in range(n):
            total += P.pmf_at(k) * Q.pmf_at(k)
        if self.recording:
            entry = _ExpLeaf()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            self.append_tape(entry, P, Q)
        return total

    cdef double _sum_sum(self, SumNode P, SumNode Q, int sP, int sQ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef CircuitNode pc
        cdef CircuitNode qc
        cdef vector[double] V
        cdef vector[size_t] child_idx
        cdef double total = 0.0
        cdef double v
        cdef _ExpSumSum entry
        V.resize(n * m)
        if self.recording:
            child_idx.resize(n * m)
        for i in range(n):
            pc = P.child_at(i)
            for j in range(m):
                qc = Q.child_at(j)
                v = self.couple_value(pc, qc, sP, sQ)
                V[i * m + j] = v
                if self.recording:
                    child_idx[i * m + j] = self.lookup_pair_tape_idx(pc, qc)
        for i in range(n):
            for j in range(m):
                total += P.parameter_at(i) * Q.parameter_at(j) * V[i * m + j]
        if self.recording:
            entry = _ExpSumSum()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.V = V
            entry.child_idx = child_idx
            entry.theta.resize(n)
            entry.phi.resize(m)
            for i in range(n):
                entry.theta[i] = P.parameter_at(i)
            for j in range(m):
                entry.phi[j] = Q.parameter_at(j)
            self.append_tape(entry, P, Q)
        return total

    cdef double _prod_prod(self, ProductNode P, ProductNode Q, int sP, int sQ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef vector[int] row_ind
        cdef vector[int] col_ind
        cdef size_t i
        cdef int q_idx
        cdef CircuitNode pc
        cdef CircuitNode qc
        cdef vector[double] child_vals
        cdef vector[size_t] child_idx
        cdef double total = 1.0
        cdef double v
        cdef size_t t
        cdef _ExpProdProd entry
        match_prod_children(P, Q, row_ind, col_ind, "expectation")
        child_vals.resize(n)
        if self.recording:
            child_idx.resize(n * m)
            for t in range(n * m):
                child_idx[t] = NO_TAPE_IDX
        for i in range(n):
            q_idx = col_ind[i]
            pc = P.child_at(i)
            qc = Q.child_at(<size_t>q_idx)
            v = self.couple_value(pc, qc, sP, sQ)
            child_vals[i] = v
            total *= v
            if self.recording:
                child_idx[i * m + <size_t>q_idx] = self.lookup_pair_tape_idx(pc, qc)
        if self.recording:
            entry = _ExpProdProd()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.child_vals = child_vals
            entry.row_ind = row_ind
            entry.col_ind = col_ind
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return total


# === log_exp_query: log E_Q[P] ================================================

cdef class _LogExpLeaf(TapeEntry):
    cdef size_t n
    cdef double ell

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef FiniteDiscreteInputNode P = <FiniteDiscreteInputNode>self.P
        cdef FiniteDiscreteInputNode Q = <FiniteDiscreteInputNode>self.Q
        cdef size_t k
        cdef object gp = c.cat_grad_arr(0, P, self.n)
        cdef object gq = c.cat_grad_arr(1, Q, self.n)
        for k in range(self.n):
            gp[k] += g * exp(sp_safe_log(Q.pmf_at(k)) - self.ell)
            gq[k] += g * exp(sp_safe_log(P.pmf_at(k)) - self.ell)


cdef class _LogExpSumSum(TapeEntry):
    cdef size_t n
    cdef size_t m
    cdef vector[double] V
    cdef vector[double] log_theta
    cdef vector[double] log_phi
    cdef vector[size_t] child_idx
    cdef double ell

    cdef void backward(self, object ctx, double g) except *:
        cdef CoupleContext c = <CoupleContext>ctx
        cdef size_t i
        cdef size_t j
        cdef size_t cid
        cdef double w
        cdef double s
        cdef object gp
        cdef object gq
        for i in range(self.n):
            for j in range(self.m):
                w = g * exp(self.log_theta[i] + self.log_phi[j] + self.V[i * self.m + j] - self.ell)
                cid = self.child_idx[i * self.m + j]
                if cid != NO_TAPE_IDX:
                    c.tape_adjoints[cid] += w
        gp = c.sum_grad_arr(0, self.P, self.n)
        for i in range(self.n):
            s = 0.0
            for j in range(self.m):
                s += exp(self.log_phi[j] + self.V[i * self.m + j] - self.ell)
            gp[i] += g * s
        gq = c.sum_grad_arr(1, self.Q, self.m)
        for j in range(self.m):
            s = 0.0
            for i in range(self.n):
                s += exp(self.log_theta[i] + self.V[i * self.m + j] - self.ell)
            gq[j] += g * s


cdef class _LogExpProdProd(TapeEntry):
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


cdef class LogExpectationContext(CoupleContext):
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
                f"expectation incompatible: cannot couple {type(P).__name__} "
                f"with {type(Q).__name__}"
            )
        self.memo_put(P, Q, res)
        return res

    cdef double _leaf(self, FiniteDiscreteInputNode P, FiniteDiscreteInputNode Q, int sP, int sQ) except *:
        _check_leaf_compat(P, Q)
        cdef size_t n = P.support_size()
        cdef size_t k
        cdef double max_val = -INFINITY
        cdef double total = 0.0
        cdef double term
        cdef double ell
        cdef _LogExpLeaf entry
        for k in range(n):
            term = sp_safe_log(P.pmf_at(k)) + sp_safe_log(Q.pmf_at(k))
            if isfinite(term) and term > max_val:
                max_val = term
        for k in range(n):
            term = sp_safe_log(P.pmf_at(k)) + sp_safe_log(Q.pmf_at(k))
            if isfinite(term):
                total += exp(term - max_val)
        ell = -INFINITY if total <= 0.0 else max_val + log(total)
        if self.recording:
            entry = _LogExpLeaf()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.ell = ell
            self.append_tape(entry, P, Q)
        return ell

    cdef double _sum_sum(self, SumNode P, SumNode Q, int sP, int sQ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef CircuitNode pc
        cdef CircuitNode qc
        cdef vector[double] V
        cdef vector[size_t] child_idx
        cdef double max_val = -INFINITY
        cdef double total = 0.0
        cdef double lt
        cdef double lp
        cdef double term
        cdef double ell
        cdef _LogExpSumSum entry
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
        for i in range(n):
            lt = sp_safe_log(P.parameter_at(i))
            for j in range(m):
                lp = sp_safe_log(Q.parameter_at(j))
                term = lt + lp + V[i * m + j]
                if isfinite(term) and term > max_val:
                    max_val = term
        for i in range(n):
            lt = sp_safe_log(P.parameter_at(i))
            for j in range(m):
                lp = sp_safe_log(Q.parameter_at(j))
                term = lt + lp + V[i * m + j]
                if isfinite(term):
                    total += exp(term - max_val)
        ell = -INFINITY if total <= 0.0 else max_val + log(total)
        if self.recording:
            entry = _LogExpSumSum()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.n = n
            entry.m = m
            entry.V = V
            entry.child_idx = child_idx
            entry.ell = ell
            entry.log_theta.resize(n)
            entry.log_phi.resize(m)
            for i in range(n):
                entry.log_theta[i] = sp_safe_log(P.parameter_at(i))
            for j in range(m):
                entry.log_phi[j] = sp_safe_log(Q.parameter_at(j))
            self.append_tape(entry, P, Q)
        return ell

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
        cdef double ell = 0.0
        cdef size_t t
        cdef _LogExpProdProd entry
        match_prod_children(P, Q, row_ind, col_ind, "expectation")
        if self.recording:
            child_idx.resize(n * m)
            for t in range(n * m):
                child_idx[t] = NO_TAPE_IDX
        for i in range(n):
            q_idx = col_ind[i]
            pc = P.child_at(i)
            qc = Q.child_at(<size_t>q_idx)
            ell += self.couple_value(pc, qc, sP, sQ)
            if self.recording:
                child_idx[i * m + <size_t>q_idx] = self.lookup_pair_tape_idx(pc, qc)
        if self.recording:
            entry = _LogExpProdProd()
            entry.side_P = sP
            entry.side_Q = sQ
            entry.P = P
            entry.Q = Q
            entry.m = m
            entry.row_ind = row_ind
            entry.col_ind = col_ind
            entry.child_idx = child_idx
            self.append_tape(entry, P, Q)
        return ell


# === Flattened nogil tape + pools =============================================
# GIL records coupling topology on flat indices; forward and backward replay
# run nogil over flat PMF/weight pools.

cdef enum ExpEntryKind:
    EK_LEAF = 0
    EK_SUMSUM = 1
    EK_PRODPROD = 2


cdef struct ExpEntry:
    int kind
    size_t pP
    size_t pQ
    size_t n
    size_t m
    double ell
    vector[double] V
    vector[size_t] child_idx
    vector[int] row_ind
    vector[int] col_ind
    vector[double] child_vals


cdef void _exp_backward_core(
    ExpEntry* ee, double* adj, size_t n_entries,
    const size_t* lpoff0, const double* lpmf0, const size_t* coff0, const double* sw0,
    const size_t* lpoff1, const double* lpmf1, const size_t* coff1, const double* sw1,
    double* cat0, double* cat1, double* sum0, double* sum1,
) noexcept nogil:
    cdef ssize_t t
    cdef ssize_t ii
    cdef ExpEntry* e
    cdef double g
    cdef double s
    cdef double factor
    cdef size_t i
    cdef size_t j
    cdef size_t k
    cdef size_t cid
    cdef size_t kidx
    cdef size_t num
    cdef size_t r
    cdef size_t n
    cdef size_t m
    cdef size_t o0p
    cdef size_t o1q
    cdef size_t cb0
    cdef size_t cb1
    cdef vector[double] prefix
    cdef vector[double] suffix
    for t in range(<ssize_t>n_entries - 1, -1, -1):
        g = adj[<size_t>t]
        if g == 0.0:
            continue
        e = &ee[<size_t>t]
        n = e.n
        m = e.m
        if e.kind == EK_LEAF:
            o0p = lpoff0[e.pP]
            o1q = lpoff1[e.pQ]
            for k in range(n):
                cat0[o0p + k] += g * lpmf1[o1q + k]
                cat1[o1q + k] += g * lpmf0[o0p + k]
        elif e.kind == EK_SUMSUM:
            cb0 = coff0[e.pP]
            cb1 = coff1[e.pQ]
            for i in range(n):
                for j in range(m):
                    cid = e.child_idx[i * m + j]
                    if cid != NO_TAPE_IDX:
                        adj[cid] += g * sw0[cb0 + i] * sw1[cb1 + j]
            for i in range(n):
                s = 0.0
                for j in range(m):
                    s += sw1[cb1 + j] * e.V[i * m + j]
                sum0[cb0 + i] += g * s
            for j in range(m):
                s = 0.0
                for i in range(n):
                    s += sw0[cb0 + i] * e.V[i * m + j]
                sum1[cb1 + j] += g * s
        else:  # EK_PRODPROD
            if n == 0:
                continue
            prefix.resize(n)
            suffix.resize(n)
            prefix[0] = 1.0
            for i in range(1, n):
                prefix[i] = prefix[i - 1] * e.child_vals[i - 1]
            suffix[n - 1] = 1.0
            for ii in range(<ssize_t>n - 1, 0, -1):
                suffix[<size_t>ii - 1] = suffix[<size_t>ii] * e.child_vals[<size_t>ii]
            num = e.row_ind.size()
            for kidx in range(num):
                r = <size_t>e.row_ind[kidx]
                factor = g * prefix[r] * suffix[r]
                cid = e.child_idx[r * m + <size_t>e.col_ind[kidx]]
                if cid != NO_TAPE_IDX:
                    adj[cid] += factor


cdef void _logexp_backward_core(
    ExpEntry* ee, double* adj, size_t n_entries,
    const size_t* lpoff0, const double* llog0, const size_t* coff0, const double* slog0,
    const size_t* lpoff1, const double* llog1, const size_t* coff1, const double* slog1,
    double* cat0, double* cat1, double* sum0, double* sum1,
) noexcept nogil:
    cdef ssize_t t
    cdef ExpEntry* e
    cdef double g
    cdef double s
    cdef double w
    cdef double lt
    cdef double lp
    cdef size_t i
    cdef size_t j
    cdef size_t k
    cdef size_t cid
    cdef size_t kidx
    cdef size_t num
    cdef size_t r
    cdef size_t n
    cdef size_t m
    cdef size_t o0p
    cdef size_t o1q
    cdef size_t cb0
    cdef size_t cb1
    cdef double ell
    for t in range(<ssize_t>n_entries - 1, -1, -1):
        g = adj[<size_t>t]
        if g == 0.0:
            continue
        e = &ee[<size_t>t]
        n = e.n
        m = e.m
        ell = e.ell
        if e.kind == EK_LEAF:
            o0p = lpoff0[e.pP]
            o1q = lpoff1[e.pQ]
            for k in range(n):
                cat0[o0p + k] += g * exp(llog1[o1q + k] - ell)
                cat1[o1q + k] += g * exp(llog0[o0p + k] - ell)
        elif e.kind == EK_SUMSUM:
            cb0 = coff0[e.pP]
            cb1 = coff1[e.pQ]
            for i in range(n):
                lt = slog0[cb0 + i]
                for j in range(m):
                    lp = slog1[cb1 + j]
                    w = g * exp(lt + lp + e.V[i * m + j] - ell)
                    cid = e.child_idx[i * m + j]
                    if cid != NO_TAPE_IDX:
                        adj[cid] += w
            for i in range(n):
                s = 0.0
                for j in range(m):
                    s += exp(slog1[cb1 + j] + e.V[i * m + j] - ell)
                sum0[cb0 + i] += g * s
            for j in range(m):
                s = 0.0
                for i in range(n):
                    s += exp(slog0[cb0 + i] + e.V[i * m + j] - ell)
                sum1[cb1 + j] += g * s
        else:  # EK_PRODPROD
            num = e.row_ind.size()
            for kidx in range(num):
                r = <size_t>e.row_ind[kidx]
                cid = e.child_idx[r * m + <size_t>e.col_ind[kidx]]
                if cid != NO_TAPE_IDX:
                    adj[cid] += g


cdef inline uint64_t _exp_flat_pair_key(size_t i0, size_t i1) noexcept:
    return (<uint64_t>i0 << 32) | <uint64_t>i1


cdef void _exp_forward_core(
    ExpEntry* ee,
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
    cdef ExpEntry* e
    cdef size_t n
    cdef size_t m
    cdef size_t i
    cdef size_t j
    cdef size_t k
    cdef size_t cid
    cdef size_t kidx
    cdef size_t num
    cdef size_t r
    cdef size_t o0
    cdef size_t o1
    cdef size_t cb0
    cdef size_t cb1
    cdef double total
    cdef double prod
    cdef vector[double] V
    for t in range(n_entries):
        e = &ee[t]
        n = e.n
        m = e.m
        if e.kind == EK_LEAF:
            o0 = lpoff0[e.pP]
            o1 = lpoff1[e.pQ]
            total = 0.0
            for k in range(n):
                total += lpmf0[o0 + k] * lpmf1[o1 + k]
            vals[t] = total
        elif e.kind == EK_SUMSUM:
            V.resize(n * m)
            for i in range(n):
                for j in range(m):
                    cid = e.child_idx[i * m + j]
                    if cid != NO_TAPE_IDX:
                        V[i * m + j] = vals[cid]
                    else:
                        V[i * m + j] = 0.0
            e.V = V
            cb0 = coff0[e.pP]
            cb1 = coff1[e.pQ]
            total = 0.0
            for i in range(n):
                for j in range(m):
                    total += sw0[cb0 + i] * sw1[cb1 + j] * V[i * m + j]
            vals[t] = total
        else:  # EK_PRODPROD
            prod = 1.0
            num = e.row_ind.size()
            for kidx in range(num):
                r = <size_t>e.row_ind[kidx]
                cid = e.child_idx[r * m + <size_t>e.col_ind[kidx]]
                if cid != NO_TAPE_IDX:
                    prod *= vals[cid]
            vals[t] = prod


cdef void _logexp_forward_core(
    ExpEntry* ee,
    double* vals,
    size_t n_entries,
    const size_t* lpoff0,
    const double* llog0,
    const size_t* lpoff1,
    const double* llog1,
    const size_t* coff0,
    const double* slog0,
    const size_t* coff1,
    const double* slog1,
) noexcept nogil:
    cdef size_t t
    cdef ExpEntry* e
    cdef size_t n
    cdef size_t m
    cdef size_t i
    cdef size_t j
    cdef size_t k
    cdef size_t cid
    cdef size_t kidx
    cdef size_t num
    cdef size_t r
    cdef size_t o0
    cdef size_t o1
    cdef size_t cb0
    cdef size_t cb1
    cdef double max_val
    cdef double total
    cdef double term
    cdef double ell
    cdef vector[double] V
    for t in range(n_entries):
        e = &ee[t]
        n = e.n
        m = e.m
        if e.kind == EK_LEAF:
            o0 = lpoff0[e.pP]
            o1 = lpoff1[e.pQ]
            max_val = -INFINITY
            for k in range(n):
                term = llog0[o0 + k] + llog1[o1 + k]
                if isfinite(term) and term > max_val:
                    max_val = term
            total = 0.0
            for k in range(n):
                term = llog0[o0 + k] + llog1[o1 + k]
                if isfinite(term):
                    total += exp(term - max_val)
            ell = -INFINITY if total <= 0.0 else max_val + log(total)
            e.ell = ell
            vals[t] = ell
        elif e.kind == EK_SUMSUM:
            V.resize(n * m)
            for i in range(n):
                for j in range(m):
                    cid = e.child_idx[i * m + j]
                    if cid != NO_TAPE_IDX:
                        V[i * m + j] = vals[cid]
                    else:
                        V[i * m + j] = -INFINITY
            e.V = V
            cb0 = coff0[e.pP]
            cb1 = coff1[e.pQ]
            max_val = -INFINITY
            for i in range(n):
                for j in range(m):
                    term = slog0[cb0 + i] + slog1[cb1 + j] + V[i * m + j]
                    if isfinite(term) and term > max_val:
                        max_val = term
            total = 0.0
            for i in range(n):
                for j in range(m):
                    term = slog0[cb0 + i] + slog1[cb1 + j] + V[i * m + j]
                    if isfinite(term):
                        total += exp(term - max_val)
            ell = -INFINITY if total <= 0.0 else max_val + log(total)
            e.ell = ell
            vals[t] = ell
        else:  # EK_PRODPROD
            ell = 0.0
            num = e.row_ind.size()
            for kidx in range(num):
                r = <size_t>e.row_ind[kidx]
                cid = e.child_idx[r * m + <size_t>e.col_ind[kidx]]
                if cid != NO_TAPE_IDX:
                    ell += vals[cid]
            vals[t] = ell


cdef class _FlatExpBase(CoupleContext):
    cdef CompiledCircuit g0
    cdef CompiledCircuit g1
    cdef unordered_map[uint64_t, size_t] flat_pair_to_tape
    cdef vector[ExpEntry] etape
    cdef vector[double] tape_vals
    cdef vector[double] cat0
    cdef vector[double] cat1
    cdef vector[double] sum0
    cdef vector[double] sum1
    cdef bint log_space

    cdef void _reset_flat(self) except *:
        self.reset_base()
        self.g0 = None
        self.g1 = None
        self.flat_pair_to_tape.clear()
        self.etape.clear()
        self.tape_vals.clear()
        self.cat0.clear()
        self.cat1.clear()
        self.sum0.clear()
        self.sum1.clear()

    cdef size_t _record_pair(self, size_t i0, size_t i1) except *:
        cdef uint64_t key = _exp_flat_pair_key(i0, i1)
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
        cdef size_t t
        cdef vector[int] row_ind
        cdef vector[int] col_ind
        cdef vector[size_t] child_idx

        if k0 == NODE_INPUT and k1 == NODE_INPUT:
            if g0.leaf_var[i0] != g1.leaf_var[i1]:
                raise ValueError("expectation incompatible: leaf scope variables differ")
            n = <size_t>g0.leaf_card[i0]
            m = <size_t>g1.leaf_card[i1]
            if n != m:
                raise ValueError("expectation incompatible: leaf support sizes differ")
            idx = self.etape.size()
            self.etape.push_back(ExpEntry())
            self.tape_adjoints.push_back(0.0)
            self.flat_pair_to_tape[key] = idx
            self.etape[idx].kind = EK_LEAF
            self.etape[idx].pP = i0
            self.etape[idx].pQ = i1
            self.etape[idx].n = n
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
            self.etape.push_back(ExpEntry())
            self.tape_adjoints.push_back(0.0)
            self.flat_pair_to_tape[key] = idx
            self.etape[idx].kind = EK_SUMSUM
            self.etape[idx].pP = i0
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
            match_prod_children_flat(g0, i0, g1, i1, row_ind, col_ind, "expectation")
            child_idx.resize(n * m)
            for t in range(n * m):
                child_idx[t] = NO_TAPE_IDX
            for i in range(n):
                qidx = <size_t>col_ind[i]
                ci0 = g0.children_flat[start0 + i]
                ci1 = g1.children_flat[start1 + qidx]
                child_idx[i * m + qidx] = self._record_pair(ci0, ci1)
            idx = self.etape.size()
            self.etape.push_back(ExpEntry())
            self.tape_adjoints.push_back(0.0)
            self.flat_pair_to_tape[key] = idx
            self.etape[idx].kind = EK_PRODPROD
            self.etape[idx].pP = i0
            self.etape[idx].pQ = i1
            self.etape[idx].n = n
            self.etape[idx].m = m
            self.etape[idx].row_ind = row_ind
            self.etape[idx].col_ind = col_ind
            self.etape[idx].child_idx = child_idx
            return idx
        else:
            raise ValueError(
                f"expectation incompatible: cannot couple node kinds {k0} with {k1}"
            )

    cdef double _solve_flat(self, CompiledCircuit c0, CompiledCircuit c1,
                            bint log_space) except *:
        self._reset_flat()
        self.g0 = c0
        self.g1 = c1
        self.log_space = log_space
        self.cat0.assign(self.g0.leaf_pmf_flat.size(), 0.0)
        self.cat1.assign(self.g1.leaf_pmf_flat.size(), 0.0)
        self.sum0.assign(self.g0.children_flat.size(), 0.0)
        self.sum1.assign(self.g1.children_flat.size(), 0.0)
        self._record_pair(self.g0.root_index, self.g1.root_index)
        self.tape_vals.assign(self.etape.size(), 0.0)
        if log_space:
            with nogil:
                _logexp_forward_core(
                    self.etape.data(), self.tape_vals.data(), self.etape.size(),
                    self.g0.leaf_pmf_off.data(), self.g0.leaf_logpmf_flat.data(),
                    self.g1.leaf_pmf_off.data(), self.g1.leaf_logpmf_flat.data(),
                    self.g0.child_off.data(), self.g0.sum_logw_flat.data(),
                    self.g1.child_off.data(), self.g1.sum_logw_flat.data(),
                )
        else:
            with nogil:
                _exp_forward_core(
                    self.etape.data(), self.tape_vals.data(), self.etape.size(),
                    self.g0.leaf_pmf_off.data(), self.g0.leaf_pmf_flat.data(),
                    self.g1.leaf_pmf_off.data(), self.g1.leaf_pmf_flat.data(),
                    self.g0.child_off.data(), self.g0.sum_w_flat.data(),
                    self.g1.child_off.data(), self.g1.sum_w_flat.data(),
                )
        return self.tape_vals[self.etape.size() - 1]

    cdef tuple _solve_with_grad_flat(self, CompiledCircuit c0, CompiledCircuit c1,
                                     bint log_space):
        cdef double value = self._solve_flat(c0, c1, log_space)
        cdef size_t root_idx = self.etape.size() - 1
        self.tape_adjoints[root_idx] = 1.0
        if log_space:
            with nogil:
                _logexp_backward_core(
                    self.etape.data(), self.tape_adjoints.data(), self.etape.size(),
                    self.g0.leaf_pmf_off.data(), self.g0.leaf_logpmf_flat.data(),
                    self.g0.child_off.data(), self.g0.sum_logw_flat.data(),
                    self.g1.leaf_pmf_off.data(), self.g1.leaf_logpmf_flat.data(),
                    self.g1.child_off.data(), self.g1.sum_logw_flat.data(),
                    self.cat0.data(), self.cat1.data(),
                    self.sum0.data(), self.sum1.data(),
                )
        else:
            with nogil:
                _exp_backward_core(
                    self.etape.data(), self.tape_adjoints.data(), self.etape.size(),
                    self.g0.leaf_pmf_off.data(), self.g0.leaf_pmf_flat.data(),
                    self.g0.child_off.data(), self.g0.sum_w_flat.data(),
                    self.g1.leaf_pmf_off.data(), self.g1.leaf_pmf_flat.data(),
                    self.g1.child_off.data(), self.g1.sum_w_flat.data(),
                    self.cat0.data(), self.cat1.data(),
                    self.sum0.data(), self.sum1.data(),
                )
        return self._materialize(value)

    cdef tuple _materialize(self, double value):
        cdef GradBundle g1 = GradBundle()
        cdef GradBundle g2 = GradBundle()
        g1.value = value
        g2.value = value
        g1.sum_grads = self._grads_for(self.g0, self.sum0, True)
        g1.cat_grads = self._grads_for(self.g0, self.cat0, False)
        g2.sum_grads = self._grads_for(self.g1, self.sum1, True)
        g2.cat_grads = self._grads_for(self.g1, self.cat1, False)
        return (value, g1, g2)

    cdef dict _grads_for(self, CompiledCircuit g, vector[double]& pool, bint sums):
        cdef dict out = {}
        cdef size_t nn
        cdef size_t start
        cdef size_t stop
        cdef size_t off
        cdef size_t k
        cdef int card
        cdef object arr
        for nn in range(g.n_nodes):
            if sums:
                if g.kinds[nn] != NODE_SUM:
                    continue
                start = g.child_off[nn]
                stop = g.child_off[nn + 1]
                arr = np.empty(stop - start, dtype=np.float64)
                for k in range(start, stop):
                    arr[k - start] = pool[k]
                out[g.node_ids[nn]] = arr
            else:
                if g.kinds[nn] != NODE_INPUT:
                    continue
                off = g.leaf_pmf_off[nn]
                card = g.leaf_card[nn]
                arr = np.empty(card, dtype=np.float64)
                for k in range(<size_t>card):
                    arr[k] = pool[off + k]
                out[g.node_ids[nn]] = arr
        return out


cdef class _FlatExpectationContext(_FlatExpBase):
    cdef double solve_value(self, CompiledCircuit c0, CompiledCircuit c1):
        return self._solve_flat(c0, c1, False)

    cdef tuple solve_with_grad(self, CompiledCircuit c0, CompiledCircuit c1):
        return self._solve_with_grad_flat(c0, c1, False)


cdef class _FlatLogExpectationContext(_FlatExpBase):
    cdef double solve_value(self, CompiledCircuit c0, CompiledCircuit c1):
        return self._solve_flat(c0, c1, True)

    cdef tuple solve_with_grad(self, CompiledCircuit c0, CompiledCircuit c1):
        return self._solve_with_grad_flat(c0, c1, True)


# === public API ===============================================================

cdef tuple _run_pair(CoupleContext ctx, CircuitNode r1, CircuitNode r2, double value):
    cdef size_t root_idx
    cdef GradBundle g1
    cdef GradBundle g2
    root_idx = ctx.lookup_pair_tape_idx(r1, r2)
    if root_idx == NO_TAPE_IDX:
        raise RuntimeError("internal: root pair has no tape entry")
    ctx.tape_adjoints[root_idx] = 1.0
    ctx.run_backward()
    g1 = GradBundle()
    g1.value = value
    g1.sum_grads = ctx.sum_grads0
    g1.cat_grads = ctx.cat_grads0
    g2 = GradBundle()
    g2.value = value
    g2.sum_grads = ctx.sum_grads1
    g2.cat_grads = ctx.cat_grads1
    return (value, g1, g2)


cpdef double exp_query(object circuit1, object circuit2) except *:
    r"""Compute :math:`E_Q[P(X)] = \sum_x P(x)\, Q(x)` for compatible PCs.

    Args:
        circuit1: Circuit defining :math:`P` (first factor).
        circuit2: Circuit defining :math:`Q` (second factor).

    Returns:
        The expectation scalar.

    Raises:
        ValueError: If scopes or decompositions are incompatible.
    """
    cdef CompiledCircuit c1
    cdef CompiledCircuit c2
    cdef CircuitNode r1
    cdef CircuitNode r2
    cdef ExpectationContext ctx
    _check_pair_types(circuit1, circuit2)
    if isinstance(circuit1, CompiledCircuit):
        c1 = circuit1
        c2 = circuit2
        return _FlatExpectationContext().solve_value(c1, c2)
    r1 = _unwrap(circuit1)
    r2 = _unwrap(circuit2)
    ctx = ExpectationContext()
    ctx.reset_base()
    return ctx.couple_value(r1, r2, 0, 1)


cpdef tuple exp_query_and_grad(object circuit1, object circuit2):
    r"""Compute :math:`E_Q[P(X)]` and exact gradients for both circuits.

    Args:
        circuit1: Circuit defining :math:`P`.
        circuit2: Circuit defining :math:`Q`.

    Returns:
        ``(value, grads1, grads2)`` where each ``grads*`` is a
        :class:`~sparc.grad.GradBundle`.
    """
    cdef CompiledCircuit c1
    cdef CompiledCircuit c2
    cdef CircuitNode r1
    cdef CircuitNode r2
    cdef ExpectationContext ctx
    cdef double value
    _check_pair_types(circuit1, circuit2)
    if isinstance(circuit1, CompiledCircuit):
        c1 = circuit1
        c2 = circuit2
        return _FlatExpectationContext().solve_with_grad(c1, c2)
    r1 = _unwrap(circuit1)
    r2 = _unwrap(circuit2)
    ctx = ExpectationContext()
    ctx.reset_base()
    ctx.recording = True
    try:
        value = ctx.couple_value(r1, r2, 0, 1)
        return _run_pair(ctx, r1, r2, value)
    finally:
        ctx.recording = False


cpdef double log_exp_query(object circuit1, object circuit2) except *:
    r"""Compute :math:`\log E_Q[P(X)]` for compatible PCs.

    Args:
        circuit1: Circuit defining :math:`P`.
        circuit2: Circuit defining :math:`Q`.

    Returns:
        The log-expectation scalar.
    """
    cdef CompiledCircuit c1
    cdef CompiledCircuit c2
    cdef CircuitNode r1
    cdef CircuitNode r2
    cdef LogExpectationContext ctx
    _check_pair_types(circuit1, circuit2)
    if isinstance(circuit1, CompiledCircuit):
        c1 = circuit1
        c2 = circuit2
        return _FlatLogExpectationContext().solve_value(c1, c2)
    r1 = _unwrap(circuit1)
    r2 = _unwrap(circuit2)
    ctx = LogExpectationContext()
    ctx.reset_base()
    return ctx.couple_value(r1, r2, 0, 1)


cpdef tuple log_exp_query_and_grad(object circuit1, object circuit2):
    r"""Compute :math:`\log E_Q[P(X)]` and exact gradients for both circuits.

    Args:
        circuit1: Circuit defining :math:`P`.
        circuit2: Circuit defining :math:`Q`.

    Returns:
        ``(value, grads1, grads2)`` where each ``grads*`` is a
        :class:`~sparc.grad.GradBundle`.
    """
    cdef CompiledCircuit c1
    cdef CompiledCircuit c2
    cdef CircuitNode r1
    cdef CircuitNode r2
    cdef LogExpectationContext ctx
    cdef double value
    _check_pair_types(circuit1, circuit2)
    if isinstance(circuit1, CompiledCircuit):
        c1 = circuit1
        c2 = circuit2
        return _FlatLogExpectationContext().solve_with_grad(c1, c2)
    r1 = _unwrap(circuit1)
    r2 = _unwrap(circuit2)
    ctx = LogExpectationContext()
    ctx.reset_base()
    ctx.recording = True
    try:
        value = ctx.couple_value(r1, r2, 0, 1)
        return _run_pair(ctx, r1, r2, value)
    finally:
        ctx.recording = False
