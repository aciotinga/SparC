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
from libc.math cimport exp, INFINITY, log

import numpy as np

from sparc._graph cimport CompiledGraph
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
    if isinstance(circuit, Circuit):
        return <CircuitNode>(<object>circuit).root
    if isinstance(circuit, CircuitNode):
        return <CircuitNode>circuit
    raise TypeError("expected a Circuit or CircuitNode")


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
            if term > max_val:
                max_val = term
        for k in range(n):
            term = sp_safe_log(P.pmf_at(k)) + sp_safe_log(Q.pmf_at(k))
            if term > -INFINITY:
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
                if term > max_val:
                    max_val = term
        for i in range(n):
            lt = sp_safe_log(P.parameter_at(i))
            for j in range(m):
                lp = sp_safe_log(Q.parameter_at(j))
                term = lt + lp + V[i * m + j]
                if term > -INFINITY:
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
# The forward recursion is kept verbatim (it raises all compatibility errors and
# drives the solvers); only the gradient tape becomes a flat tagged C struct and
# the backward replay runs nogil over flat per-side gradient pools, materialized
# into the documented per-node GradBundle dicts once at the end.

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


cdef class _FlatExpBase(CoupleContext):
    cdef CompiledGraph g0
    cdef CompiledGraph g1
    cdef unordered_map[size_t, size_t] pos0
    cdef unordered_map[size_t, size_t] pos1
    cdef vector[ExpEntry] etape
    cdef vector[double] cat0
    cdef vector[double] cat1
    cdef vector[double] sum0
    cdef vector[double] sum1

    cdef void _setup(self, CircuitNode r1, CircuitNode r2) except *:
        self.reset_base()
        self.recording = True
        self.g0 = CompiledGraph()
        self.g0.build(r1)
        self.g1 = CompiledGraph()
        self.g1.build(r2)
        cdef size_t pos
        self.pos0.clear()
        self.pos1.clear()
        for pos in range(self.g0.n_nodes):
            self.pos0[self.g0.node_ids[pos]] = pos
        for pos in range(self.g1.n_nodes):
            self.pos1[self.g1.node_ids[pos]] = pos
        self.cat0.assign(self.g0.leaf_pmf_flat.size(), 0.0)
        self.cat1.assign(self.g1.leaf_pmf_flat.size(), 0.0)
        self.sum0.assign(self.g0.children_flat.size(), 0.0)
        self.sum1.assign(self.g1.children_flat.size(), 0.0)
        self.etape.clear()

    cdef size_t _eappend(self, CircuitNode P, CircuitNode Q) except *:
        cdef size_t idx = self.etape.size()
        self.etape.push_back(ExpEntry())
        self.tape_adjoints.push_back(0.0)
        self.pair_to_tape[pair_key(P, Q)] = idx
        return idx

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

    cdef dict _grads_for(self, CompiledGraph g, vector[double]& pool, bint sums):
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
        cdef size_t idx
        for k in range(n):
            total += P.pmf_at(k) * Q.pmf_at(k)
        idx = self._eappend(P, Q)
        self.etape[idx].kind = EK_LEAF
        self.etape[idx].pP = self.pos0[P.id]
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
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
        cdef size_t idx
        V.resize(n * m)
        child_idx.resize(n * m)
        for i in range(n):
            pc = P.child_at(i)
            for j in range(m):
                qc = Q.child_at(j)
                V[i * m + j] = self.couple_value(pc, qc, sP, sQ)
                child_idx[i * m + j] = self.lookup_pair_tape_idx(pc, qc)
        for i in range(n):
            for j in range(m):
                total += P.parameter_at(i) * Q.parameter_at(j) * V[i * m + j]
        idx = self._eappend(P, Q)
        self.etape[idx].kind = EK_SUMSUM
        self.etape[idx].pP = self.pos0[P.id]
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].m = m
        self.etape[idx].V = V
        self.etape[idx].child_idx = child_idx
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
        cdef size_t idx
        match_prod_children(P, Q, row_ind, col_ind, "expectation")
        child_vals.resize(n)
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
            child_idx[i * m + <size_t>q_idx] = self.lookup_pair_tape_idx(pc, qc)
        idx = self._eappend(P, Q)
        self.etape[idx].kind = EK_PRODPROD
        self.etape[idx].pP = self.pos0[P.id]
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].m = m
        self.etape[idx].child_vals = child_vals
        self.etape[idx].row_ind = row_ind
        self.etape[idx].col_ind = col_ind
        self.etape[idx].child_idx = child_idx
        return total

    cdef tuple solve_with_grad(self, CircuitNode r1, CircuitNode r2):
        self._setup(r1, r2)
        cdef double value = self.couple_value(r1, r2, 0, 1)
        cdef size_t root_idx = self.lookup_pair_tape_idx(r1, r2)
        if root_idx == NO_TAPE_IDX:
            raise RuntimeError("internal: root pair has no tape entry")
        self.tape_adjoints[root_idx] = 1.0
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


cdef class _FlatLogExpectationContext(_FlatExpBase):
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
        cdef size_t idx
        for k in range(n):
            term = sp_safe_log(P.pmf_at(k)) + sp_safe_log(Q.pmf_at(k))
            if term > max_val:
                max_val = term
        for k in range(n):
            term = sp_safe_log(P.pmf_at(k)) + sp_safe_log(Q.pmf_at(k))
            if term > -INFINITY:
                total += exp(term - max_val)
        ell = -INFINITY if total <= 0.0 else max_val + log(total)
        idx = self._eappend(P, Q)
        self.etape[idx].kind = EK_LEAF
        self.etape[idx].pP = self.pos0[P.id]
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].ell = ell
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
        cdef size_t idx
        V.resize(n * m)
        child_idx.resize(n * m)
        for i in range(n):
            pc = P.child_at(i)
            for j in range(m):
                qc = Q.child_at(j)
                V[i * m + j] = self.couple_value(pc, qc, sP, sQ)
                child_idx[i * m + j] = self.lookup_pair_tape_idx(pc, qc)
        for i in range(n):
            lt = sp_safe_log(P.parameter_at(i))
            for j in range(m):
                lp = sp_safe_log(Q.parameter_at(j))
                term = lt + lp + V[i * m + j]
                if term > max_val:
                    max_val = term
        for i in range(n):
            lt = sp_safe_log(P.parameter_at(i))
            for j in range(m):
                lp = sp_safe_log(Q.parameter_at(j))
                term = lt + lp + V[i * m + j]
                if term > -INFINITY:
                    total += exp(term - max_val)
        ell = -INFINITY if total <= 0.0 else max_val + log(total)
        idx = self._eappend(P, Q)
        self.etape[idx].kind = EK_SUMSUM
        self.etape[idx].pP = self.pos0[P.id]
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].m = m
        self.etape[idx].V = V
        self.etape[idx].child_idx = child_idx
        self.etape[idx].ell = ell
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
        cdef size_t idx
        match_prod_children(P, Q, row_ind, col_ind, "expectation")
        child_idx.resize(n * m)
        for t in range(n * m):
            child_idx[t] = NO_TAPE_IDX
        for i in range(n):
            q_idx = col_ind[i]
            pc = P.child_at(i)
            qc = Q.child_at(<size_t>q_idx)
            ell += self.couple_value(pc, qc, sP, sQ)
            child_idx[i * m + <size_t>q_idx] = self.lookup_pair_tape_idx(pc, qc)
        idx = self._eappend(P, Q)
        self.etape[idx].kind = EK_PRODPROD
        self.etape[idx].pP = self.pos0[P.id]
        self.etape[idx].pQ = self.pos1[Q.id]
        self.etape[idx].n = n
        self.etape[idx].m = m
        self.etape[idx].row_ind = row_ind
        self.etape[idx].col_ind = col_ind
        self.etape[idx].child_idx = child_idx
        return ell

    cdef tuple solve_with_grad(self, CircuitNode r1, CircuitNode r2):
        self._setup(r1, r2)
        cdef double value = self.couple_value(r1, r2, 0, 1)
        cdef size_t root_idx = self.lookup_pair_tape_idx(r1, r2)
        if root_idx == NO_TAPE_IDX:
            raise RuntimeError("internal: root pair has no tape entry")
        self.tape_adjoints[root_idx] = 1.0
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
        return self._materialize(value)


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
    """Compute E_Q[P(X)] = sum_x P(x) Q(x) for compatible PCs."""
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef ExpectationContext ctx = ExpectationContext()
    ctx.reset_base()
    return ctx.couple_value(r1, r2, 0, 1)


cpdef tuple exp_query_and_grad(object circuit1, object circuit2):
    """Compute E_Q[P(X)] and exact gradients ``(value, grads1, grads2)``."""
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef CompiledGraph g0 = CompiledGraph()
    cdef CompiledGraph g1 = CompiledGraph()
    g0.build(r1)
    g1.build(r2)
    cdef ExpectationContext ctx
    cdef double value
    if g0.has_fallback or g1.has_fallback:
        ctx = ExpectationContext()
        ctx.reset_base()
        ctx.recording = True
        try:
            value = ctx.couple_value(r1, r2, 0, 1)
            return _run_pair(ctx, r1, r2, value)
        finally:
            ctx.recording = False
    return _FlatExpectationContext().solve_with_grad(r1, r2)


cpdef double log_exp_query(object circuit1, object circuit2) except *:
    """Compute log(E_Q[P(X)]) for compatible PCs."""
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef LogExpectationContext ctx = LogExpectationContext()
    ctx.reset_base()
    return ctx.couple_value(r1, r2, 0, 1)


cpdef tuple log_exp_query_and_grad(object circuit1, object circuit2):
    """Compute log(E_Q[P(X)]) and exact gradients ``(value, grads1, grads2)``."""
    cdef CircuitNode r1 = _unwrap(circuit1)
    cdef CircuitNode r2 = _unwrap(circuit2)
    cdef CompiledGraph g0 = CompiledGraph()
    cdef CompiledGraph g1 = CompiledGraph()
    g0.build(r1)
    g1.build(r2)
    cdef LogExpectationContext ctx
    cdef double value
    if g0.has_fallback or g1.has_fallback:
        ctx = LogExpectationContext()
        ctx.reset_base()
        ctx.recording = True
        try:
            value = ctx.couple_value(r1, r2, 0, 1)
            return _run_pair(ctx, r1, r2, value)
        finally:
            ctx.recording = False
    return _FlatLogExpectationContext().solve_with_grad(r1, r2)
