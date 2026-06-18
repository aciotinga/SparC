# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Tractable inner-product queries between two compatible PCs.

``exp_query``     : E_Q[P(X)] = sum_x P(x) Q(x)
``log_exp_query`` : log(E_Q[P(X)]) with log-sum-exp stabilization

Both return exact reverse-mode gradients for *both* circuits. Built on the
shared :class:`~sparc.queries._engine.CoupleContext`.
"""

from libcpp.vector cimport vector
from libc.math cimport exp, INFINITY, log

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
    cdef ExpectationContext ctx = ExpectationContext()
    cdef double value
    ctx.reset_base()
    ctx.recording = True
    try:
        value = ctx.couple_value(r1, r2, 0, 1)
        return _run_pair(ctx, r1, r2, value)
    finally:
        ctx.recording = False


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
    cdef LogExpectationContext ctx = LogExpectationContext()
    cdef double value
    ctx.reset_base()
    ctx.recording = True
    try:
        value = ctx.couple_value(r1, r2, 0, 1)
        return _run_pair(ctx, r1, r2, value)
    finally:
        ctx.recording = False
