# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Expected squared distance E[d(X, X')^2] for a single circuit.

Forward propagates ``(mu, nu) = (E[d], E[d^2])`` under two independent draws;
backward is reverse-mode AD on the smooth ``(mu, nu)`` recursion. Single-circuit
query, so it uses :class:`~sparc.grad.GradBundle` directly rather than the
pairwise engine.
"""

from cython.operator cimport dereference as deref
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

from sparc._graph cimport CompiledCircuit
from sparc.grad cimport GradBundle, grad_arr
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


cdef class _ESDContext:
    cdef GroundMetric metric
    cdef unordered_map[size_t, vector[double]] dist_cache
    cdef unordered_map[size_t, double] mu_cache
    cdef unordered_map[size_t, double] nu_cache
    cdef list order
    cdef unordered_map[size_t, double] bar_mu
    cdef unordered_map[size_t, double] bar_nu
    cdef dict sum_grads
    cdef dict cat_grads

    def __cinit__(self):
        self.order = []
        self.sum_grads = {}
        self.cat_grads = {}

    cdef vector[double]* _dist(self, FiniteDiscreteInputNode node, size_t n) except *:
        cdef size_t key = node.id
        if self.dist_cache.find(key) == self.dist_cache.end():
            self.metric.pairwise(node.scope_var_c(), n, self.dist_cache[key])
        return &self.dist_cache[key]

    cdef tuple _forward(self, CircuitNode node) except *:
        cdef size_t nid = node.id
        if self.mu_cache.find(nid) != self.mu_cache.end():
            return (self.mu_cache[nid], self.nu_cache[nid])

        cdef double mu = 0.0
        cdef double nu = 0.0
        cdef double total_mu = 0.0
        cdef double total_nu = 0.0
        cdef double sum_child_mu_sq = 0.0
        cdef size_t i
        cdef size_t nc
        cdef size_t n_out
        cdef size_t a
        cdef size_t b
        cdef double pa
        cdef double d_ab
        cdef CircuitNode child
        cdef vector[double]* d_ptr
        cdef FiniteDiscreteInputNode leaf
        cdef SumNode s
        cdef ProductNode prod
        cdef double child_mu
        cdef double child_nu

        if node.node_kind == NODE_INPUT:
            leaf = <FiniteDiscreteInputNode>node
            n_out = leaf.support_size()
            d_ptr = self._dist(leaf, n_out)
            for a in range(n_out):
                pa = leaf.pmf_at(a)
                for b in range(n_out):
                    d_ab = deref(d_ptr)[a * n_out + b]
                    total_mu += pa * d_ab * leaf.pmf_at(b)
                    total_nu += pa * d_ab * d_ab * leaf.pmf_at(b)
            mu = total_mu
            nu = total_nu
        elif node.node_kind == NODE_SUM:
            s = <SumNode>node
            nc = s.num_children()
            for i in range(nc):
                child = s.child_at(i)
                child_mu, child_nu = self._forward(child)
                total_mu += s.parameter_at(i) * child_mu
                total_nu += s.parameter_at(i) * child_nu
            mu = total_mu
            nu = total_nu
        elif node.node_kind == NODE_PRODUCT:
            prod = <ProductNode>node
            nc = prod.num_children()
            for i in range(nc):
                child = prod.child_at(i)
                child_mu, child_nu = self._forward(child)
                total_mu += child_mu
                total_nu += child_nu
                sum_child_mu_sq += child_mu * child_mu
            mu = total_mu
            nu = total_nu + mu * mu - sum_child_mu_sq
        else:
            raise TypeError(f"unsupported node type: {type(node).__name__}")

        self.mu_cache[nid] = mu
        self.nu_cache[nid] = nu
        self.order.append(node)
        return (mu, nu)

    cdef void _backward(self) except *:
        cdef ssize_t k
        cdef CircuitNode node
        cdef size_t nid
        cdef double adj_mu
        cdef double adj_nu
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
        cdef double d_ab
        cdef double child_mu
        cdef double child_nu
        cdef double theta_i
        cdef double mu_node
        cdef FiniteDiscreteInputNode leaf
        cdef SumNode s
        cdef ProductNode prod

        for k in range(<ssize_t>len(self.order) - 1, -1, -1):
            node = <CircuitNode>self.order[k]
            nid = node.id
            adj_mu = self.bar_mu[nid] if self.bar_mu.find(nid) != self.bar_mu.end() else 0.0
            adj_nu = self.bar_nu[nid] if self.bar_nu.find(nid) != self.bar_nu.end() else 0.0
            if adj_mu == 0.0 and adj_nu == 0.0:
                continue

            if node.node_kind == NODE_INPUT:
                leaf = <FiniteDiscreteInputNode>node
                n_out = leaf.support_size()
                d_ptr = self._dist(leaf, n_out)
                arr = grad_arr(self.cat_grads, nid, n_out)
                for kk in range(n_out):
                    accum = 0.0
                    for j in range(n_out):
                        d_ab = deref(d_ptr)[kk * n_out + j]
                        accum += d_ab * leaf.pmf_at(j)
                    arr[kk] += adj_mu * 2.0 * accum
                    accum = 0.0
                    for j in range(n_out):
                        d_ab = deref(d_ptr)[kk * n_out + j]
                        accum += d_ab * d_ab * leaf.pmf_at(j)
                    arr[kk] += adj_nu * 2.0 * accum
            elif node.node_kind == NODE_SUM:
                s = <SumNode>node
                nc = s.num_children()
                arr = grad_arr(self.sum_grads, nid, nc)
                for i in range(nc):
                    child = s.child_at(i)
                    cid = child.id
                    child_mu = self.mu_cache[cid]
                    child_nu = self.nu_cache[cid]
                    theta_i = s.parameter_at(i)
                    arr[i] += adj_nu * child_nu + adj_mu * child_mu
                    self.bar_mu[cid] = self.bar_mu[cid] + adj_mu * theta_i
                    self.bar_nu[cid] = self.bar_nu[cid] + adj_nu * theta_i
            elif node.node_kind == NODE_PRODUCT:
                prod = <ProductNode>node
                nc = prod.num_children()
                mu_node = self.mu_cache[nid]
                for i in range(nc):
                    child = prod.child_at(i)
                    cid = child.id
                    child_mu = self.mu_cache[cid]
                    self.bar_mu[cid] = (
                        self.bar_mu[cid] + adj_mu + adj_nu * (2.0 * mu_node - 2.0 * child_mu)
                    )
                    self.bar_nu[cid] = self.bar_nu[cid] + adj_nu

    cdef double solve(self, CircuitNode root) except *:
        cdef double nu
        _, nu = self._forward(root)
        return nu

    cdef tuple solve_with_grad(self, CircuitNode root):
        cdef double value
        cdef GradBundle grads
        _, value = self._forward(root)
        self.bar_nu[root.id] = 1.0
        self._backward()
        grads = GradBundle()
        grads.value = value
        grads.sum_grads = self.sum_grads
        grads.cat_grads = self.cat_grads
        return (value, grads)


cdef CircuitNode _unwrap(object circuit):
    cdef CircuitNode node
    if isinstance(circuit, CompiledCircuit):
        raise TypeError(
            "expected a CircuitNode for object-graph queries; "
            "pass a CompiledCircuit directly for the fast path"
        )
    if not isinstance(circuit, CircuitNode):
        raise TypeError("expected a CircuitNode")
    node = <CircuitNode>circuit
    if node.scope.size() == 0:
        node.propagate_scope()
    return node


# --- Flattened nogil fast path ------------------------------------------------

cdef void _build_dist_pool(
    CompiledCircuit g, GroundMetric metric,
    vector[size_t]& dist_off, vector[double]& dist_flat,
) except *:
    """Precompute each leaf's ground-distance matrix into a flat pool (GIL)."""
    cdef size_t n
    cdef size_t k
    cdef int card
    cdef size_t base
    cdef vector[double] tmp
    dist_off.assign(g.n_nodes + 1, 0)
    for n in range(g.n_nodes):
        if g.kinds[n] == NODE_INPUT:
            card = g.leaf_card[n]
            dist_off[n + 1] = <size_t>card * <size_t>card
        else:
            dist_off[n + 1] = 0
    for n in range(g.n_nodes):
        dist_off[n + 1] += dist_off[n]
    dist_flat.assign(dist_off[g.n_nodes], 0.0)
    for n in range(g.n_nodes):
        if g.kinds[n] == NODE_INPUT:
            card = g.leaf_card[n]
            metric.pairwise(g.leaf_var[n], <size_t>card, tmp)
            base = dist_off[n]
            for k in range(<size_t>card * <size_t>card):
                dist_flat[base + k] = tmp[k]


cdef void _flat_esd_forward(
    CompiledCircuit g, const size_t* dist_off, const double* dist_flat,
    double* mu, double* nu,
) noexcept nogil:
    cdef size_t n
    cdef size_t k
    cdef size_t a
    cdef size_t b
    cdef size_t start
    cdef size_t stop
    cdef size_t poff
    cdef size_t doff
    cdef size_t child
    cdef int kind
    cdef int card
    cdef double tmu
    cdef double tnu
    cdef double sumsq
    cdef double pa
    cdef double pb
    cdef double dab
    cdef double cmu
    cdef double cnu
    for n in range(g.n_nodes):
        kind = g.kinds[n]
        if kind == NODE_INPUT:
            card = g.leaf_card[n]
            poff = g.leaf_pmf_off[n]
            doff = dist_off[n]
            tmu = 0.0
            tnu = 0.0
            for a in range(<size_t>card):
                pa = g.leaf_pmf_flat[poff + a]
                for b in range(<size_t>card):
                    dab = dist_flat[doff + a * <size_t>card + b]
                    pb = g.leaf_pmf_flat[poff + b]
                    tmu += pa * dab * pb
                    tnu += pa * dab * dab * pb
            mu[n] = tmu
            nu[n] = tnu
        elif kind == NODE_PRODUCT:
            start = g.child_off[n]
            stop = g.child_off[n + 1]
            tmu = 0.0
            tnu = 0.0
            sumsq = 0.0
            for k in range(start, stop):
                child = g.children_flat[k]
                cmu = mu[child]
                cnu = nu[child]
                tmu += cmu
                tnu += cnu
                sumsq += cmu * cmu
            mu[n] = tmu
            nu[n] = tnu + tmu * tmu - sumsq
        else:  # NODE_SUM
            start = g.child_off[n]
            stop = g.child_off[n + 1]
            tmu = 0.0
            tnu = 0.0
            for k in range(start, stop):
                child = g.children_flat[k]
                tmu += g.sum_w_flat[k] * mu[child]
                tnu += g.sum_w_flat[k] * nu[child]
            mu[n] = tmu
            nu[n] = tnu


cdef void _flat_esd_backward(
    CompiledCircuit g, const size_t* dist_off, const double* dist_flat,
    const double* mu, const double* nu,
    double* bar_mu, double* bar_nu,
    double* cat_pool, double* sum_pool,
) noexcept nogil:
    cdef ssize_t ni
    cdef size_t n
    cdef size_t k
    cdef size_t kk
    cdef size_t j
    cdef size_t start
    cdef size_t stop
    cdef size_t poff
    cdef size_t doff
    cdef size_t coff
    cdef size_t child
    cdef int kind
    cdef int card
    cdef double amu
    cdef double anu
    cdef double acc
    cdef double acc2
    cdef double dab
    cdef double cmu
    cdef double cnu
    cdef double theta
    cdef double mu_node
    for n in range(g.n_nodes):
        bar_mu[n] = 0.0
        bar_nu[n] = 0.0
    bar_nu[g.root_index] = 1.0
    for ni in range(<ssize_t>g.n_nodes - 1, -1, -1):
        n = <size_t>ni
        amu = bar_mu[n]
        anu = bar_nu[n]
        if amu == 0.0 and anu == 0.0:
            continue
        kind = g.kinds[n]
        if kind == NODE_INPUT:
            card = g.leaf_card[n]
            poff = g.leaf_pmf_off[n]
            doff = dist_off[n]
            coff = g.leaf_pmf_off[n]
            for kk in range(<size_t>card):
                acc = 0.0
                acc2 = 0.0
                for j in range(<size_t>card):
                    dab = dist_flat[doff + kk * <size_t>card + j]
                    acc += dab * g.leaf_pmf_flat[poff + j]
                    acc2 += dab * dab * g.leaf_pmf_flat[poff + j]
                cat_pool[coff + kk] += amu * 2.0 * acc + anu * 2.0 * acc2
        elif kind == NODE_SUM:
            start = g.child_off[n]
            stop = g.child_off[n + 1]
            for k in range(start, stop):
                child = g.children_flat[k]
                cmu = mu[child]
                cnu = nu[child]
                theta = g.sum_w_flat[k]
                sum_pool[k] += anu * cnu + amu * cmu
                bar_mu[child] += amu * theta
                bar_nu[child] += anu * theta
        else:  # NODE_PRODUCT
            mu_node = mu[n]
            start = g.child_off[n]
            stop = g.child_off[n + 1]
            for k in range(start, stop):
                child = g.children_flat[k]
                cmu = mu[child]
                bar_mu[child] += amu + anu * (2.0 * mu_node - 2.0 * cmu)
                bar_nu[child] += anu


cdef double _flat_esd_solve(CompiledCircuit g, GroundMetric metric) except *:
    cdef vector[size_t] dist_off
    cdef vector[double] dist_flat
    _build_dist_pool(g, metric, dist_off, dist_flat)
    cdef vector[double] mu
    cdef vector[double] nu
    mu.assign(g.n_nodes, 0.0)
    nu.assign(g.n_nodes, 0.0)
    with nogil:
        _flat_esd_forward(g, dist_off.data(), dist_flat.data(), mu.data(), nu.data())
    return nu[g.root_index]


cdef tuple _flat_esd_solve_with_grad(CompiledCircuit g, GroundMetric metric):
    cdef vector[size_t] dist_off
    cdef vector[double] dist_flat
    _build_dist_pool(g, metric, dist_off, dist_flat)
    cdef vector[double] mu
    cdef vector[double] nu
    cdef vector[double] bar_mu
    cdef vector[double] bar_nu
    cdef vector[double] cat_pool
    cdef vector[double] sum_pool
    mu.assign(g.n_nodes, 0.0)
    nu.assign(g.n_nodes, 0.0)
    bar_mu.assign(g.n_nodes, 0.0)
    bar_nu.assign(g.n_nodes, 0.0)
    cat_pool.assign(g.leaf_pmf_flat.size(), 0.0)
    sum_pool.assign(g.children_flat.size(), 0.0)
    with nogil:
        _flat_esd_forward(g, dist_off.data(), dist_flat.data(), mu.data(), nu.data())
        _flat_esd_backward(
            g, dist_off.data(), dist_flat.data(), mu.data(), nu.data(),
            bar_mu.data(), bar_nu.data(), cat_pool.data(), sum_pool.data(),
        )
    cdef double value = nu[g.root_index]
    cdef GradBundle grads = GradBundle()
    grads.value = value
    cdef dict sum_grads = {}
    cdef dict cat_grads = {}
    cdef size_t nn
    cdef size_t start
    cdef size_t stop
    cdef size_t off
    cdef size_t k
    cdef int card
    cdef object arr
    for nn in range(g.n_nodes):
        if g.kinds[nn] == NODE_SUM:
            start = g.child_off[nn]
            stop = g.child_off[nn + 1]
            arr = np.empty(stop - start, dtype=np.float64)
            for k in range(start, stop):
                arr[k - start] = sum_pool[k]
            sum_grads[g.node_ids[nn]] = arr
        elif g.kinds[nn] == NODE_INPUT:
            off = g.leaf_pmf_off[nn]
            card = g.leaf_card[nn]
            arr = np.empty(card, dtype=np.float64)
            for k in range(<size_t>card):
                arr[k] = cat_pool[off + k]
            cat_grads[g.node_ids[nn]] = arr
    grads.sum_grads = sum_grads
    grads.cat_grads = cat_grads
    return (value, grads)


cpdef double expected_squared_distance(
    object circuit,
    double metric_p=1.0,
    double scale_factor=1.0,
    object metric=None,
) except *:
    r"""Compute :math:`E[d(X, X')^p]` for two independent draws from ``circuit``.

    Args:
        circuit: Source circuit (:class:`~sparc.circuit.Circuit` or root node).
        metric_p: Exponent for the default :class:`PNormMetric`.
        scale_factor: Scale for the default metric.
        metric: Optional :class:`GroundMetric` instance.

    Returns:
        Expected squared-distance objective under the ground metric.
    """
    cdef GroundMetric m
    cdef CircuitNode root
    cdef _ESDContext ctx
    m = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    if isinstance(circuit, CompiledCircuit):
        return _flat_esd_solve(circuit, m)
    root = _unwrap(circuit)
    ctx = _ESDContext()
    ctx.metric = m
    return ctx.solve(root)


cpdef tuple expected_squared_distance_and_grad(
    object circuit,
    double metric_p=1.0,
    double scale_factor=1.0,
    object metric=None,
):
    r"""Compute ESD and its gradient w.r.t. all circuit parameters.

    Args:
        circuit: Source circuit (:class:`~sparc.circuit.Circuit` or root node).
        metric_p: Exponent for the default :class:`PNormMetric`.
        scale_factor: Scale for the default metric.
        metric: Optional :class:`GroundMetric` instance.

    Returns:
        ``(value, grads)`` where ``grads`` is a :class:`~sparc.grad.GradBundle`.
    """
    cdef GroundMetric m
    cdef CircuitNode root
    cdef _ESDContext ctx
    m = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    if isinstance(circuit, CompiledCircuit):
        return _flat_esd_solve_with_grad(circuit, m)
    root = _unwrap(circuit)
    ctx = _ESDContext()
    ctx.metric = m
    return ctx.solve_with_grad(root)
