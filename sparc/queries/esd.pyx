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
    from sparc.circuit import Circuit
    if isinstance(circuit, Circuit):
        return <CircuitNode>(<object>circuit).root
    if isinstance(circuit, CircuitNode):
        return <CircuitNode>circuit
    raise TypeError("expected a Circuit or CircuitNode")


cpdef double expected_squared_distance(
    object circuit,
    double metric_p=1.0,
    double scale_factor=1.0,
    object metric=None,
) except *:
    """Compute E[d(X, X')^2] for two independent draws from ``circuit``."""
    cdef CircuitNode root = _unwrap(circuit)
    cdef _ESDContext ctx = _ESDContext()
    ctx.metric = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    return ctx.solve(root)


cpdef tuple expected_squared_distance_and_grad(
    object circuit,
    double metric_p=1.0,
    double scale_factor=1.0,
    object metric=None,
):
    """Compute E[d(X, X')^2] and its gradient w.r.t. all circuit parameters."""
    cdef CircuitNode root = _unwrap(circuit)
    cdef _ESDContext ctx = _ESDContext()
    ctx.metric = metric if metric is not None else PNormMetric(metric_p, scale_factor)
    return ctx.solve_with_grad(root)
