# Extending SparC

SparC is designed so new leaf types, ground metrics, and pairwise queries can
be added without editing existing query code.

## New leaf type

1. Subclass [`InputNode`][sparc.nodes.InputNode] for likelihood and sampling
   only, [`FiniteDiscreteInputNode`][sparc.nodes.FiniteDiscreteInputNode]
   for discrete CW/GCW/expectation, or
   [`ContinuousInputNode`][sparc.nodes.ContinuousInputNode] for continuous
   density / W₂ / inner-product / ESD hooks.
2. Override the `cdef` hooks. Discrete: `prob_c`, `sample_into_c(self, rng, int* out)`,
   and (for finite discrete) `support_size`, `pmf_at`, `scope_var_c`. Sampling writes
   `out[scope_var] = value`. Continuous: `n_params` / `params_into` / `set_params_from`,
   `log_density_c` / `density_c`, `sample_into_continuous`, plus the query hooks
   (`cw_w2sq_c`, `inner_product_c`, `esd_c` and backwards). Missing evidence is
   `-1` (discrete) or `NaN` (continuous).
3. A DAG is **all-discrete or all-continuous**; mixing the two leaf domains raises.
4. Register cloning in [`node_clone.clone_node`][sparc.node_clone] if you use
   [`CircuitNode.clone`][sparc.nodes.CircuitNode.clone].
5. Add serializer support in [`CircuitSerializer`][sparc.io.serializer.CircuitSerializer]
   if you need save/load.

No changes are required in `eval.pyx` or query modules for the object-graph path — dispatch goes through
the leaf vtable and `node_kind` tags. For the discrete fast path, subclass
[`FiniteDiscreteInputNode`][sparc.nodes.FiniteDiscreteInputNode] and implement
`pmf_at`; then `circuit.compile()` materializes PMFs into flat pools.
[`GaussianInputNode`][sparc.nodes.GaussianInputNode] compiles into `leaf_param_flat`
(`[μ, σ]`). Other continuous families need a compile-time leaf kind before they
get a nogil path.

Hard-forward differentiable sampling additionally requires finite support and
a consistent cardinality for every occurrence of a variable. Built-in
`CategoricalInputNode` and `BernoulliInputNode` leaves receive SIMPLE
leaf-parameter gradients; custom finite-discrete leaves are sampled as fixed
distributions unless their gradient support is added to the sampling engine.
Continuous circuits use reparameterization `x = μ + σ ε` and write
`GradBundle.cont_grads`; they do not pack into `DifferentiableSample.one_hot`.

## New ground metric

Subclass [`GroundMetric`][sparc.metrics.GroundMetric] and implement
`pairwise` (same support) and `cross` (two supports) cost matrix fill methods.
Pass your metric to CW/GCW/ESD query functions via the `metric` argument.

## New pairwise query

1. Subclass `CoupleContext` (in `sparc/queries/_engine.pyx`) in a new Cython
   module under `sparc/queries/`.
2. Implement `couple_value` for the forward recursion.
3. Define `TapeEntry` subclasses with
   `backward` methods for reverse-mode gradients.
4. Export `cpdef` wrapper functions and add them to `sparc/queries/__init__.py`.

See the [query engine handbook](../handbook/query-engine.md) for the shared
tape, memo, and product-child matching machinery.

## New structure

Add a constructor under `sparc/structures/` using the block algebra in
`_blocks.py` and pluggable
[`InputDistribution`][sparc.structures.distributions] specs.

## Example scripts

The [examples overview](../examples/overview.md) includes optimization loops
that combine queries with [`apply_grads`][sparc.optim.apply_grads].
