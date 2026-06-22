# Extending SparC

SparC is designed so new leaf types, ground metrics, and pairwise queries can
be added without editing existing query code.

## New leaf type

1. Subclass [`InputNode`][sparc.nodes.InputNode] for likelihood and sampling
   only, or [`FiniteDiscreteInputNode`][sparc.nodes.FiniteDiscreteInputNode]
   to also support CW/GCW/expectation queries.
2. Override the `cdef` hooks: `prob_c`, `sample_into_c`, and (for finite
   discrete) `support_size`, `pmf_at`, `scope_var_c`.
3. Register cloning in [`circuit._clone_node`][sparc.circuit] if you use
   [`Circuit.clone`][sparc.circuit.Circuit.clone].
4. Add serializer support in [`CircuitSerializer`][sparc.io.serializer.CircuitSerializer]
   if you need save/load.

No changes are required in `eval.pyx` or query modules for the object-graph path — dispatch goes through
the leaf vtable and `node_kind` tags. For the fast path, subclass
[`FiniteDiscreteInputNode`][sparc.nodes.FiniteDiscreteInputNode] and implement
`pmf_at`; then `circuit.compile()` materializes PMFs into flat pools.

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
