# Compiled Evaluation

Fast inference uses a pre-built **`CompiledCircuit`** ([`sparc._graph.CompiledCircuit`][sparc._graph.CompiledCircuit]).

## Two inference tiers

| Tier | Input | Path |
|------|-------|------|
| Object-graph | `CircuitNode` | Memoized recursion on live nodes (GIL) |
| Flat / nogil | `CompiledCircuit` | CSR layout, precomputed PMFs, `nogil` numeric cores |

Compile once when topology is fixed:

```python
compiled = circuit.compile()
log_lls = compiled.log_likelihood(data, var_to_col=None)
```

After parameter updates (e.g. MLE steps), refresh flat pools without rebuilding topology:

```python
compiled.refresh_parameters()
```

Changing a leaf's cardinality changes flat offsets and therefore requires a
new `circuit.compile()`; `refresh_parameters()` raises instead of refreshing
an incompatible snapshot.

## Requirements

- All leaves must be [`FiniteDiscreteInputNode`][sparc.nodes.FiniteDiscreteInputNode] (any subclass; PMFs materialized via `pmf_at` at compile time).
- Non-discrete custom `InputNode` subclasses remain object-graph-only on the root `CircuitNode`.

## Pairwise queries

CW, GCW, and expectation queries require **both** operands to be `CompiledCircuit` for the flat path. Mixed `CircuitNode` + `CompiledCircuit` raises `TypeError`.

Module-level functions (`cw_distance`, `gcw_crossterm`, …) dispatch on operand type automatically.

## Layout

The flattened representation stores:

- `kinds`: per-node type tag (input / product / sum)
- `child_off`, `children_flat`: CSR child indices
- `sum_w_flat`, `sum_logw_flat`: mixture weights for sum nodes
- `leaf_var`, `leaf_card`, `leaf_pmf_flat`, `leaf_logpmf_flat`: leaf metadata and PMFs
- `node_ids`, `scope_sig`: gradient keys and product-child matching

## Differentiable sampling

`compiled.sample(n, differentiable=True)` reads the flat parameter snapshot
and returns a [`DifferentiableSample`][sparc.sampling.DifferentiableSample].
Its hard assignments and SIMPLE tape are frozen, so a later parameter refresh
does not change a retained VJP. The differentiable path retains Python-visible
tape state and does not use the ordinary one-path `nogil` sampling loop.

See [Differentiable sampling](../guides/differentiable-sampling.md) for the
packed one-hot layout and structural requirements.

## Batched evaluation

2-D input `(n_samples, n_columns)` is evaluated on a **node-major** layout: each
post-order node updates all batch lanes in one contiguous sweep so the C++
compiler can auto-vectorize across the sample axis. The public API is unchanged
(`log_likelihood` / `likelihood` on `CompiledCircuit`); only the internal
`nogil` kernel differs from the per-row scalar loop.

## Migration

| Before | After |
|--------|-------|
| `circuit.batched_log_likelihood(data)` | `circuit.compile().log_likelihood(data)` |
| Dict evidence `{var: value}` | 1D `np.ndarray` (index = variable id); 2D batches for vectorized eval |
| `sample()` returned list of dicts | `sample()` returns `(n, max_var+1)` int32 ndarray |
