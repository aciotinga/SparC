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

- Discrete circuits: all leaves must be [`FiniteDiscreteInputNode`][sparc.nodes.FiniteDiscreteInputNode]
  (any subclass; PMFs materialized via `pmf_at` at compile time).
- Continuous circuits: univariate [`GaussianInputNode`][sparc.nodes.GaussianInputNode]
  leaves; parameters live in `leaf_param_flat` (`[μ, σ]` per leaf). Evidence is
  `float64` with `NaN` marking missing variables.
- Do not mix discrete and continuous leaves in one DAG.
- Non-Gaussian custom `InputNode` subclasses remain object-graph-only on the root `CircuitNode`.

## Pairwise queries

CW, GCW, and expectation queries require **both** operands to be `CompiledCircuit` for the flat path. Mixed `CircuitNode` + `CompiledCircuit` raises `TypeError`.

Module-level functions (`cw_distance`, `gcw_crossterm`, …) dispatch on operand type automatically.

## Layout

The flattened representation stores:

- `kinds`: per-node type tag (input / product / sum)
- `child_off`, `children_flat`: CSR child indices
- `sum_w_flat`, `sum_logw_flat`: mixture weights for sum nodes
- `leaf_var`, `leaf_card`, `leaf_pmf_flat`, `leaf_logpmf_flat`: discrete leaf metadata and PMFs
- `leaf_param_off`, `leaf_param_flat`, `leaf_n_param`: continuous leaf parameters
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

On Linux/Windows (and macOS wheels built with `libomp`), that row axis may run
under OpenMP: the batch is split once per eval, each thread walking the full
DAG on its row slice. Cap the team at the top of a script:

```python
import sparc

sparc.num_threads = 8          # or sparc.set_num_threads(8)
```

`OMP_NUM_THREADS` still works if you never set this. There is no per-call
`n_threads` argument. Tiny batches (`< 64` rows) and tiny `n_nodes × n_rows`
stay serial so fork overhead cannot dominate. `sparc.num_threads = 1` forces
serial compiled batches.

## Migration

| Before | After |
|--------|-------|
| `circuit.batched_log_likelihood(data)` | `circuit.compile().log_likelihood(data)` |
| Dict evidence `{var: value}` | 1D `np.ndarray` (index = variable id); 2D batches for vectorized eval |
| `sample()` returned list of dicts | `sample()` returns `(n, max_var+1)` int32 (discrete) or float64 (continuous) ndarray |
