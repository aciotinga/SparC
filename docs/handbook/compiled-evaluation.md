# Compiled Evaluation

Batched log-likelihood and several query fast paths flatten the circuit DAG
into a **`CompiledGraph`** ([`sparc._graph`][sparc._graph]) or user-facing
**`CompiledCircuit`** ([`sparc.eval.CompiledCircuit`][sparc.eval.CompiledCircuit]).

## Why flatten?

The object-graph evaluator dispatches through Python/Cython method calls per
node. For datasets with many rows, flattening to CSR-like arrays allows a
single `nogil` pass over all samples.

## CompiledCircuit

```python
compiled = circuit.compile()
log_lls = compiled.log_likelihood(data, var_to_col=None)
```

Requirements:

- Leaves must be [`FiniteDiscreteInputNode`][sparc.nodes.FiniteDiscreteInputNode]
  over a single variable (the leaf family used by OT/expectation queries).
- Unsupported leaf types set `has_fallback=True`; the library falls back to
  the object-graph path automatically.

## Layout

The flattened representation stores:

- `kinds`: per-node type tag (input / product / sum)
- `child_off`, `children_flat`: CSR child indices
- `sum_logw_flat`: log mixture weights for sum nodes
- `leaf_var`, `leaf_card`, `leaf_logpmf_flat`: leaf metadata and log-PMFs

## Single-datapoint path

`likelihood`, `log_likelihood`, and `sample` try the flat path when possible;
otherwise they use memoized recursion on the live node objects.

## Gradients

[`mean_log_likelihood_and_grad`][sparc.grad.mean_log_likelihood_and_grad] uses
the same flat graph when available, accumulating into a [`GradBundle`][sparc.grad.GradBundle]
over the full dataset in one reverse pass.
