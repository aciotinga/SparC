# Serialization Format (`gcw-circuit-v1`)

Circuits are stored as UTF-8 JSON with top-level fields:

```json
{
  "format": "gcw-circuit-v1",
  "backend": "numpy",
  "root": 3,
  "nodes": [ ... ]
}
```

## Node records

Nodes appear in post-order (children before parents). Each record has:

| Field | Description |
|-------|-------------|
| `id` | Integer node id (contiguous `0 .. N-1`) |
| `kind` | Node type (see below) |
| `children` | List of child node indices |
| type-specific | Parameters, scope, etc. |

### Internal nodes

**sum**

```json
{"id": 3, "kind": "sum", "children": [1, 2], "params": [0.6, 0.4]}
```

**product**

```json
{"id": 2, "kind": "product", "children": [0, 1]}
```

### Leaf nodes

**categorical**

```json
{"id": 0, "kind": "categorical", "children": [], "scope": [0], "params": [0.7, 0.3]}
```

**bernoulli**

```json
{"id": 0, "kind": "bernoulli", "children": [], "scope": [0], "p": 0.3}
```

**literal**

```json
{"id": 0, "kind": "literal", "children": [], "scope": [0], "value": 1}
```

**indicator**

```json
{"id": 0, "kind": "indicator", "children": [], "scope": [0], "value": 2, "num_cats": 4}
```

**discrete_logistic**

```json
{"id": 0, "kind": "discrete_logistic", "children": [], "scope": [0], "mu": 0.0, "s": 1.0, "num_cats": 8}
```

## DAG sharing

If two parents reference the same child object in memory, the serializer emits
one node record and multiple parent indices pointing to it. On load, each index
reconstructs a distinct Python object (no automatic re-sharing).

## Validation

- Node ids must be contiguous from zero.
- Children indices must be strictly less than the parent index (post-order).
- Categorical PMFs must be single-variable and sum to 1.
- Unknown kinds or unsupported legacy types (e.g. `gaussian`) raise
  `ValueError`.

Implementation: [`CircuitSerializer`][sparc.io.serializer.CircuitSerializer].
