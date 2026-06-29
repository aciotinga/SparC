# Serialization

Circuits are saved as UTF-8 JSON in the **`gcw-circuit-v1`** format.

## Save and load

```python
from sparc import CircuitNode

root.save("model.json")
loaded = CircuitNode.load("model.json")
```

Lower-level access via [`CircuitSerializer`][sparc.io.serializer.CircuitSerializer]:

```python
from sparc.io import CircuitSerializer

json_text = CircuitSerializer.dumps(root)
root = CircuitSerializer.load("model.json")
```

Shared subgraphs (DAG reuse) are deduplicated by object identity during
serialization.

## Pre-trained circuits

[`load_learned_pc`][sparc.io.learned_pc.load_learned_pc] loads circuits from a
standard directory layout:

```
{base_dir}/{structure}/{dataset}/{block_size}/
    {structure}_{dataset}_blocksize{block_size}_seed{seed}.json
```

```python
from sparc.io import load_learned_pc

circuit = load_learned_pc(
    "examples/example_pcs",
    structure="hclt",
    dataset="adult",
    block_size=2,
    seed=0,
)
```

## Supported leaf kinds

| Kind | Fields |
|------|--------|
| `categorical` | `scope`, `params` (PMF) |
| `bernoulli` | `scope`, `p` |
| `literal` | `scope`, `value` |
| `indicator` | `scope`, `value`, `num_cats` |
| `discrete_logistic` | `scope`, `mu`, `s`, `num_cats` |
| `sum` | `params`, `children` |
| `product` | `children` |

See the [serialization format handbook](../handbook/serialization-format.md)
for the full JSON schema.
