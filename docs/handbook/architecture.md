# Architecture

SparC layers a small Python API over a Cython/C++17 core optimized for CPU
inference and differentiable queries.

```mermaid
flowchart TB
    subgraph userLayer [User Layer]
        RootNode[CircuitNode root]
        Compiled[CompiledCircuit]
        Builders[builders/]
        Structures[structures/]
        Optim[optim.MLETrainer]
        IO[io/ serializer]
    end

    subgraph objectPath [Object-graph path]
        EvalObj[eval likelihood / sample]
        SampleObj[sampling SIMPLE tape / VJP]
        GradObj[grad mean_log_likelihood_and_grad]
        CoupleObj[CoupleContext queries]
    end

    subgraph flatPath [CompiledCircuit flat path]
        Graph[_graph CompiledCircuit]
        EvalFlat[log_likelihood / likelihood / sample]
        SampleFlat[sampling compiled SIMPLE tape / VJP]
        GradFlat[compiled_mean_log_likelihood_and_grad]
        QueriesFlat[CW GCW expectation ESD]
    end

    Builders --> RootNode
    Structures --> RootNode
    RootNode --> objectPath
    RootNode -->|compile once| Compiled
    Compiled --> flatPath
    Optim --> RootNode
```

## Package layout

| Path | Role |
|------|------|
| `sparc/nodes.pyx` | `CircuitNode` types, inference/query API, discrete + continuous leaf vtables |
| `sparc/_continuous.pxd` | Shared nogil Gaussian formulas |
| `sparc/node_clone.py` | Deep-copy helpers |
| `sparc/_graph.pyx` | `CompiledCircuit` flattened layout |
| `sparc/eval.pyx` | Object-graph likelihood / sampling |
| `sparc/sampling.pyx` | Hard-forward SIMPLE sampling and explicit VJPs |
| `sparc/grad.pyx` | `GradBundle`, object + compiled gradients |
| `sparc/metrics.pyx` | Pluggable ground metrics |
| `sparc/queries/` | CW, GCW, expectation, ESD |
| `sparc/solvers/` | Transport, Hungarian, NW coupling |
| `sparc/builders/` | Random circuit construction |
| `sparc/structures/` | HMM, HCLT, PD, RAT-SPN, ... |
| `sparc/io/` | JSON serialization |
| `sparc/optim.py` | Simplex-projected optimization |

## Data flow

1. User builds or loads a circuit (`root`).
2. Object-graph queries walk live nodes with memoization.
3. `circuit.compile()` flattens the DAG into `CompiledCircuit` once.
4. Compiled queries use `nogil` numeric cores over CSR arrays; call `refresh_parameters()` after weight updates.
5. Gradients accumulate into `GradBundle` dicts keyed by `node.id`
   (`sum_grads`, `cat_grads`, `cont_grads`).
6. Differentiable samples retain occurrence-indexed hard choices; their VJP
   reduces those occurrences into the same `GradBundle` format.

## Related handbooks

- [Query engine](query-engine.md)
- [Compiled evaluation](compiled-evaluation.md)
- [Solvers](solvers.md)
