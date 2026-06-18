# Builders

The [`sparc.builders`][sparc.builders] subpackage constructs random circuits
for benchmarks, experiments, and baselines.

## EmbeddingBuilder

Recursively partitions a fixed variable scope with optional node reuse:

```python
from sparc.builders import EmbeddingBuilder

circuit = EmbeddingBuilder(
    num_vars=8,
    num_categories=3,
    sum_arity=2,
    prod_arity=2,
    sum_concentration=1.0,
    sum_reuse_probability=0.0,
    prod_reuse_probability=0.0,
    input_distribution="categorical",
    alpha=1.0,
).build()
```

Parameters:

- `sum_arity` / `prod_arity`: branching at sum and product nodes.
- `sum_reuse_probability` / `prod_reuse_probability`: probability of reusing
  a cached subtree for the same scope.
- `input_distribution`: `"categorical"` (Dirichlet PMF) or `"binomial"`.

## Region graphs

For explicit hierarchical structure, build a region graph first:

```python
from sparc.builders import RandomRegionGraph, RegionEmbeddingBuilder

rg = RandomRegionGraph(
    starting_scope=frozenset(range(8)),
    partitions_per_region=2,
    sub_regions_per_partition=2,
)
region = rg.generate(frozenset(range(8)))

circuit = RegionEmbeddingBuilder(
    region_graph=region,
    num_categories=3,
    block_size=2,
    sum_concentration=1.0,
    input_distribution="categorical",
    alpha=1.0,
).build()
```

Each region is realized as `block_size` parallel sum nodes; partitions become
product nodes over aligned sub-region blocks.

## API reference

- [`EmbeddingBuilder`][sparc.builders.embedding.EmbeddingBuilder]
- [`RegionEmbeddingBuilder`][sparc.builders.embedding.RegionEmbeddingBuilder]
- [`RandomRegionGraph`][sparc.builders.region_graph.RandomRegionGraph]
