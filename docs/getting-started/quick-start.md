# Quick Start

This page walks through constructing a small circuit, evaluating it, and
sampling from it.

## Construct a circuit

A probabilistic circuit is a DAG of **sum** (mixture), **product**
(factorization), and **input** (leaf) nodes:

```python
import numpy as np
from sparc import CategoricalInputNode, SumNode, ProductNode, Circuit

# Two categorical variables mixed by a sum over their product
x0 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.7, 0.3])
x1 = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
prod = ProductNode(id=2, children=[x0, x1])
root = SumNode(id=3, children=[prod], parameters=[1.0])

circuit = Circuit(root)
```

## Inference

```python
circuit.log_likelihood({0: 0, 1: 1})   # single datapoint
circuit.likelihood({0: 0, 1: 1})

# Vectorized log-likelihood: rows = datapoints, cols = variables
data = np.random.randint(0, 2, size=(1000, 2)).astype(np.int32)
circuit.compile().log_likelihood(data)
```

## Sampling

```python
samples = circuit.sample(5, seed=0)
# [{0: 0, 1: 1}, ...]
```

## Save and load

```python
circuit.save("model.json")
loaded = Circuit.load("model.json")
```

See [Serialization](../guides/serialization.md) for the on-disk format and
[Training](training.md) for parameter learning.
