# Quick Start

This page walks through constructing a small circuit, evaluating it, and
sampling from it.

## Construct a circuit

A probabilistic circuit is a DAG of **sum** (mixture), **product**
(factorization), and **input** (leaf) nodes:

```python
import numpy as np
from sparc import CategoricalInputNode, SumNode, ProductNode, CircuitNode

# Two categorical variables mixed by a sum over their product
x0 = CategoricalInputNode(scope_var=0, probabilities=[0.7, 0.3])
x1 = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
prod = ProductNode(children=[x0, x1])
root = SumNode(children=[prod], parameters=[1.0])

circuit = root
```

## Inference

```python
point = np.array([0, 1], dtype=np.int32)
circuit.log_likelihood(point)   # scalar
circuit.likelihood(point)

# Vectorized log-likelihood: rows = datapoints, cols = variables
data = np.random.randint(0, 2, size=(1000, 2)).astype(np.int32)
circuit.compile().log_likelihood(data)

# Marginal query: NaN marks a missing variable (summed out)
partial = np.array([0.0, np.nan], dtype=np.float64)
circuit.log_likelihood(partial)
```

## Sampling

```python
samples = circuit.sample(5, seed=0)  # shape (5, max_var+1), int32
# samples[i, var] is the outcome for variable var; -1 where out of scope
```

## Save and load

```python
circuit.save("model.json")
loaded = CircuitNode.load("model.json")
```

See [Serialization](../guides/serialization.md) for the on-disk format and
[Training](training.md) for parameter learning.
