# SparC

**Spar**se **C**ircuits — a fast, modular, CPU-only library for probabilistic
circuits (PCs) written in Cython. SparC is a clean-room rewrite of the
research-only `fastcircuits` code, keeping what worked (typed Cython, gradient
tapes, OT-based circuit distances) and dropping the technical debt.

## Highlights

- **Speed**: typed Cython with C++ STL containers, `nogil` inner loops, and a
  compiled/batched evaluation path.
- **Modularity**: nodes dispatch through a C-level vtable, and all pairwise
  queries share one tape/gradient engine. New leaf types and new queries are
  added without editing existing files.
- **Minimal deps**: `numpy` only at runtime. The optimal-transport and
  assignment problems are solved by built-in pure-Cython solvers — no SciPy,
  no Gurobi. `gurobipy` is an optional extra for solver cross-checks.

## Install

SparC ships Cython/C++17 extensions, so a C++ compiler is required.

```bash
pip install -e .            # build extensions in place
pip install -e ".[dev]"     # + pytest
pip install -e ".[gurobi]"  # optional Gurobi acceleration
```

## Quick start

```python
import numpy as np
from sparc import CategoricalInputNode, SumNode, ProductNode, Circuit

# x0 ~ Cat, x1 ~ Cat, joint via a product, mixed by a sum
x0 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.7, 0.3])
x1 = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.5, 0.5])
prod = ProductNode(id=2, children=[x0, x1])
root = SumNode(id=3, children=[prod], parameters=[1.0])

circuit = Circuit(root)
circuit.log_likelihood({0: 0, 1: 1})          # single datapoint
circuit.sample(5, seed=0)                       # ancestral sampling

# vectorized log-likelihood over an int matrix (rows = datapoints, cols = vars)
data = np.random.randint(0, 2, size=(1000, 2)).astype(np.int32)
circuit.batched_log_likelihood(data)
```

### Training (maximum likelihood)

```python
from sparc.optim import MLETrainer

trainer = MLETrainer(circuit, lr=0.5)         # projected gradient ascent
trainer.fit(dataset, epochs=100)              # dataset: list[{var: value}]
```

### Circuit distances / queries

All pairwise queries take `Circuit` or raw node roots and return `(value, grads)`
(or `(value, grad_circuit1, grad_circuit2)` for the expectation queries). Grads
are `GradBundle`s keyed by `node.id`.

```python
from sparc import (
    exp_query, log_exp_query,            # E_Q[P(X)] and its log
    cw_distance,                          # Circuit-Wasserstein W_p^p
    gcw_crossterm,                        # Gromov-Circuit-Wasserstein cross-term
    expected_squared_distance,            # single-circuit ESD
)
from sparc import PNormMetric

cw_distance(p, q, metric=PNormMetric(p=2.0, scale=1.0))   # pluggable ground metric
```

The GCW coupling can also be **materialized** as a circuit over the joint space
(`vars(circuit1)` plus `vars(circuit2)` shifted to a disjoint range). Ancestral
sampling from it draws joint `(x, y)` pairs whose marginals are the two inputs:

```python
from sparc import gcw_coupling_circuit

coupling = gcw_coupling_circuit(circuit1, circuit2)   # -> Circuit
coupling.sample(1000)
```

## Extending

- **New leaf type**: subclass `InputNode` (gives likelihood + sampling) or
  `FiniteDiscreteInputNode` (also gets CW/GCW/expectation support) and implement
  a couple of `cdef` hooks. No query code changes.
- **New ground metric**: subclass `GroundMetric` and implement `cost_matrix`.
- **New pairwise query**: subclass the engine's `PairwiseQuery`/`CoupleContext`
  and override only the per-node-pair math hooks.

## Package layout

```
sparc/
  nodes.pyx        # CircuitNode / Sum / Product / Categorical (+ vtable base)
  eval.pyx         # likelihood / log_likelihood / sample + CompiledCircuit
  grad.pyx         # GradBundle + mean_log_likelihood_and_grad
  metrics.pyx      # GroundMetric, PNormMetric
  solvers/         # transport (network simplex + duals), assignment (Hungarian), northwest
  queries/         # _engine + esd / expectation / cw / gcw
  builders/        # region graphs, embedding builders
  io/              # gcw-circuit-v1 JSON serializer, learned-PC loader
  optim.py         # simplex_step, apply_grads, MLETrainer
  circuit.py       # high-level Circuit wrapper (+ clone)
examples/          # mle, exp_minimization, cw_minimization, gcw_optimization, dro
tests/
```

## Examples

```bash
PYTHONPATH=. python examples/mle.py
PYTHONPATH=. python examples/cw_minimization.py
PYTHONPATH=. python examples/gcw_optimization.py --direction max
PYTHONPATH=. python examples/dro.py
```

## License

MIT
