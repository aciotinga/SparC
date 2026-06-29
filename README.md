# SparC

[![PyPI](https://img.shields.io/pypi/v/sparc-pc)](https://pypi.org/project/sparc-pc/)

**Spar**se **C**ircuits — a fast, modular, CPU-only library for probabilistic
circuits (PCs) written in Cython.

Documentation: [https://sparc-docs.readthedocs.io](https://sparc-docs.readthedocs.io)
(build locally with `pip install -e ".[docs]" && mkdocs serve`).

## Highlights

- **Speed**: typed Cython with C++ STL containers, `nogil` inner loops, and a
  compiled/batched evaluation path.
- **Modularity**: nodes dispatch through a C-level vtable, and all pairwise
  queries share one tape/gradient engine. New leaf types and new queries are
  added without editing existing files.
- **Minimal deps**: `numpy` only at runtime. Transportation and assignment
  problems are solved by built-in pure-Cython solvers.

## Supported queries

| Query | Math | Gradients w.r.t. | Docs |
|-------|------|------------------|------|
| CW | $W_p^p$ (Circuit-Wasserstein objective) | circuit2 | [queries](docs/guides/queries.md) |
| GCW cross-term | Gromov-Circuit-Wasserstein cross-term | circuit2 | [queries](docs/guides/queries.md) |
| Expectation | $E_Q[P(X)]$ | both circuits | [queries](docs/guides/queries.md) |
| Log expectation | $\log E_Q[P(X)]$ | both circuits | [queries](docs/guides/queries.md) |
| ESD | $E[d(X,X')^p]$ (two i.i.d. draws) | single circuit | [queries](docs/guides/queries.md) |

CW and GCW gradient variants return gradients with respect to the **second**
circuit only. Expectation queries return gradients for both circuits.

## Supported structures

| Constructor | Description | Docs |
|-------------|-------------|------|
| `HMM` / `GeneralizedHMM` | Latent-chain sequence models | [structures](docs/guides/structures.md) |
| `HCLT` | Hidden tree from data (MI + MST) | [structures](docs/guides/structures.md) |
| `PD` / `PDHCLT` | Recursive grid decompositions | [structures](docs/guides/structures.md) |
| `RAT_SPN` | Randomized tensorized sum-product network | [structures](docs/guides/structures.md) |
| `EmbeddingBuilder` | Random recursive PC with node reuse | [builders](docs/guides/builders.md) |

Import structures from `sparc.structures` and builders from `sparc.builders`.

## Install

Prebuilt wheels are available for Linux, Windows, and macOS (Python 3.10–3.14):

```bash
pip install sparc-pc
```

The PyPI package name is `sparc-pc`; import as `sparc`:

```python
import sparc
from sparc import CategoricalInputNode, SumNode, ProductNode
```

### From source (developers)

SparC ships Cython/C++17 extensions, so a C++ compiler is required when
installing from source or in editable mode:

```bash
pip install -e .            # build extensions in place
pip install -e ".[dev]"     # + pytest, scipy, build tools
pip install -e ".[docs]"    # + MkDocs documentation builder
pip install -e ".[gurobi]"  # optional Gurobi extra (unused by core library)
```

See [Releasing](docs/releasing.md) for maintainer release steps.

## Quick start

```python
import numpy as np
from sparc import CategoricalInputNode, SumNode, ProductNode

x0 = CategoricalInputNode(scope_var=0, probabilities=[0.7, 0.3])
x1 = CategoricalInputNode(scope_var=1, probabilities=[0.5, 0.5])
prod = ProductNode(children=[x0, x1])
root = SumNode(children=[prod], parameters=[1.0])

circuit = root
point = np.array([0, 1], dtype=np.int32)
circuit.log_likelihood(point)
circuit.sample(5, seed=0)  # ndarray (5, max_var+1)

data = np.random.randint(0, 2, size=(1000, 2)).astype(np.int32)
circuit.compile().log_likelihood(data)
```

### Training

```python
from sparc.optim import MLETrainer

trainer = MLETrainer(circuit, lr=0.5)
trainer.fit(dataset, epochs=100)
```

### Circuit distances

```python
from sparc import cw_distance, gcw_crossterm, gcw_coupling_circuit, PNormMetric

cw_distance(p, q, metric=PNormMetric(p=2.0, scale=1.0))
gcw_crossterm(circuit1, circuit2)
gcw_coupling_circuit(circuit1, circuit2).sample(1000)
```

See the [documentation](docs/index.md) for compatibility rules, all leaf types,
and extension points.

## Package layout

```
sparc/
  node_clone.py    # deep-copy helpers for circuit DAGs
  nodes.pyx        # CircuitNode / Sum / Product / leaf nodes (+ vtable)
  eval.pyx         # likelihood / sampling + CompiledCircuit
  grad.pyx         # GradBundle + mean_log_likelihood_and_grad
  metrics.pyx      # GroundMetric, PNormMetric
  solvers/         # transport, assignment, northwest
  queries/         # CW, GCW, expectation, ESD
  builders/        # region graphs, embedding builders
  structures/      # HMM, HCLT, PD, RAT-SPN, ...
  io/              # gcw-circuit-v1 JSON serializer
  optim.py         # simplex_step, apply_grads, MLETrainer
docs/              # MkDocs site (guides, handbook, API reference)
examples/          # runnable demo scripts
tests/
```

## Examples

```bash
PYTHONPATH=. python examples/mle.py
PYTHONPATH=. python examples/cw_minimization.py
PYTHONPATH=. python examples/gcw_optimization.py --direction max
PYTHONPATH=. python examples/dro.py
```

Full list: [docs/examples/overview.md](docs/examples/overview.md).

## Development

```bash
pip install -e ".[dev]"
pytest

pip install -e ".[docs]"
mkdocs serve        # browse docs at http://127.0.0.1:8000
mkdocs build        # static site in site/
```

## License

MIT
