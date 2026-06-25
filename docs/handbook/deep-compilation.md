# Deep Compilation

Ultra-fast inference for circuits compiled to native C with a **mutable parameter
tape**, **SIMD runtime dispatch** (scalar / AVX2 / AVX-512), and **OpenMP
row-block** batch evaluation.

## When to use

| API | Use when |
|-----|----------|
| `Circuit.likelihood` | Debugging, custom leaves |
| `circuit.compile()` | Training or inference with `refresh_parameters()` |
| `circuit.deep_compile()` | Production inference on a fixed topology |

Deep compilation trades some flexibility for speed:

- Topology is fixed at `deep_compile` time; parameter values live in a **tape**
  updated via `refresh_parameters()` (no recompile for weight/PMF changes).
- Requires a **C compiler** (gcc/clang) when you call `deep_compile`, not when
  installing SparC from a wheel.
- **Fully observed integer** assignments only (no `NaN` / `-1` marginalization in v1).
- No gradients, sampling, or query APIs on the deep-compiled object.

## Quick start

```python
import numpy as np
from sparc import Circuit, BernoulliInputNode, SumNode

l0 = BernoulliInputNode(id=0, scope_var=0, p=0.9)
l1 = BernoulliInputNode(id=1, scope_var=1, p=0.3)
circuit = Circuit(SumNode(id=2, children=[l0, l1], parameters=[0.8, 0.2]))

deep = circuit.deep_compile()  # managed temp artifacts; call close() when done
row = np.array([1, 0], dtype=np.int32)
ll = deep.log_likelihood(row)

batch = np.array([[1, 0], [0, 1]], dtype=np.int32)
lls = deep.log_likelihood(batch)

# After MLE steps on the live circuit:
deep.refresh_parameters()

print(deep.active_isa)  # e.g. "avx2" after CPUID dispatch
deep.close()  # unload native code and delete temp artifacts
```

Pass an optional path stem (e.g. ``circuit.deep_compile("/tmp/mymodel")``) to
keep ``.c`` / ``.so`` files on disk; :meth:`close` then only unloads the library.

Use as a context manager to ensure cleanup::

```python
with circuit.deep_compile() as deep:
    lls = deep.log_likelihood(batch)
```

The API mirrors [`CompiledCircuit`](compiled-evaluation.md): 1-D input returns a
scalar; 2-D batch returns `(n_samples,)`. Optional `var_to_col` remaps variables
to columns for batched data.

## Generated code

Each circuit emits a small **glue** `.c` file (~KB) containing:

- A static `SparcOp` table (node kind, children, tape indices, leaf column ids).
- Thin exports `sparc_likelihood_row` / `sparc_log_likelihood_batch` that call
  the shared SIMD runtime via `sparc_dispatch()`.

The hand-tuned runtime under `sparc/deep_compile/rt/` implements:

- **Binary leaf gather** — compare + blend on pre-extracted `leaf_ev` columns.
- **Products** — vectorized multiply / log-add over child workspace rows.
- **Sums** — FMA linear mixes; scalar logsumexp for log mixtures.
- **Batch layout** — node-major workspace `workspace[node * n_rows + r]`.
- **OpenMP** — one parallel region over row tiles (not per-node loops).

Inspect `deep.source_path` for the glue file and `deep.library_path` for the
linked shared library (glue + runtime objects).

## Compiler options

```python
deep = circuit.deep_compile(
    "/tmp/mymodel",
    compiler="clang",
    flags=("-O3", "-std=c11", "-march=native", "-ffast-math"),
    parallel=True,   # OpenMP row tiles + -fopenmp (default True)
    simd="multi",    # link scalar+AVX2+AVX-512; CPUID picks at load
    tile=128,        # rows per OpenMP tile
)
```

`simd` can also be `"scalar"`, `"avx2"`, or `"avx512"` to force the dispatch
path (useful for tests). Set `SPARC_DEEP_ISA=scalar` in the environment before
load to override CPUID without recompiling.

Default flags include `-march=native` and `-ffast-math` for inference throughput.
`-ffast-math` is not suitable when bit-identical numerics with the object path
are required; pass conservative `flags` if needed.

On failure, the `.c` source is kept for debugging; stderr from the compiler is
included in the raised `RuntimeError`.

## Code size and compile time

The glue file is tiny (an op table plus wrappers), so `deep_compile()` on large
circuits is much faster than the old per-node scalar loop emission. SIMD kernels
live in the shared runtime and are compiled once per invocation.

## Parity

Deep-compiled likelihoods match `circuit.compile()` on the same fully observed
integer data (see `tests/test_deep_compile.py`). Compare backends with
`examples/bench_inference.py`.
