# Installation

## PyPI install (recommended)

Prebuilt wheels are available for Linux, Windows, and macOS (Python 3.10–3.14):

```bash
pip install sparc-pc
```

The distribution name on PyPI is **`sparc-pc`**; import as **`sparc`**:

```python
import sparc
from sparc import CategoricalInputNode, SumNode
```

No C++ compiler is needed when installing from a wheel.

## Editable install (from source)

SparC ships Cython/C++17 extensions, so a C++ compiler is required when
building from source.

```bash
pip install -e .
```

This compiles all extension modules in place. On Linux and macOS you need
`g++` or `clang++`; on Windows, MSVC with C++17 support.

The default build enables aggressive portable optimizations (including
link-time optimization). The first compile may take longer than a typical
Cython project. For faster iterative rebuilds during development, disable
LTO:

```bash
SPARC_FAST_BUILD=1 pip install -e .
```

## Optional extras

```bash
pip install "sparc-pc[dev]"   # pytest, scipy (from PyPI)
pip install -e ".[dev]"       # same, editable from source
pip install -e ".[docs]"      # MkDocs site builder
pip install -e ".[gurobi]"    # optional Gurobi extra (not used by core library)
```

## Requirements

- Python >= 3.10, < 3.15
- NumPy >= 1.20
- C++17 compiler (source/editable installs only)

## Verify the install

```python
from sparc import CategoricalInputNode, SumNode, ProductNode

x = CategoricalInputNode(scope_var=0, probabilities=[0.7, 0.3])
root = SumNode(children=[x], parameters=[1.0])
root.log_likelihood(np.array([0], dtype=np.int32))
```

## Building documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000` to browse the site.
