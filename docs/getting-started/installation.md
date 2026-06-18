# Installation

SparC ships Cython/C++17 extensions, so a C++ compiler is required.

## Editable install

```bash
pip install -e .
```

This compiles all extension modules in place. On Linux and macOS you need
`g++` or `clang++`; on Windows, MSVC with C++17 support.

## Optional extras

```bash
pip install -e ".[dev]"    # pytest, scipy (solver cross-checks in tests)
pip install -e ".[docs]"   # MkDocs site builder
pip install -e ".[gurobi]" # optional Gurobi extra (not used by core library)
```

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- C++17 compiler

## Verify the install

```python
from sparc import Circuit, CategoricalInputNode, SumNode, ProductNode

x = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.7, 0.3])
root = SumNode(id=1, children=[x], parameters=[1.0])
Circuit(root).log_likelihood({0: 0})
```

## Building documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000` to browse the site.
