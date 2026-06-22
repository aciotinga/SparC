import sys
import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

_cc = sysconfig.get_config_var("CC") or ""
if sys.platform == "win32" and "gcc" not in _cc.lower() and "clang" not in _cc.lower():
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    extra_compile_args = ["-O3", "-std=c++17"]

_numpy_include = np.get_include()

# Every Cython extension module in SparC. All are C++ (libcpp containers,
# C++ <random>, etc.).
_pyx_modules = [
    "sparc.nodes",
    "sparc._graph",
    "sparc.eval",
    "sparc.grad",
    "sparc.metrics",
    "sparc.solvers.northwest",
    "sparc.solvers.transport",
    "sparc.solvers.assignment",
    "sparc.queries._engine",
    "sparc.queries.esd",
    "sparc.queries.expectation",
    "sparc.queries.cw",
    "sparc.queries.gcw",
]


def _to_path(dotted: str) -> str:
    return dotted.replace(".", "/") + ".pyx"


ext_modules = cythonize(
    [
        Extension(
            name,
            [_to_path(name)],
            language="c++",
            include_dirs=[_numpy_include],
            extra_compile_args=extra_compile_args,
        )
        for name in _pyx_modules
    ],
    compiler_directives={"language_level": "3", "embedsignature": True},
)

setup(ext_modules=ext_modules)
