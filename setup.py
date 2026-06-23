import os
import sys
import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

_fast_build = os.environ.get("SPARC_FAST_BUILD", "").lower() in ("1", "true", "yes")

_cc = sysconfig.get_config_var("CC") or ""
_is_msvc = (
    sys.platform == "win32"
    and "gcc" not in _cc.lower()
    and "clang" not in _cc.lower()
)

if _is_msvc:
    extra_compile_args = ["/std:c++17", "/Ox", "/fp:fast"]
    extra_link_args = []
    if not _fast_build:
        extra_compile_args.append("/GL")
        extra_link_args.append("/LTCG")
else:
    # Avoid -ffast-math: it breaks log-sum-exp guards (isfinite(-inf), -inf
    # comparisons) and causes log_exp_query to return -1 instead of -inf on Linux.
    extra_compile_args = ["-std=c++17", "-O3", "-funroll-loops"]
    extra_link_args = []
    if not _fast_build:
        extra_compile_args.append("-flto")
        extra_link_args.append("-flto")

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
            extra_link_args=extra_link_args,
        )
        for name in _pyx_modules
    ],
    compiler_directives={"language_level": "3", "embedsignature": True},
)

setup(ext_modules=ext_modules)
