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
    "sparc.sampling",
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


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _libomp_prefix():
    for path in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if os.path.isdir(path):
            return path
    return None


def _openmp_config():
    """OpenMP flags for sparc._graph only. macOS is serial unless SPARC_OPENMP=1."""
    if _truthy("SPARC_NO_OPENMP"):
        return False, [], []
    if sys.platform == "darwin":
        if not _truthy("SPARC_OPENMP"):
            return False, [], []
        prefix = _libomp_prefix()
        compile_args = ["-Xpreprocessor", "-fopenmp"]
        link_args = ["-lomp"]
        if prefix:
            compile_args.extend(["-I" + os.path.join(prefix, "include")])
            link_args.extend(["-L" + os.path.join(prefix, "lib")])
        return True, compile_args, link_args
    if _is_msvc:
        return True, ["/openmp"], []
    return True, ["-fopenmp"], ["-fopenmp"]


_use_openmp, _omp_cflags, _omp_lflags = _openmp_config()

_pxi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparc", "_openmp_flag.pxi")
with open(_pxi_path, "w", encoding="utf-8") as _pxi:
    _pxi.write(f"DEF SPARC_OPENMP = {1 if _use_openmp else 0}\n")

_extensions = []
for _name in _pyx_modules:
    _cflags = list(extra_compile_args)
    _lflags = list(extra_link_args)
    if _name == "sparc._graph":
        _cflags.extend(_omp_cflags)
        _lflags.extend(_omp_lflags)
    _extensions.append(
        Extension(
            _name,
            [_to_path(_name)],
            language="c++",
            include_dirs=[_numpy_include],
            extra_compile_args=_cflags,
            extra_link_args=_lflags,
        )
    )

ext_modules = cythonize(
    _extensions,
    compiler_directives={"language_level": "3", "embedsignature": True},
)

setup(ext_modules=ext_modules)
