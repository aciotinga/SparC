"""ctypes wrapper for deep-compiled native inference libraries."""

from __future__ import annotations

import ctypes
import os
import platform
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from sparc._graph import CompiledCircuit
from sparc.deep_compile.compiler import (
    DEFAULT_FLAGS,
    compile_shared,
    register_dll_search_paths,
)
from sparc.deep_compile.emitter import NODE_INPUT, emit_c_source, leaf_var_order
from sparc.deep_compile.tape import TapeLayout, fill_linear_tape, fill_log_tape, make_tape_buffers

_DEEP_DATA_ERROR = (
    "deep-compiled circuits require fully observed integer assignments"
)

_TAPE_PTR = ctypes.POINTER(ctypes.c_double)
_I32_PTR = ctypes.POINTER(ctypes.c_int32)
_F64_PTR = ctypes.POINTER(ctypes.c_double)


def _coerce_deep_data(data: np.ndarray, *, allow_1d: bool) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if np.issubdtype(data.dtype, np.floating):
        raise ValueError(_DEEP_DATA_ERROR)
    arr = np.ascontiguousarray(data, dtype=np.int32)
    if allow_1d and arr.ndim == 1:
        if np.any(arr < 0):
            raise ValueError(_DEEP_DATA_ERROR)
        return arr
    if arr.ndim == 2:
        if np.any(arr < 0):
            raise ValueError(_DEEP_DATA_ERROR)
        return arr
    if allow_1d:
        raise ValueError("data must be 1-D or 2-D (n_samples, n_columns)")
    raise ValueError("data must be 2-D (n_samples, n_columns)")


def _build_col_for_var(
    snap: dict,
    var_to_col: Optional[dict[int, int]],
    n_cols: int,
) -> np.ndarray:
    max_var = snap["max_var"]
    col = np.arange(max_var + 1, dtype=np.int32)
    leaf_vars = {
        snap["leaf_var"][n]
        for n in range(snap["n_nodes"])
        if snap["kinds"][n] == NODE_INPUT
    }
    if var_to_col is not None:
        for var in leaf_vars:
            col[var] = int(var_to_col[var])
    for var in leaf_vars:
        c = int(col[var])
        if c < 0 or c >= n_cols:
            raise ValueError(
                f"variable {var} maps to column {c} out of range [0, {n_cols})"
            )
    return np.ascontiguousarray(col, dtype=np.int32)


def _validate_row(data: np.ndarray, snap: dict) -> None:
    max_var = snap["max_var"]
    if data.size <= max_var:
        raise ValueError(
            f"assignment array length {data.size} is shorter than required "
            f"width {max_var + 1}"
        )
    for n in range(snap["n_nodes"]):
        if snap["kinds"][n] != NODE_INPUT:
            continue
        var = snap["leaf_var"][n]
        val = int(data[var])
        card = snap["leaf_card"][n]
        if val < 0 or val >= card:
            raise ValueError(
                f"evidence for variable {var}: outcome {val} out of range "
                f"[0, {card})"
            )


def _validate_batch(
    data: np.ndarray,
    snap: dict,
    col_for_var: np.ndarray,
) -> None:
    for n in range(snap["n_nodes"]):
        if snap["kinds"][n] != NODE_INPUT:
            continue
        var = snap["leaf_var"][n]
        col = int(col_for_var[var])
        card = snap["leaf_card"][n]
        values = data[:, col]
        if np.any(values < 0) or np.any(values >= card):
            raise ValueError(
                f"evidence for variable {var}: outcome out of range [0, {card})"
            )


def _extract_leaf_ev(
    data: np.ndarray,
    col_for_var: np.ndarray,
    leaf_vars: Sequence[int],
    out: np.ndarray,
) -> np.ndarray:
    """Fill *out* with shape (n_leaf, n_rows) from batch *data*."""
    n_rows = data.shape[0]
    n_leaf = len(leaf_vars)
    need = n_leaf * n_rows
    if out.shape != (n_leaf, n_rows):
        if out.size < need:
            out = np.empty((n_leaf, n_rows), dtype=np.int32)
        else:
            out = out.reshape(n_leaf, n_rows)
    for i, var in enumerate(leaf_vars):
        out[i, :] = data[:, int(col_for_var[var])]
    return np.ascontiguousarray(out, dtype=np.int32)


class DeepCompiledCircuit:
    """Native deep-compiled circuit for ultra-fast inference with mutable tape."""

    def __init__(
        self,
        compiled: CompiledCircuit,
        source_path: Path,
        library_path: Path,
        *,
        tile: int = 128,
        parallel: bool = True,
        artifact_tmpdir: tempfile.TemporaryDirectory[str] | None = None,
    ):
        self._compiled = compiled
        self._snap = compiled.codegen_snapshot()
        self._layout = TapeLayout.from_snapshot(self._snap)
        self.source_path = Path(source_path)
        self.library_path = Path(library_path)
        self.max_var = int(self._snap["max_var"])
        self.variables = list(compiled.variables)
        self.tile = int(tile)
        self.parallel = bool(parallel)
        self._artifact_tmpdir = artifact_tmpdir
        self._closed = False

        self._tape_lin, self._tape_log = make_tape_buffers(self._snap)
        self._workspace = np.empty(self._layout.n_nodes, dtype=np.float64)
        self._workspace_rows = 1
        self._leaf_ev_buffer = np.empty(0, dtype=np.int32)
        self._leaf_vars = leaf_var_order(self._snap)
        self._col_for_var: Optional[np.ndarray] = None

        register_dll_search_paths()
        try:
            lib = ctypes.CDLL(str(self.library_path))
        except OSError as exc:
            raise OSError(
                f"failed to load deep-compiled library {self.library_path}: {exc}. "
                "On Windows with MinGW/gcc, ensure the compiler bin directory is on "
                "PATH or reinstall with parallel=True (static OpenMP link)."
            ) from exc
        lib.sparc_likelihood_row.argtypes = [_TAPE_PTR, _I32_PTR]
        lib.sparc_likelihood_row.restype = ctypes.c_double
        lib.sparc_log_likelihood_row.argtypes = [_TAPE_PTR, _I32_PTR]
        lib.sparc_log_likelihood_row.restype = ctypes.c_double

        batch_args = [
            _TAPE_PTR,
            _I32_PTR,
            ctypes.c_int32,
            ctypes.c_int32,
            _F64_PTR,
            _F64_PTR,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.sparc_likelihood_batch.argtypes = batch_args
        lib.sparc_likelihood_batch.restype = None
        lib.sparc_log_likelihood_batch.argtypes = batch_args
        lib.sparc_log_likelihood_batch.restype = None

        lib.sparc_init_dispatch.argtypes = []
        lib.sparc_init_dispatch.restype = None
        lib.sparc_active_isa_name.argtypes = []
        lib.sparc_active_isa_name.restype = ctypes.c_char_p
        lib.sparc_force_isa.argtypes = [ctypes.c_char_p]
        lib.sparc_force_isa.restype = None
        lib.sparc_workspace_doubles.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ]
        lib.sparc_workspace_doubles.restype = ctypes.c_int32

        isa_override = os.environ.get("SPARC_DEEP_ISA")
        if isa_override:
            lib.sparc_force_isa(isa_override.encode("ascii"))
        lib.sparc_init_dispatch()
        self._active_isa = lib.sparc_active_isa_name().decode("ascii")

        self._lib = lib

    @property
    def active_isa(self) -> str:
        """Name of the SIMD path selected at load (scalar, avx2, avx512)."""
        return self._active_isa

    def refresh_parameters(self) -> None:
        """Update tape buffers from live circuit parameters (no recompile)."""
        self._compiled.refresh_parameters()
        snap = self._compiled.codegen_snapshot()
        if TapeLayout.from_snapshot(snap).tape_len != self._layout.tape_len:
            raise RuntimeError(
                "topology changed; deep_compile again after structural edits"
            )
        self._snap = snap
        fill_linear_tape(snap, self._tape_lin)
        fill_log_tape(snap, self._tape_log)

    def _ensure_workspace(self, n_rows: int) -> np.ndarray:
        need = int(
            self._lib.sparc_workspace_doubles(
                ctypes.c_int32(self._layout.n_nodes),
                ctypes.c_int32(self.tile),
                ctypes.c_int32(1 if self.parallel else 0),
            )
        )
        if self._workspace.size < need:
            self._workspace = np.empty(need, dtype=np.float64)
        self._workspace_rows = n_rows
        return self._workspace

    def close(self) -> None:
        """Unload the native library and remove managed build artifacts."""
        if self._closed:
            return
        self._closed = True

        lib = getattr(self, "_lib", None)
        self._lib = None
        if lib is not None:
            handle = getattr(lib, "_handle", None)
            if handle is not None:
                if platform.system() == "Windows":
                    ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
                else:
                    libc = ctypes.CDLL(None)
                    libc.dlclose(ctypes.c_void_p(handle))

        tmpdir = self._artifact_tmpdir
        self._artifact_tmpdir = None
        if tmpdir is not None:
            tmpdir.cleanup()

    def __enter__(self) -> "DeepCompiledCircuit":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def likelihood(
        self,
        data: np.ndarray,
        var_to_col: Optional[dict[int, int]] = None,
    ):
        return self._score(data, var_to_col, log_space=False)

    def log_likelihood(
        self,
        data: np.ndarray,
        var_to_col: Optional[dict[int, int]] = None,
    ):
        return self._score(data, var_to_col, log_space=True)

    def _score(
        self,
        data: np.ndarray,
        var_to_col: Optional[dict[int, int]],
        *,
        log_space: bool,
    ):
        if self._lib is None:
            raise RuntimeError("deep-compiled circuit is closed")
        arr = _coerce_deep_data(data, allow_1d=True)
        tape = self._tape_log if log_space else self._tape_lin
        tape_ptr = tape.ctypes.data_as(_TAPE_PTR)
        row_fn = (
            self._lib.sparc_log_likelihood_row
            if log_space
            else self._lib.sparc_likelihood_row
        )
        batch_fn = (
            self._lib.sparc_log_likelihood_batch
            if log_space
            else self._lib.sparc_likelihood_batch
        )

        if arr.ndim == 1:
            _validate_row(arr, self._snap)
            ev = np.ascontiguousarray(arr, dtype=np.int32)
            return row_fn(tape_ptr, ev.ctypes.data_as(_I32_PTR))

        n_rows, n_cols = arr.shape
        col_for_var = _build_col_for_var(self._snap, var_to_col, n_cols)
        _validate_batch(arr, self._snap, col_for_var)
        leaf_ev = _extract_leaf_ev(
            arr, col_for_var, self._leaf_vars, self._leaf_ev_buffer
        )
        self._leaf_ev_buffer = leaf_ev
        workspace = self._ensure_workspace(n_rows)
        out = np.empty(n_rows, dtype=np.float64)
        batch_fn(
            tape_ptr,
            leaf_ev.ctypes.data_as(_I32_PTR),
            ctypes.c_int32(n_rows),
            ctypes.c_int32(n_rows),
            workspace.ctypes.data_as(_F64_PTR),
            out.ctypes.data_as(_F64_PTR),
            ctypes.c_int32(self.tile),
            ctypes.c_int32(1 if self.parallel else 0),
        )
        return out


def deep_compile_circuit(
    root,
    path: str | Path | None = None,
    *,
    compiler: str | None = None,
    flags: Sequence[str] = DEFAULT_FLAGS,
    parallel: bool = True,
    simd: str = "multi",
    tile: int = 128,
) -> DeepCompiledCircuit:
    """Build, emit, compile, and load a deep-compiled circuit.

  When *path* is omitted, build artifacts live in a managed temporary
  directory that is removed by :meth:`DeepCompiledCircuit.close`.
    """
    from sparc.circuit import Circuit

    if isinstance(root, Circuit):
        root = root.root
    compiled = CompiledCircuit(root)
    snap = compiled.codegen_snapshot()
    source = emit_c_source(snap, parallel=parallel, tile=tile)

    artifact_tmpdir: tempfile.TemporaryDirectory[str] | None = None
    if path is None:
        artifact_tmpdir = tempfile.TemporaryDirectory(
            prefix="sparc_deep_", ignore_cleanup_errors=True
        )
        stem = Path(artifact_tmpdir.name) / "circuit"
    else:
        stem = Path(path)

    source_path = stem.with_suffix(".c")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    library_path = compile_shared(
        source_path,
        stem,
        compiler=compiler,
        flags=flags,
        parallel=parallel,
    )
    deep = DeepCompiledCircuit(
        compiled,
        source_path,
        library_path,
        tile=tile,
        parallel=parallel,
        artifact_tmpdir=artifact_tmpdir,
    )
    if simd != "multi":
        deep._lib.sparc_force_isa(simd.encode("ascii"))
        deep._lib.sparc_init_dispatch()
        deep._active_isa = deep._lib.sparc_active_isa_name().decode("ascii")
    return deep
