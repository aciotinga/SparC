"""ctypes wrapper for deep-compiled native inference libraries."""

from __future__ import annotations

import ctypes
import os
import platform
import tempfile
import warnings
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from sparc._graph import CompiledCircuit
from sparc.deep_compile.compiler import (
    DEFAULT_FLAGS,
    compile_shared,
    register_dll_search_paths,
)
from sparc.deep_compile.cache import (
    build_cache_key,
    copy_to_stem,
    store_cached,
    try_load_cached,
)
from sparc.deep_compile.emitter import NODE_INPUT, emit_c_source
from sparc.deep_compile.isa import (
    IsaPlan,
    compile_flags_for_opt,
    merge_compile_flags,
    resolve_isa,
)
from sparc.deep_compile.tape import TapeLayout, fill_linear_tape, fill_log_tape, make_tape_buffers

try:
    from sparc.deep_compile._native import (
        eval_log_likelihood_batch as _eval_log_batch_cython,
        eval_likelihood_batch as _eval_lin_batch_cython,
    )
except ImportError:
    _eval_log_batch_cython = None
    _eval_lin_batch_cython = None

_DEEP_DATA_ERROR = (
    "deep-compiled circuits require fully observed integer assignments"
)

_TAPE_PTR = ctypes.POINTER(ctypes.c_double)
_I32_PTR = ctypes.POINTER(ctypes.c_int32)
_F64_PTR = ctypes.POINTER(ctypes.c_double)

_L3_BUDGET_BYTES = 8 * 1024 * 1024
_MIN_TILE = 32

PerformanceMode = Literal["max", "balanced"]
CompileOpt = Literal["fast", "max"]


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


def _col_cache_key(var_to_col: Optional[dict[int, int]], n_cols: int) -> tuple:
    if var_to_col is None:
        return (None, n_cols)
    return (frozenset(var_to_col.items()), n_cols)


def _effective_exec_params(
    n_nodes: int,
    n_rows: int,
    tile: int,
    parallel: bool,
    *,
    performance: PerformanceMode,
) -> tuple[int, bool]:
    """Adapt tile size and OpenMP use; ``performance='max'`` disables L3 cap."""
    tile = max(_MIN_TILE, int(tile))
    if performance == "max":
        if parallel and n_rows < (os.cpu_count() or 1) * tile:
            parallel = False
        return tile, parallel

    n_threads = (os.cpu_count() or 1) if parallel else 1
    ws_bytes = n_nodes * tile * 8 * n_threads
    if parallel and ws_bytes > _L3_BUDGET_BYTES:
        parallel = False
        n_threads = 1
        ws_bytes = n_nodes * tile * 8
    if n_nodes > 0:
        max_tile = max(_MIN_TILE, _L3_BUDGET_BYTES // (n_nodes * 8 * max(n_threads, 1)))
        if tile > max_tile:
            tile = max(_MIN_TILE, max_tile)
    if parallel and n_rows < n_threads * tile:
        parallel = False
    return tile, parallel


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
        performance: PerformanceMode = "max",
        isa_plan: IsaPlan,
        mode: str = "ultra",
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
        self.performance = performance
        self._mode = mode
        self._isa_plan = isa_plan
        self._artifact_tmpdir = artifact_tmpdir
        self._closed = False

        self._tape_lin, self._tape_log = make_tape_buffers(self._snap)
        self._workspace = np.empty(self._layout.n_nodes, dtype=np.float64)
        self._workspace_rows = 1
        self._col_for_var: Optional[np.ndarray] = None
        self._col_cache_key: Optional[tuple] = None
        self._validated_once = False

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

        if mode == "ultra":
            ultra_batch_args = [
                _TAPE_PTR,
                _I32_PTR,
                ctypes.c_int32,
                _I32_PTR,
                ctypes.c_int32,
                _F64_PTR,
                _F64_PTR,
                ctypes.c_int32,
                ctypes.c_int32,
            ]
            lib.sparc_likelihood_batch.argtypes = ultra_batch_args
            lib.sparc_likelihood_batch.restype = None
            lib.sparc_log_likelihood_batch.argtypes = ultra_batch_args
            lib.sparc_log_likelihood_batch.restype = None
            self._lin_batch_ptr = ctypes.cast(
                lib.sparc_likelihood_batch, ctypes.c_void_p
            ).value or 0
            self._log_batch_ptr = ctypes.cast(
                lib.sparc_log_likelihood_batch, ctypes.c_void_p
            ).value or 0
        else:
            compat_batch_args = [
                _TAPE_PTR,
                _I32_PTR,
                ctypes.c_int32,
                ctypes.c_int32,
                _F64_PTR,
                _F64_PTR,
                ctypes.c_int32,
                ctypes.c_int32,
            ]
            lib.sparc_likelihood_batch.argtypes = compat_batch_args
            lib.sparc_likelihood_batch.restype = None
            lib.sparc_log_likelihood_batch.argtypes = compat_batch_args
            lib.sparc_log_likelihood_batch.restype = None
            self._lin_batch_ptr = 0
            self._log_batch_ptr = 0

        lib.sparc_workspace_doubles.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ]
        lib.sparc_workspace_doubles.restype = ctypes.c_int32
        lib.sparc_active_isa_name.argtypes = []
        lib.sparc_active_isa_name.restype = ctypes.c_char_p

        if mode == "compat":
            lib.sparc_init_dispatch.argtypes = []
            lib.sparc_init_dispatch.restype = None
            lib.sparc_force_isa.argtypes = [ctypes.c_char_p]
            lib.sparc_force_isa.restype = None
            isa_override = os.environ.get("SPARC_DEEP_ISA")
            if isa_override:
                lib.sparc_force_isa(isa_override.encode("ascii"))
            lib.sparc_init_dispatch()

        self._active_isa = lib.sparc_active_isa_name().decode("ascii")
        self._lib = lib

    @property
    def active_isa(self) -> str:
        """ISA the native library was compiled for (or dispatch-selected in compat)."""
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
        self._col_for_var = None
        self._col_cache_key = None
        self._validated_once = False
        fill_linear_tape(snap, self._tape_lin)
        fill_log_tape(snap, self._tape_log)

    def _get_col_for_var(
        self,
        var_to_col: Optional[dict[int, int]],
        n_cols: int,
    ) -> np.ndarray:
        key = _col_cache_key(var_to_col, n_cols)
        if self._col_for_var is not None and self._col_cache_key == key:
            return self._col_for_var
        col = _build_col_for_var(self._snap, var_to_col, n_cols)
        self._col_for_var = col
        self._col_cache_key = key
        return col

    def _ensure_workspace(self, n_rows: int, tile: int, parallel: bool) -> np.ndarray:
        need = int(
            self._lib.sparc_workspace_doubles(
                ctypes.c_int32(self._layout.n_nodes),
                ctypes.c_int32(tile),
                ctypes.c_int32(1 if parallel else 0),
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
        *,
        validate: bool = True,
    ):
        return self._score(data, var_to_col, log_space=False, validate=validate)

    def log_likelihood(
        self,
        data: np.ndarray,
        var_to_col: Optional[dict[int, int]] = None,
        *,
        validate: bool = True,
    ):
        return self._score(data, var_to_col, log_space=True, validate=validate)

    def _score(
        self,
        data: np.ndarray,
        var_to_col: Optional[dict[int, int]],
        *,
        log_space: bool,
        validate: bool,
    ):
        if self._lib is None:
            raise RuntimeError("deep-compiled circuit is closed")
        arr = _coerce_deep_data(data, allow_1d=True)
        tape = self._tape_log if log_space else self._tape_lin
        row_fn = (
            self._lib.sparc_log_likelihood_row
            if log_space
            else self._lib.sparc_likelihood_row
        )

        if arr.ndim == 1:
            if validate:
                _validate_row(arr, self._snap)
            ev = np.ascontiguousarray(arr, dtype=np.int32)
            return row_fn(tape.ctypes.data_as(_TAPE_PTR), ev.ctypes.data_as(_I32_PTR))

        n_rows, n_cols = arr.shape
        col_for_var = self._get_col_for_var(var_to_col, n_cols)
        if validate and not self._validated_once:
            _validate_batch(arr, self._snap, col_for_var)
            self._validated_once = True
        tile, parallel = _effective_exec_params(
            self._layout.n_nodes,
            n_rows,
            self.tile,
            self.parallel,
            performance=self.performance,
        )
        workspace = self._ensure_workspace(n_rows, tile, parallel)
        out = np.empty(n_rows, dtype=np.float64)

        if self._mode == "ultra":
            self._run_ultra_batch(
                log_space=log_space,
                tape=tape,
                data=arr,
                col_for_var=col_for_var,
                workspace=workspace,
                out=out,
                tile=tile,
                parallel=parallel,
            )
        else:
            self._run_compat_batch(
                log_space=log_space,
                tape=tape,
                data=arr,
                col_for_var=col_for_var,
                workspace=workspace,
                out=out,
                tile=tile,
                parallel=parallel,
            )
        return out

    def _run_ultra_batch(
        self,
        *,
        log_space: bool,
        tape: np.ndarray,
        data: np.ndarray,
        col_for_var: np.ndarray,
        workspace: np.ndarray,
        out: np.ndarray,
        tile: int,
        parallel: bool,
    ) -> None:
        fn_ptr = self._log_batch_ptr if log_space else self._lin_batch_ptr
        eval_cython = _eval_log_batch_cython if log_space else _eval_lin_batch_cython
        n_cols = data.shape[1]
        par_i = 1 if parallel else 0

        if eval_cython is not None and fn_ptr:
            eval_cython(
                fn_ptr,
                tape,
                data,
                col_for_var,
                workspace,
                out,
                tile,
                par_i,
            )
            return

        batch_fn = (
            self._lib.sparc_log_likelihood_batch
            if log_space
            else self._lib.sparc_likelihood_batch
        )
        batch_fn(
            tape.ctypes.data_as(_TAPE_PTR),
            data.ctypes.data_as(_I32_PTR),
            ctypes.c_int32(n_cols),
            col_for_var.ctypes.data_as(_I32_PTR),
            ctypes.c_int32(data.shape[0]),
            workspace.ctypes.data_as(_F64_PTR),
            out.ctypes.data_as(_F64_PTR),
            ctypes.c_int32(tile),
            ctypes.c_int32(par_i),
        )

    def _run_compat_batch(
        self,
        *,
        log_space: bool,
        tape: np.ndarray,
        data: np.ndarray,
        col_for_var: np.ndarray,
        workspace: np.ndarray,
        out: np.ndarray,
        tile: int,
        parallel: bool,
    ) -> None:
        from sparc.deep_compile.compat_emitter import leaf_var_order

        n_rows = data.shape[0]
        n_leaf = len(leaf_var_order(self._snap))
        leaf_ev = np.empty((n_leaf, n_rows), dtype=np.int32)
        leaf_vars = np.asarray(leaf_var_order(self._snap), dtype=np.int32)
        for i, var in enumerate(leaf_vars):
            leaf_ev[i, :] = data[:, col_for_var[var]]

        batch_fn = (
            self._lib.sparc_log_likelihood_batch
            if log_space
            else self._lib.sparc_likelihood_batch
        )
        batch_fn(
            tape.ctypes.data_as(_TAPE_PTR),
            leaf_ev.ctypes.data_as(_I32_PTR),
            ctypes.c_int32(n_rows),
            ctypes.c_int32(n_rows),
            workspace.ctypes.data_as(_F64_PTR),
            out.ctypes.data_as(_F64_PTR),
            ctypes.c_int32(tile),
            ctypes.c_int32(1 if parallel else 0),
        )


def deep_compile_circuit(
    root,
    path: str | Path | None = None,
    *,
    compiler: str | None = None,
    flags: Sequence[str] | None = None,
    parallel: bool = True,
    tile: int = 128,
    isa: str | None = None,
    performance: PerformanceMode = "max",
    compile_opt: CompileOpt = "fast",
    mode: str = "ultra",
    simd: str | None = None,
    use_cache: bool = True,
) -> DeepCompiledCircuit:
    """Build, emit, compile, and load a deep-compiled circuit.

    When *path* is omitted, build artifacts live in a managed temporary
    directory that is removed by :meth:`DeepCompiledCircuit.close`.
    """
    from sparc.circuit import Circuit
    from sparc.deep_compile.compiler import _library_extension

    if isinstance(root, Circuit):
        root = root.root

    if simd is not None:
        warnings.warn(
            "simd= is deprecated; use isa= instead",
            DeprecationWarning,
            stacklevel=2,
        )
        isa = simd if simd != "multi" else isa

    base_flags = tuple(flags) if flags is not None else compile_flags_for_opt(compile_opt)
    if mode == "ultra":
        isa_plan = resolve_isa(isa)
        compile_flags = merge_compile_flags(base_flags, isa_plan, parallel=parallel)
    else:
        isa_plan = resolve_isa(isa if isa not in (None, "multi") else None)
        compile_flags = base_flags

    compiled = CompiledCircuit(root)
    snap = compiled.codegen_snapshot()

    lib_suffix = _library_extension()
    cache_key: str | None = None
    cached: tuple[Path, Path] | None = None
    if use_cache and mode == "ultra":
        cache_key = build_cache_key(
            snap,
            isa=isa_plan.name,
            parallel=parallel,
            tile=tile,
            compile_opt=compile_opt,
            mode=mode,
            flags=compile_flags,
        )
        cached = try_load_cached(cache_key, lib_suffix=lib_suffix)

    source: str | None = None
    if cached is None:
        source = emit_c_source(
            snap,
            isa=isa_plan if mode == "ultra" else None,
            parallel=parallel,
            tile=tile,
            mode=mode,
        )

    artifact_tmpdir: tempfile.TemporaryDirectory[str] | None = None
    if path is None:
        if cached is not None:
            source_path, library_path = cached
        else:
            artifact_tmpdir = tempfile.TemporaryDirectory(
                prefix="sparc_deep_", ignore_cleanup_errors=True
            )
            stem = Path(artifact_tmpdir.name) / "circuit"
            source_path = stem.with_suffix(".c")
            source_path.write_text(source, encoding="utf-8")
            library_path = compile_shared(
                source_path,
                stem,
                compiler=compiler,
                flags=compile_flags,
                parallel=parallel,
                mode=mode,
                isa=isa_plan if mode == "ultra" else None,
            )
            if cache_key is not None:
                source_path, library_path = store_cached(
                    cache_key,
                    source_text=source,
                    source_path=source_path,
                    library_path=library_path,
                    metadata={
                        "isa": isa_plan.name,
                        "parallel": parallel,
                        "tile": tile,
                        "compile_opt": compile_opt,
                    },
                )
    else:
        stem = Path(path)
        if cached is not None:
            source_path, library_path = copy_to_stem(
                cached[1], cached[0], stem, lib_suffix=lib_suffix
            )
        else:
            source_path = stem.with_suffix(".c")
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source, encoding="utf-8")
            library_path = compile_shared(
                source_path,
                stem,
                compiler=compiler,
                flags=compile_flags,
                parallel=parallel,
                mode=mode,
                isa=isa_plan if mode == "ultra" else None,
            )
            if cache_key is not None:
                store_cached(
                    cache_key,
                    source_text=source,
                    source_path=source_path,
                    library_path=library_path,
                    metadata={
                        "isa": isa_plan.name,
                        "parallel": parallel,
                        "tile": tile,
                        "compile_opt": compile_opt,
                    },
                )

    return DeepCompiledCircuit(
        compiled,
        source_path,
        library_path,
        tile=tile,
        parallel=parallel,
        performance=performance,
        isa_plan=isa_plan,
        mode=mode,
        artifact_tmpdir=artifact_tmpdir,
    )
