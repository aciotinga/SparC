"""Compile generated C sources into platform shared libraries."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import hashlib

from sparc.deep_compile.isa import IsaPlan, merge_compile_flags

FAST_COMPILE_FLAGS: tuple[str, ...] = ("-O2", "-std=c11", "-ffast-math")
MAX_COMPILE_FLAGS: tuple[str, ...] = ("-O3", "-std=c11", "-march=native", "-ffast-math")
DEFAULT_FLAGS: tuple[str, ...] = FAST_COMPILE_FLAGS

_RT_DIR = Path(__file__).resolve().parent / "rt"
_ULTRA_RT_SOURCE = "sparc_ultra_rt.c"

_RT_SCALAR_SOURCES = (
    "sparc_deep_rt_scalar.c",
    "sparc_deep_dispatch.c",
)

_RT_AVX2_SOURCES = ("sparc_deep_rt_avx2.c",)

_RT_AVX512_SOURCES = ("sparc_deep_rt_avx512.c",)


def _library_extension() -> str:
    system = platform.system()
    if system == "Windows":
        return ".dll"
    if system == "Darwin":
        return ".dylib"
    return ".so"


def find_compiler(compiler: str | None = None) -> str | None:
    """Return a usable C compiler executable path, or None."""
    if compiler is not None:
        return compiler if shutil.which(compiler) else None
    cc_env = os.environ.get("CC")
    if cc_env and shutil.which(cc_env):
        return cc_env
    for name in ("gcc", "clang", "cc", "cl"):
        if shutil.which(name):
            return name
    return None


def compiler_available(compiler: str | None = None) -> bool:
    return find_compiler(compiler) is not None


def _is_msvc(compiler: str) -> bool:
    base = Path(compiler).name.lower()
    return base in {"cl", "cl.exe"}


def compiler_bin_dir(compiler: str | None = None) -> Path | None:
    """Return the directory containing runtime DLLs for *compiler*, if known."""
    cc = find_compiler(compiler)
    if cc is None:
        return None
    return Path(cc).resolve().parent


def register_dll_search_paths(compiler: str | None = None) -> None:
    """On Windows, add the compiler runtime directory to the DLL search path."""
    if platform.system() != "Windows":
        return
    bin_dir = compiler_bin_dir(compiler)
    if bin_dir is None or not bin_dir.is_dir():
        return
    try:
        os.add_dll_directory(str(bin_dir))
    except (AttributeError, OSError):
        path = os.environ.get("PATH", "")
        prefix = str(bin_dir)
        if prefix.lower() not in path.lower():
            os.environ["PATH"] = prefix + os.pathsep + path


def _rt_cache_key(cc: str, isa: IsaPlan, flags: Sequence[str], parallel: bool) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(Path(cc).resolve().as_posix().encode("utf-8"))
    h.update(isa.name.encode("ascii"))
    h.update(b"\x01" if parallel else b"\x00")
    h.update("\0".join(flags).encode("utf-8"))
    return h.hexdigest()


def _get_ultra_rt_object(
    cc: str,
    isa: IsaPlan,
    *,
    flags: Sequence[str],
    parallel: bool,
    build_dir: Path,
) -> Path:
    """Return path to cached sparc_ultra_rt object for this compiler/ISA."""
    rt_cache = default_cache_dir() / "rt"
    rt_cache.mkdir(parents=True, exist_ok=True)
    key = _rt_cache_key(cc, isa, flags, parallel)
    obj_path = rt_cache / f"sparc_ultra_rt_{isa.name}_{key}.o"
    if obj_path.is_file():
        return obj_path

    src = _RT_DIR / _ULTRA_RT_SOURCE
    extra: list[str] = []
    if isa.msvc_arch and _is_msvc(cc):
        extra = [isa.msvc_arch]
    _compile_object(
        cc,
        src,
        obj_path,
        include_dir=_RT_DIR,
        flags=flags,
        extra_flags=extra,
        parallel=parallel,
    )
    return obj_path


def default_cache_dir() -> Path:
    from sparc.deep_compile.cache import default_cache_dir as _dd

    return _dd()


def _windows_mingw_link_flags(*, parallel: bool) -> list[str]:
    """Link flags so MinGW-built DLLs load without MSYS2 on PATH."""
    flags: list[str] = []
    if parallel:
        flags.extend(["-Wl,-Bstatic", "-lgomp", "-lpthread", "-Wl,-Bdynamic"])
    flags.append("-static-libgcc")
    return flags


def _compile_object(
    cc: str,
    source: Path,
    obj_path: Path,
    *,
    include_dir: Path,
    flags: Sequence[str],
    extra_flags: Sequence[str] = (),
    parallel: bool = False,
) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    compile_flags = list(flags)
    if parallel and not _is_msvc(cc):
        if "-fopenmp" not in compile_flags:
            compile_flags.append("-fopenmp")

    if _is_msvc(cc):
        cmd = [
            cc,
            "/nologo",
            "/c",
            "/O2",
            f"/I{include_dir}",
            *extra_flags,
            f"/Fo{obj_path}",
            str(source),
        ]
        if parallel:
            cmd.insert(1, "/openmp")
    else:
        cmd = [
            cc,
            "-c",
            "-fPIC",
            f"-I{include_dir}",
            *compile_flags,
            *extra_flags,
            "-o",
            str(obj_path),
            str(source),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"failed to compile {source} with {cc}:\n{stderr}")


def _link_shared(
    cc: str,
    objects: Sequence[Path],
    lib_path: Path,
    *,
    flags: Sequence[str],
    parallel: bool = False,
) -> None:
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    link_flags: list[str] = []
    if parallel and not _is_msvc(cc):
        link_flags.append("-fopenmp")
        if platform.system() == "Windows":
            link_flags.extend(_windows_mingw_link_flags(parallel=True))

    if _is_msvc(cc):
        cmd = [cc, "/nologo", "/LD", "/O2", f"/Fe:{lib_path}", *[str(o) for o in objects]]
        if parallel:
            cmd.insert(1, "/openmp")
    else:
        cmd = [
            cc,
            "-shared",
            *flags,
            "-o",
            str(lib_path),
            *[str(o) for o in objects],
            *link_flags,
        ]
        if platform.system() != "Windows":
            cmd.append("-lm")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"failed to link {lib_path} with {cc}:\n{stderr}")


def _try_compile_rt_objects(
    cc: str,
    build_dir: Path,
    *,
    include_dir: Path,
    flags: Sequence[str],
    parallel: bool,
) -> list[Path]:
    """Compile runtime translation units; return object paths."""
    objects: list[Path] = []

    for name in _RT_SCALAR_SOURCES:
        src = include_dir / name
        obj = build_dir / f"{Path(name).stem}.o"
        _compile_object(
            cc, src, obj, include_dir=include_dir, flags=flags, parallel=parallel
        )
        objects.append(obj)

    for name in _RT_AVX2_SOURCES:
        src = include_dir / name
        obj = build_dir / f"{Path(name).stem}.o"
        if _is_msvc(cc):
            extra = ["/arch:AVX2"]
        else:
            extra = ["-mavx2", "-mfma"]
        _compile_object(
            cc,
            src,
            obj,
            include_dir=include_dir,
            flags=flags,
            extra_flags=extra,
            parallel=parallel,
        )
        objects.append(obj)

    for name in _RT_AVX512_SOURCES:
        src = include_dir / name
        obj = build_dir / f"{Path(name).stem}.o"
        if _is_msvc(cc):
            extra = ["/arch:AVX512"]
        else:
            extra = ["-mavx512f", "-mavx512dq"]
        try:
            _compile_object(
                cc,
                src,
                obj,
                include_dir=include_dir,
                flags=flags,
                extra_flags=extra,
                parallel=parallel,
            )
            objects.append(obj)
        except RuntimeError:
            pass

    return objects


def compile_shared(
    source_path: Path,
    output_stem: Path,
    *,
    compiler: str | None = None,
    flags: Sequence[str] = DEFAULT_FLAGS,
    parallel: bool = False,
    build_dir: Path | None = None,
    mode: str = "ultra",
    isa: IsaPlan | None = None,
) -> Path:
    """Compile generated C into a shared library.

    *mode* ``"ultra"`` compiles only the self-contained circuit source.
    *mode* ``"compat"`` links the legacy SIMD dispatch runtime objects.
    """
    if mode == "ultra":
        if isa is None:
            from sparc.deep_compile.isa import resolve_isa

            isa = resolve_isa(None)
        flags = merge_compile_flags(tuple(flags), isa, parallel=parallel)
        return _compile_ultra_shared(
            source_path,
            output_stem,
            compiler=compiler,
            flags=flags,
            parallel=parallel,
            isa=isa,
            build_dir=build_dir,
        )
    return _compile_compat_shared(
        source_path,
        output_stem,
        compiler=compiler,
        flags=flags,
        parallel=parallel,
        build_dir=build_dir,
    )


def _compile_ultra_shared(
    source_path: Path,
    output_stem: Path,
    *,
    compiler: str | None,
    flags: Sequence[str],
    parallel: bool,
    isa: IsaPlan,
    build_dir: Path | None,
) -> Path:
    cc = find_compiler(compiler)
    if cc is None:
        raise RuntimeError(
            "no C compiler found; install gcc or clang, or pass compiler=..."
        )

    source_path = Path(source_path)
    output_stem = Path(output_stem)
    lib_path = output_stem.with_suffix(_library_extension())

    own_build_dir = build_dir is None
    if own_build_dir:
        build_dir = Path(tempfile.mkdtemp(prefix="sparc_deep_"))
    else:
        build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    extra: list[str] = []
    if isa.msvc_arch and _is_msvc(cc):
        extra = [isa.msvc_arch]

    try:
        glue_obj = build_dir / "sparc_circuit.o"
        rt_obj = _get_ultra_rt_object(
            cc, isa, flags=flags, parallel=parallel, build_dir=build_dir
        )
        _compile_object(
            cc,
            source_path,
            glue_obj,
            include_dir=_RT_DIR,
            flags=flags,
            extra_flags=extra,
            parallel=parallel,
        )
        _link_shared(cc, [glue_obj, rt_obj], lib_path, flags=flags, parallel=parallel)
    finally:
        if own_build_dir:
            shutil.rmtree(build_dir, ignore_errors=True)

    if not lib_path.is_file():
        raise RuntimeError(f"compiler did not produce {lib_path}")
    return lib_path


def _compile_compat_shared(
    source_path: Path,
    output_stem: Path,
    *,
    compiler: str | None = None,
    flags: Sequence[str] = DEFAULT_FLAGS,
    parallel: bool = False,
    build_dir: Path | None = None,
) -> Path:
    """Compile glue *source_path* plus SIMD runtime into a shared library."""
    cc = find_compiler(compiler)
    if cc is None:
        raise RuntimeError(
            "no C compiler found; install gcc or clang, or pass compiler=..."
        )

    source_path = Path(source_path)
    output_stem = Path(output_stem)
    lib_path = output_stem.with_suffix(_library_extension())
    rt_dir = _RT_DIR
    include_dir = rt_dir

    own_build_dir = build_dir is None
    if own_build_dir:
        build_dir = Path(tempfile.mkdtemp(prefix="sparc_deep_"))
    else:
        build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        glue_obj = build_dir / "sparc_graph.o"
        _compile_object(
            cc,
            source_path,
            glue_obj,
            include_dir=include_dir,
            flags=flags,
            parallel=parallel,
        )
        rt_objects = _try_compile_rt_objects(
            cc,
            build_dir,
            include_dir=include_dir,
            flags=flags,
            parallel=parallel,
        )
        _link_shared(
            cc,
            [glue_obj, *rt_objects],
            lib_path,
            flags=flags,
            parallel=parallel,
        )
    finally:
        if own_build_dir:
            shutil.rmtree(build_dir, ignore_errors=True)

    if not lib_path.is_file():
        raise RuntimeError(f"compiler did not produce {lib_path}")
    return lib_path


def smoke_compile(compiler: str | None = None) -> bool:
    """Return True if a trivial OpenMP shared library can be built and loaded."""
    cc = find_compiler(compiler)
    if cc is None:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "smoke.c"
            stem = Path(tmp) / "smoke"
            lib_path = stem.with_suffix(_library_extension())
            src.write_text(
                "#include <stdint.h>\n"
                "#ifdef _OPENMP\n"
                "#include <omp.h>\n"
                "#endif\n"
                "double smoke_add(double a, double b) {\n"
                "#ifdef _OPENMP\n"
                "  return a + b + (double)omp_get_max_threads();\n"
                "#else\n"
                "  return a + b;\n"
                "#endif\n"
                "}\n",
                encoding="utf-8",
            )
            if _is_msvc(cc):
                cmd = [cc, "/nologo", "/LD", "/O2", "/openmp", f"/Fe:{lib_path}", str(src)]
            else:
                cmd = [
                    cc,
                    "-shared",
                    "-fPIC",
                    *DEFAULT_FLAGS,
                    "-fopenmp",
                    "-o",
                    str(lib_path),
                    str(src),
                ]
                if platform.system() == "Windows":
                    cmd.extend(_windows_mingw_link_flags(parallel=True))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False
            if not lib_path.is_file():
                return False
            import ctypes

            register_dll_search_paths(cc)
            lib = ctypes.CDLL(str(lib_path))
            lib.smoke_add.argtypes = [ctypes.c_double, ctypes.c_double]
            lib.smoke_add.restype = ctypes.c_double
            if lib.smoke_add(1.0, 2.0) < 3.0:
                return False
            if platform.system() == "Windows":
                handle = getattr(lib, "_handle", None)
                if handle is not None:
                    ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            return True
    except (RuntimeError, OSError, PermissionError):
        return False
