"""Host ISA detection and compile-time ISA plans for ultra deep compile."""

from __future__ import annotations

import logging
import os
import platform
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

IsaName = Literal["avx512", "avx2", "scalar"]

logger = logging.getLogger(__name__)

# Leaf indices (EBX bits) for CPUID leaf 7, subleaf 0
_EBX_AVX2 = 1 << 5
_EBX_AVX512F = 1 << 16
_EBX_AVX512DQ = 1 << 17


@dataclass(frozen=True)
class IsaPlan:
    """Compile-time ISA selection and compiler flags."""

    name: IsaName
    vector_lanes: int
    compile_flags: tuple[str, ...]
    msvc_arch: str | None


@lru_cache(maxsize=1)
def _cpuid_leaf7_ebx() -> int:
    """Return EBX from CPUID leaf 7 subleaf 0, or 0 if unavailable."""
    env = os.environ.get("SPARC_HOST_CPUID_EBX")
    if env:
        return int(env, 0)

    if platform.machine().lower() not in ("amd64", "x86_64", "x86"):
        return 0
    try:
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as f:
                    flags_line = ""
                    for line in f:
                        if line.startswith("flags"):
                            flags_line = line
                            break
                if flags_line:
                    flags = flags_line.split(":", 1)[1].strip().split()
                    ebx = 0
                    if "avx2" in flags:
                        ebx |= _EBX_AVX2
                    if "avx512f" in flags:
                        ebx |= _EBX_AVX512F
                    if "avx512dq" in flags:
                        ebx |= _EBX_AVX512DQ
                    return ebx
            except OSError:
                pass

        if platform.system() == "Windows":
            # Avoid slow subprocess probes; x64 Windows targets assume AVX2.
            return _EBX_AVX2

        if platform.system() == "Darwin":
            if platform.machine().lower() == "arm64":
                return 0
            return _EBX_AVX2

        return _EBX_AVX2
    except Exception:
        return 0


def _host_supports(name: IsaName) -> bool:
    ebx = _cpuid_leaf7_ebx()
    if name == "scalar":
        return True
    if name == "avx2":
        return (ebx & _EBX_AVX2) != 0
    if name == "avx512":
        return (ebx & (_EBX_AVX512F | _EBX_AVX512DQ)) == (_EBX_AVX512F | _EBX_AVX512DQ)
    return False


@lru_cache(maxsize=1)
def detect_host_isa() -> IsaName:
    """Return the best ISA supported on the current host."""
    if _host_supports("avx512"):
        return "avx512"
    if _host_supports("avx2"):
        return "avx2"
    return "scalar"


def resolve_isa(requested: str | None) -> IsaPlan:
    """Resolve ISA name and compiler flags for ultra emission."""
    if requested is None:
        name = detect_host_isa()
    else:
        key = requested.lower().strip()
        if key not in ("avx512", "avx2", "scalar"):
            raise ValueError(f"unknown isa {requested!r}; use avx512, avx2, or scalar")
        name = key  # type: ignore[assignment]
        if not _host_supports(name):
            warnings.warn(
                f"requested ISA {name!r} may not be supported on this CPU; "
                "compiled code can fault at runtime",
                stacklevel=2,
            )

    if name == "avx512":
        return IsaPlan(
            name="avx512",
            vector_lanes=8,
            compile_flags=("-mavx512f", "-mavx512dq"),
            msvc_arch="/arch:AVX512",
        )
    if name == "avx2":
        return IsaPlan(
            name="avx2",
            vector_lanes=4,
            compile_flags=("-mavx2", "-mfma"),
            msvc_arch="/arch:AVX2",
        )
    return IsaPlan(
        name="scalar",
        vector_lanes=1,
        compile_flags=(),
        msvc_arch=None,
    )


def merge_compile_flags(
    base: tuple[str, ...],
    isa: IsaPlan,
    *,
    parallel: bool,
) -> tuple[str, ...]:
    """Combine user flags with ISA and OpenMP flags."""
    flags = list(base)
    for f in isa.compile_flags:
        if f not in flags:
            flags.append(f)
    if "-funroll-loops" not in flags:
        flags.append("-funroll-loops")
    if parallel and "-fopenmp" not in flags:
        flags.append("-fopenmp")
    return tuple(flags)


def compile_flags_for_opt(compile_opt: str) -> tuple[str, ...]:
    """Return base compiler flags for *compile_opt* (``fast`` or ``max``)."""
    from sparc.deep_compile.compiler import FAST_COMPILE_FLAGS, MAX_COMPILE_FLAGS

    key = compile_opt.lower().strip()
    if key == "max":
        return MAX_COMPILE_FLAGS
    if key != "fast":
        raise ValueError(f"unknown compile_opt {compile_opt!r}; use fast or max")
    return FAST_COMPILE_FLAGS
