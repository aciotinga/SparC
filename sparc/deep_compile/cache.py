"""Artifact cache for deep-compiled circuit libraries."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Sequence

_CACHE_VERSION = 1


def default_cache_dir() -> Path:
    env = os.environ.get("SPARC_DEEP_CACHE")
    if env:
        return Path(env)
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "sparc" / "deep"
    return Path.home() / ".cache" / "sparc" / "deep"


def _topology_digest(snap: dict[str, Any]) -> bytes:
    """Hash frozen topology fields from a codegen snapshot."""
    h = hashlib.blake2b(digest_size=32)
    for key in (
        "n_nodes",
        "root_index",
        "max_var",
        "kinds",
        "child_off",
        "children_flat",
        "leaf_var",
        "leaf_card",
        "leaf_pmf_off",
    ):
        val = snap[key]
        h.update(repr(val).encode("utf-8"))
    return h.digest()


def build_cache_key(
    snap: dict[str, Any],
    *,
    isa: str,
    parallel: bool,
    tile: int,
    compile_opt: str,
    mode: str,
    flags: Sequence[str],
) -> str:
    """Return hex cache key for a deep-compile configuration."""
    h = hashlib.blake2b(digest_size=32)
    h.update(_topology_digest(snap))
    h.update(isa.encode("ascii"))
    h.update(b"\x01" if parallel else b"\x00")
    h.update(str(tile).encode("ascii"))
    h.update(compile_opt.encode("ascii"))
    h.update(mode.encode("ascii"))
    h.update("\0".join(flags).encode("utf-8"))
    h.update(str(_CACHE_VERSION).encode("ascii"))
    return h.hexdigest()


def cache_entry_dir(cache_key: str, *, cache_dir: Path | None = None) -> Path:
    root = default_cache_dir() if cache_dir is None else cache_dir
    return root / cache_key


def try_load_cached(
    cache_key: str,
    *,
    lib_suffix: str,
    cache_dir: Path | None = None,
) -> tuple[Path, Path] | None:
    """Return (source_path, library_path) if a valid cache entry exists."""
    entry = cache_entry_dir(cache_key, cache_dir=cache_dir)
    lib_path = entry / f"circuit{lib_suffix}"
    src_path = entry / "circuit.c"
    meta_path = entry / "metadata.json"
    if not lib_path.is_file() or not src_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_key") != cache_key:
            return None
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return src_path, lib_path


def store_cached(
    cache_key: str,
    *,
    source_text: str,
    source_path: Path,
    library_path: Path,
    metadata: dict[str, Any],
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write artifacts into the cache and return canonical paths."""
    entry = cache_entry_dir(cache_key, cache_dir=cache_dir)
    entry.mkdir(parents=True, exist_ok=True)
    lib_suffix = library_path.suffix
    dst_src = entry / "circuit.c"
    dst_lib = entry / f"circuit{lib_suffix}"
    dst_src.write_text(source_text, encoding="utf-8")
    shutil.copy2(library_path, dst_lib)
    meta = dict(metadata)
    meta["cache_key"] = cache_key
    (entry / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return dst_src, dst_lib


def copy_to_stem(
    cached_lib: Path,
    cached_src: Path,
    stem: Path,
    *,
    lib_suffix: str,
) -> tuple[Path, Path]:
    """Copy cached artifacts to an explicit output stem."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    out_src = stem.with_suffix(".c")
    out_lib = stem.with_suffix(lib_suffix)
    shutil.copy2(cached_src, out_src)
    shutil.copy2(cached_lib, out_lib)
    return out_src, out_lib
