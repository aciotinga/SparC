"""Benchmark batched log-likelihood: Circuit vs compile() vs deep_compile().

    python examples/bench_inference.py plants
    python examples/bench_inference.py adult --repeats 20
    python examples/bench_inference.py model.json --data test.data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from sparc import Circuit
from sparc.deep_compile.compiler import smoke_compile

_ROOT = Path(__file__).resolve().parent
_PCS = _ROOT / "example_pcs"
_DATA = _ROOT / "original_datasets"


def _resolve_circuit(name: str) -> Path:
    path = Path(name)
    for candidate in (path, _PCS / name, _PCS / f"{name}.json"):
        if candidate.is_file():
            return candidate.resolve()
    choices = ", ".join(p.name for p in sorted(_PCS.glob("*.json")))
    raise FileNotFoundError(f"Circuit not found: {name!r} (example_pcs: {choices})")


def _resolve_data(circuit_path: Path, data_arg: str | None, split: str) -> Path:
    if data_arg is not None:
        path = Path(data_arg)
        if not path.is_file():
            raise FileNotFoundError(f"Test data not found: {path}")
        return path.resolve()
    stem = circuit_path.stem
    if stem.startswith("hclt_"):
        rest = stem[5:]
        for sep in ("_blocksize", "_seed"):
            if sep in rest:
                rest = rest.split(sep)[0]
                break
        else:
            rest = rest.split("_")[0]
        stem = rest
    path = _DATA / stem / f"{stem}.{split}.data"
    if not path.is_file():
        raise FileNotFoundError(f"Test data not found: {path} (try --data)")
    return path.resolve()


def _load_rows(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", dtype=np.int32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return np.ascontiguousarray(rows, dtype=np.int32)


def _time_mean_ll(fn, *, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    best, total, value = float("inf"), 0.0, fn()
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
    return best, total / repeats, value


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("circuit", help="JSON path or example_pcs basename (e.g. plants)")
    p.add_argument("--data", help="CSV test rows (default: original_datasets/...)")
    p.add_argument("--split", choices=("test", "valid"), default="test")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--deep-stem", help="Keep deep-compile artifacts at this path stem")
    p.add_argument("--compiler", help="C compiler for deep_compile (default: auto)")
    args = p.parse_args(argv)

    circuit_path = _resolve_circuit(args.circuit)
    data_path = _resolve_data(circuit_path, args.data, args.split)
    rows = _load_rows(data_path)

    print(f"circuit: {circuit_path.name}  data: {data_path.name}  "
          f"({rows.shape[0]} rows x {rows.shape[1]} cols)")

    circuit = Circuit.load(circuit_path)
    mean_ll = lambda ev, fn: float(fn(ev).mean())

    # --- one-time setup ---
    print("\nsetup")
    t0 = time.perf_counter()
    compiled = circuit.compile()
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"  compile()       {compile_ms:8.2f} ms")

    deep = None
    deep_ms = None
    if smoke_compile(args.compiler):
        t0 = time.perf_counter()
        if args.deep_stem:
            Path(args.deep_stem).parent.mkdir(parents=True, exist_ok=True)
            deep = circuit.deep_compile(args.deep_stem, compiler=args.compiler)
        else:
            deep = circuit.deep_compile(compiler=args.compiler)
        deep_ms = (time.perf_counter() - t0) * 1e3
        print(f"  deep_compile()  {deep_ms:8.2f} ms  (ISA: {deep.active_isa})")
    else:
        print("  deep_compile()  skipped (no C compiler)")

    ref_ll = mean_ll(rows, circuit.log_likelihood)
    backends: list[tuple[str, object]] = [
        ("Circuit", circuit.log_likelihood),
        ("CompiledCircuit", compiled.log_likelihood),
    ]
    if deep is not None:
        backends.append(("DeepCompiledCircuit", deep.log_likelihood))

    # --- timed evaluation ---
    print(f"\nevaluation (warmup={args.warmup}, repeats={args.repeats})")
    print(f"{'backend':<20} {'mean LL':>12} {'best ms':>10} {'mean ms':>10} {'rows/s':>10}")
    print("-" * 66)

    best: dict[str, float] = {}
    for name, fn in backends:
        b, m, ll = _time_mean_ll(lambda r=rows, f=fn: mean_ll(r, f),
                                 warmup=args.warmup, repeats=args.repeats)
        best[name] = b
        rps = rows.shape[0] / b if b > 0 else float("inf")
        print(f"{name:<20} {ll:12.6f} {b * 1e3:10.2f} {m * 1e3:10.2f} {rps:10.0f}")
        if abs(ll - ref_ll) > 1e-9:
            print(f"  warning: LL differs from Circuit by {ll - ref_ll:+.3e}")

    if "DeepCompiledCircuit" in best and best["DeepCompiledCircuit"] > 0:
        speedup = best["CompiledCircuit"] / best["DeepCompiledCircuit"]
        print(f"\ndeep / compiled: {speedup:.2f}x")

    if deep is not None:
        deep.close()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
