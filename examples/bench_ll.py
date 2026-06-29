"""Benchmark log-likelihood on train data: Circuit vs compile() vs deep_compile().

    python examples/bench_ll.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sparc import Circuit
from sparc.deep_compile.compiler import smoke_compile

_ROOT = Path(__file__).resolve().parent
_PCS = _ROOT / "example_pcs"
_DATA = _ROOT / "original_datasets"

WARMUP = 2
REPEATS = 5


def _best_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    has_deep = smoke_compile()
    header = f"{'circuit':<16} {'rows':>8} {'Circuit':>10} {'compiled':>10}"
    if has_deep:
        header += f" {'deep':>10}"
    print(header)
    print("-" * len(header))

    for circuit_path in sorted(_PCS.glob("*.json")):
        stem = circuit_path.stem
        data_path = _DATA / stem / f"{stem}.train.data"
        if not data_path.is_file():
            continue

        rows = np.loadtxt(data_path, delimiter=",", dtype=np.int32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        rows = np.ascontiguousarray(rows, dtype=np.int32)

        circuit = Circuit.load(circuit_path)
        compiled = circuit.compile()

        ll = lambda ev, fn: float(fn(ev).mean())
        regular_ms = _best_ms(lambda: ll(rows, circuit.log_likelihood))
        compiled_ms = _best_ms(lambda: ll(rows, compiled.log_likelihood))

        line = f"{stem:<16} {rows.shape[0]:8d} {regular_ms:10.2f} {compiled_ms:10.2f}"
        if has_deep:
            deep = circuit.deep_compile()
            try:
                deep_ms = _best_ms(
                    lambda: ll(rows, lambda ev, fn=deep.log_likelihood: fn(ev, validate=False))
                )
            finally:
                deep.close()
            line += f" {deep_ms:10.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
