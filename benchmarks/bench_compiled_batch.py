"""Compiled log-likelihood throughput vs OpenMP thread count.

Not collected by pytest. Run from the repo root after an OpenMP build:

    python benchmarks/bench_compiled_batch.py
"""

from __future__ import annotations

import os
import statistics
import time

import numpy as np

from sparc import (
    CategoricalInputNode,
    GaussianInputNode,
    ProductNode,
    SumNode,
)
from sparc._graph import _omp_enabled, _omp_max_threads, _omp_set_num_threads
from sparc.optim import iter_nodes
from sparc.structures import Bernoulli, GeneralizedHMM, HCLT, RAT_SPN


BATCHES = (1, 8, 32, 128, 512, 2048, 8192)
REPEATS = 7
WARMUP = 2


def _n_nodes(root) -> int:
    return sum(1 for _ in iter_nodes(root))


def _mixed_tiny():
    l0 = CategoricalInputNode(scope_var=5, probabilities=[0.6, 0.4])
    l1 = CategoricalInputNode(scope_var=9, probabilities=[0.25, 0.75])
    prod = ProductNode(children=[l0, l1])
    l2 = CategoricalInputNode(scope_var=5, probabilities=[0.1, 0.9])
    return SumNode(children=[prod, l2], parameters=[0.7, 0.3])


def _int_data(root, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    vars_ = root.scope_as_list()
    width = (max(vars_) + 1) if vars_ else 1
    data = np.zeros((n_rows, width), dtype=np.int32)
    for v in vars_:
        data[:, v] = rng.integers(0, 2, size=n_rows)
    return data


def _float_data(n_rows: int, n_vars: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n_rows, n_vars)).astype(np.float64)


def _median_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _thread_grid() -> list[int]:
    ncpu = os.cpu_count() or 8
    grid = [1, 2, 4, 8]
    if ncpu > 8 and ncpu not in grid:
        grid.append(ncpu)
    return [t for t in grid if t <= max(ncpu, 8)]


def _skip_batch(label: str, n_rows: int, n_nodes: int) -> bool:
    bytes_val = n_nodes * n_rows * 8
    if label == "mnist14" and n_rows >= 8192 and bytes_val > 512 * 1024 * 1024:
        return True
    return False


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"OpenMP compiled: {_omp_enabled()}  max_threads={_omp_max_threads()}")
    circuits = {}

    tiny = _mixed_tiny()
    circuits["tiny"] = tiny.compile()

    hmm = GeneralizedHMM(
        seq_length=32, num_latents=8, input_dist=Bernoulli(), seed=0
    )
    circuits["hmm"] = hmm.compile()

    rat = RAT_SPN(
        num_vars=32,
        num_latents=8,
        depth=3,
        num_repetitions=2,
        num_cats=2,
        seed=0,
    )
    circuits["rat"] = rat.compile()

    hclt_data = rng.integers(0, 2, size=(2000, 64), dtype=np.int32)
    hclt = HCLT(hclt_data, num_latents=8, input_dist=Bernoulli(), seed=0)
    circuits["hclt"] = hclt.compile()

    mnist_data = rng.integers(0, 2, size=(500, 196), dtype=np.int32)
    mnist = HCLT(mnist_data, num_latents=8, input_dist=Bernoulli(), seed=0)
    circuits["mnist14"] = mnist.compile()

    gauss = ProductNode(
        [GaussianInputNode(i, 0.0, 1.0) for i in range(32)]
    )
    circuits["gauss"] = gauss.compile()

    roots = {
        "tiny": tiny,
        "hmm": hmm,
        "rat": rat,
        "hclt": hclt,
        "mnist14": mnist,
        "gauss": gauss,
    }

    threads = _thread_grid()
    prev = int(_omp_max_threads())
    print(
        f"{'circuit':<10} {'nodes':>6} {'vars':>5} {'B':>6} {'T':>3} "
        f"{'ms':>10} {'rows/s':>12} {'vs T=1':>8}"
    )
    try:
        for label, compiled in circuits.items():
            n_nodes = _n_nodes(roots[label])
            n_vars = len(compiled.variables)
            print(f"# {label}: n_nodes={n_nodes} n_vars={n_vars}")
            for n_rows in BATCHES:
                if _skip_batch(label, n_rows, n_nodes):
                    print(f"{label:<10} {n_nodes:>6} {n_vars:>5} {n_rows:>6} skip (val buffer)")
                    continue
                if label == "gauss":
                    data = _float_data(n_rows, 32, rng)
                else:
                    data = _int_data(roots[label], n_rows, rng)
                baseline = None
                for nt in threads:
                    _omp_set_num_threads(nt)
                    ms = _median_ms(lambda: compiled.log_likelihood(data))
                    rps = n_rows / (ms / 1e3) if ms > 0 else float("inf")
                    if nt == 1:
                        baseline = ms
                    speedup = (baseline / ms) if baseline and ms > 0 else float("nan")
                    print(
                        f"{label:<10} {n_nodes:>6} {n_vars:>5} {n_rows:>6} {nt:>3} "
                        f"{ms:10.3f} {rps:12.1f} {speedup:7.2f}x"
                    )
                if label == "hmm" and n_rows == 512:
                    _omp_set_num_threads(1)
                    root = roots[label]
                    obj_ms = _median_ms(
                        lambda r=root, d=data: np.array(
                            [r.log_likelihood(d[i]) for i in range(d.shape[0])]
                        )
                    )
                    print(
                        f"{'hmm-obj':<10} {n_nodes:>6} {n_vars:>5} {n_rows:>6} "
                        f"{'og':>3} {obj_ms:10.3f} {n_rows / (obj_ms / 1e3):12.1f} "
                        f"{(obj_ms / baseline) if baseline else float('nan'):7.2f}x compiled-T1 vs obj"
                    )
    finally:
        _omp_set_num_threads(max(prev, 1))


if __name__ == "__main__":
    main()
