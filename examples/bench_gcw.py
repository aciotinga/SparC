"""Benchmark the GCW cross-term forward and forward+grad computations.

Usage:
    python examples/bench_gcw.py
"""

import argparse
import cProfile
import pstats
import random
import time

import numpy as np

from sparc.builders import EmbeddingBuilder
from sparc.queries import gcw_crossterm, gcw_crossterm_and_grad


def make_circuit(seed, num_vars=10, num_categories=3, sum_arity=2, prod_arity=2):
    np.random.seed(seed)
    random.seed(seed)
    return EmbeddingBuilder(
        num_vars=num_vars, num_categories=num_categories,
        sum_arity=sum_arity, prod_arity=prod_arity,
        sum_concentration=1.0, sum_reuse_probability=0.0,
        prod_reuse_probability=0.0, input_distribution="categorical", alpha=1.0,
    ).build()


def time_call(fn, repeats):
    best = float("inf")
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
    return best, total / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    configs = [
        dict(num_vars=10, num_categories=3, sum_arity=2, prod_arity=2),
        dict(num_vars=10, num_categories=5, sum_arity=3, prod_arity=3),
        dict(num_vars=15, num_categories=5, sum_arity=3, prod_arity=3),
    ]

    print(f"{'config':<48}{'fwd best (ms)':>16}{'fwd+grad best (ms)':>22}")
    print("-" * 86)
    for cfg in configs:
        c1 = make_circuit(0, **cfg)
        c2 = make_circuit(1, **cfg)
        gcw_crossterm(c1, c2)
        gcw_crossterm_and_grad(c1, c2)

        fwd_best, _ = time_call(lambda: gcw_crossterm(c1, c2), args.repeats)
        grad_best, _ = time_call(lambda: gcw_crossterm_and_grad(c1, c2), args.repeats)
        label = (f"vars={cfg['num_vars']} cats={cfg['num_categories']} "
                 f"sa={cfg['sum_arity']} pa={cfg['prod_arity']}")
        print(f"{label:<48}{fwd_best * 1e3:>16.3f}{grad_best * 1e3:>22.3f}")

    if args.profile:
        print("\n=== cProfile (fwd+grad, largest config) ===")
        cfg = configs[-1]
        c1 = make_circuit(0, **cfg)
        c2 = make_circuit(1, **cfg)
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.repeats):
            gcw_crossterm_and_grad(c1, c2)
        pr.disable()
        stats = pstats.Stats(pr).sort_stats("cumulative")
        stats.print_stats(25)


if __name__ == "__main__":
    main()
