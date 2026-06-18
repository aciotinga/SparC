"""Minimize E_Q[P(X)] over a learnable Q with P fixed.

Minimizing ``log E_Q[P(X)]`` is equivalent to minimizing ``E_Q[P(X)]`` (the
expectation is positive) and is numerically better behaved. Each step applies
projected simplex *descent* to every parameter of ``circuit2`` (Q).

    python examples/exp_minimization.py
"""

import random

import numpy as np

from sparc.builders import RandomRegionGraph, RegionEmbeddingBuilder
from sparc.optim import apply_grads
from sparc.queries import log_exp_query, log_exp_query_and_grad


def build_pair(num_vars=6):
    """Two structured-decomposable circuits sharing a region graph (compatible
    structure required by exp_query), with independently sampled parameters."""
    random.seed(0)
    rg = RandomRegionGraph(
        frozenset(range(num_vars)),
        partitions_per_region=1,  # single partition => structured decomposable
        sub_regions_per_partition=2,
    )
    root_region = rg.generate(frozenset(range(num_vars)))

    def build(seed):
        np.random.seed(seed)
        return RegionEmbeddingBuilder(
            root_region, num_categories=3, block_size=2,
            sum_concentration=1.0, input_distribution="categorical", alpha=1.0,
        ).build()

    return build(0), build(1)


def main():
    circuit1, circuit2 = build_pair()  # fixed P, learnable Q

    init = log_exp_query(circuit1, circuit2)
    print(f"initial log(E) = {init:.8f}   E = {np.exp(init):.8e}")

    for step in range(1, 51):
        # log_exp_query_and_grad returns (value, grad_circuit1, grad_circuit2)
        val, _, grad2 = log_exp_query_and_grad(circuit1, circuit2)
        apply_grads(circuit2, grad2, lr=1e-2, ascent=False)  # descent => minimize
        if step % 10 == 0:
            print(f"  step {step:4d}: log(E) = {val:.8f}   E = {np.exp(val):.8e}")

    final = log_exp_query(circuit1, circuit2)
    print(f"\nfinal log(E)   = {final:.8f}   E = {np.exp(final):.8e}")
    print(f"log improvement = {final - init:+.8f}  (should be negative)")


if __name__ == "__main__":
    main()
