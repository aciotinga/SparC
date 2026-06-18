"""Minimize the Circuit-Wasserstein distance W_p^p(P, Q) over a learnable Q.

``cw_distance_and_grad`` returns subgradients w.r.t. ``circuit2`` (Q); we step Q
toward P. Requires structurally compatible circuits (same region structure),
so we clone P and perturb its parameters to get Q.

    python examples/cw_minimization.py
"""

import random

import numpy as np

from sparc.builders import RandomRegionGraph, RegionEmbeddingBuilder
from sparc.optim import apply_grads
from sparc.queries import cw_distance, cw_distance_and_grad


def main():
    random.seed(0)
    rg = RandomRegionGraph(
        frozenset(range(6)), partitions_per_region=1, sub_regions_per_partition=2
    )
    root_region = rg.generate(frozenset(range(6)))

    def build(seed):
        np.random.seed(seed)
        return RegionEmbeddingBuilder(
            root_region, num_categories=4, block_size=2,
            sum_concentration=1.0, input_distribution="categorical", alpha=1.0,
        ).build()

    p = build(0)
    q = build(1)  # same structure, different params; step q toward p

    init = cw_distance(p, q, metric_p=1.0, scale_factor=1.0)
    print(f"initial CW = {init:.8f}")

    for step in range(1, 41):
        val, grad_q = cw_distance_and_grad(p, q, metric_p=1.0, scale_factor=1.0)
        apply_grads(q, grad_q, lr=5e-2, ascent=False)  # descent => minimize distance
        if step % 5 == 0:
            print(f"  step {step:4d}: CW = {val:.8f}")

    print(f"\nfinal CW = {cw_distance(p, q):.8f}  (should approach 0)")


if __name__ == "__main__":
    main()
