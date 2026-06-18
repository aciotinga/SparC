"""Distributionally-robust optimization (DRO) saddle point over circuits.

Solves   max_theta  min_{Q : CW(P_hat, Q) <= k}  log E_Q[P_theta]
via a Lagrangian inner loop on Q (and a dual variable lambda for the CW-ball
constraint) and projected gradient ascent on the outer player theta.

This is a self-contained recipe on synthetic circuits using only the built-in
solvers (no Gurobi, no file IO, no plotting).

    python examples/dro.py
"""

import random

import numpy as np

from sparc.builders import RandomRegionGraph, RegionEmbeddingBuilder
from sparc.optim import apply_grads, global_grad_norm
from sparc.queries import (
    cw_distance,
    cw_distance_and_grad,
    log_exp_query,
    log_exp_query_and_grad,
)

K = 2.0            # Wasserstein-ball radius
NUM_Q_ITERS = 5    # inner phi/lambda iterations per outer step
ETA_THETA = 1e-1
ETA_PHI = 1e-1
ETA_LAMBDA = 10.0
LAMBDA_MAX = 1000.0


def combine_phi_grads(logexp_grads, cw_grads, lam):
    """Normalized convex combination of the two phi descent directions."""
    n_e = global_grad_norm(logexp_grads) or 1.0
    n_c = global_grad_norm(cw_grads) or 1.0
    w = lam / (1.0 + lam)
    c_e = (1.0 - w) / n_e
    c_c = w / n_c

    sum_g, cat_g = {}, {}
    for out, le_d, cw_d in (
        (sum_g, logexp_grads.sum_grads, cw_grads.sum_grads),
        (cat_g, logexp_grads.cat_grads, cw_grads.cat_grads),
    ):
        for nid in set(le_d) | set(cw_d):
            le = np.asarray(le_d.get(nid, 0.0), dtype=np.float64)
            cw = np.asarray(cw_d.get(nid, 0.0), dtype=np.float64)
            out[nid] = c_e * le + c_c * cw
    return (sum_g, cat_g)


def update_q(p_theta, q_phi, p_hat, lam):
    for _ in range(NUM_Q_ITERS):
        _, _, grad_phi = log_exp_query_and_grad(p_theta, q_phi)
        cw_val, cw_grads = cw_distance_and_grad(p_hat, q_phi)
        lam = min(LAMBDA_MAX, max(0.0, lam + ETA_LAMBDA * (cw_val - K)))
        phi_grads = combine_phi_grads(grad_phi, cw_grads, lam)
        apply_grads(q_phi, phi_grads, ETA_PHI, ascent=False)
    return lam


def main():
    np.random.seed(0)
    random.seed(0)

    num_vars = 500

    rg = RandomRegionGraph(
        frozenset(range(num_vars)), partitions_per_region=1, sub_regions_per_partition=2
    )
    root_region = rg.generate(frozenset(range(num_vars)))
    p_hat = RegionEmbeddingBuilder(
        root_region, num_categories=2, block_size=4,
        sum_concentration=1.0, input_distribution="categorical", alpha=1.0,
    ).build()
    p_theta = p_hat.clone()
    q_phi = p_hat.clone()
    lam = 0.0

    print(f"initial: log(E)={log_exp_query(p_theta, q_phi):.6f}  "
          f"CW={cw_distance(p_hat, q_phi):.6f}")

    for it in range(1, 21):
        lam = update_q(p_theta, q_phi, p_hat, lam)
        _, grad_theta, _ = log_exp_query_and_grad(p_theta, q_phi)
        apply_grads(p_theta, grad_theta, ETA_THETA, ascent=True)

        log_e = log_exp_query(p_theta, q_phi)
        cw = cw_distance(p_hat, q_phi)
        print(f"  iter {it:3d}: log(E)={log_e:.6f}  CW={cw:.6f}  "
              f"violation={cw - K:+.6f}  lambda={lam:.3f}")


if __name__ == "__main__":
    main()
