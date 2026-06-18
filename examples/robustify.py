"""Robustify a saved PC via distributionally-robust optimization (DRO).

Loads a ``gcw-circuit-v1`` JSON from ``examples/example_pcs/`` and solves

    max_theta  min_{Q : CW(P_hat, Q) <= k}  log E_Q[P_theta]

via a Lagrangian inner loop on Q (dual variable lambda for the CW-ball
constraint) and projected gradient ascent on theta.

    python examples/robustify.py plants.json
    python examples/robustify.py example_pcs/adult.json --iters 50 --output robust_plants.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sparc.circuit import Circuit
from sparc.optim import apply_grads, global_grad_norm
from sparc.queries import (
    cw_distance,
    cw_distance_and_grad,
    log_exp_query,
    log_exp_query_and_grad,
)

_EXAMPLE_PCS = Path(__file__).resolve().parent / "example_pcs"

K = 2.0
NUM_Q_ITERS = 50
ETA_THETA = 1e-2
ETA_PHI = 1e-2
ETA_LAMBDA = 10.0
LAMBDA_MAX = 1000.0


def resolve_circuit_path(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path.resolve()
    candidate = _EXAMPLE_PCS / name
    if candidate.is_file():
        return candidate.resolve()
    if not name.endswith(".json"):
        candidate = _EXAMPLE_PCS / f"{name}.json"
        if candidate.is_file():
            return candidate.resolve()
    choices = sorted(p.name for p in _EXAMPLE_PCS.glob("*.json"))
    raise FileNotFoundError(
        f"Circuit not found: {name!r}. "
        f"Pass a path or a basename under example_pcs/ ({', '.join(choices)})."
    )


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


def update_q(p_theta, q_phi, p_hat, lam, k):
    for _ in range(NUM_Q_ITERS):
        _, _, grad_phi = log_exp_query_and_grad(p_theta, q_phi)
        cw_val, cw_grads = cw_distance_and_grad(p_hat, q_phi)
        lam = min(LAMBDA_MAX, max(0.0, lam + ETA_LAMBDA * (cw_val - k)))
        phi_grads = combine_phi_grads(grad_phi, cw_grads, lam)
        apply_grads(q_phi, phi_grads, ETA_PHI, ascent=False)
    return lam


def run_dro(p_hat, *, k=K, num_iters=20, eta_theta=ETA_THETA):
    p_theta = p_hat.clone()
    q_phi = p_hat.clone()
    lam = 0.0

    print(f"initial: log(E)={log_exp_query(p_theta, q_phi):.6f}  "
          f"CW={cw_distance(p_hat, q_phi):.6f}")

    for it in range(1, num_iters + 1):
        lam = update_q(p_theta, q_phi, p_hat, lam, k)
        _, grad_theta, _ = log_exp_query_and_grad(p_theta, q_phi)
        apply_grads(p_theta, grad_theta, eta_theta, ascent=True)

        log_e = log_exp_query(p_theta, q_phi)
        cw = cw_distance(p_hat, q_phi)
        print(f"  iter {it:3d}: log(E)={log_e:.6f}  CW={cw:.6f}  "
              f"violation={cw - k:+.6f}  lambda={lam:.3f}")

    return p_theta, lam


def main():
    parser = argparse.ArgumentParser(description="Robustify a PC with DRO.")
    parser.add_argument(
        "circuit",
        help="Basename or path to a gcw-circuit-v1 JSON (e.g. plants.json or adult)",
    )
    parser.add_argument("--k", type=float, default=K, help="CW-ball radius")
    parser.add_argument("--iters", type=int, default=20, help="outer theta steps")
    parser.add_argument(
        "--output", "-o",
        help="Optional path to save the robustified circuit (gcw-circuit-v1 JSON)",
    )
    args = parser.parse_args()

    path = resolve_circuit_path(args.circuit)
    print(f"loading {path.name} from {path.parent}")
    p_hat = Circuit.load(path)
    print(f"  nodes in scope: {len(p_hat.root.scope_as_list())}")

    p_theta, lam = run_dro(p_hat, k=args.k, num_iters=args.iters)

    if args.output:
        out = Path(args.output)
        p_theta.save(out)
        print(f"\nsaved robustified circuit to {out.resolve()}")

    print(f"\nfinal lambda={lam:.3f}")


if __name__ == "__main__":
    main()
