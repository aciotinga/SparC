"""Minimize / maximize the GCW cross-term over a learnable circuit2.

The Gromov-Circuit-Wasserstein cross-term is the structure-aware coupling cost
between two circuits. ``gcw_crossterm_and_grad`` returns subgradients w.r.t.
``circuit2``. Pass ``ascent=True`` to maximize, ``ascent=False`` to minimize.

    python examples/gcw_optimization.py --direction min
    python examples/gcw_optimization.py --direction max
"""

import argparse
import random

import numpy as np

from sparc.builders import EmbeddingBuilder
from sparc.optim import apply_grads
from sparc.queries import gcw_crossterm, gcw_crossterm_and_grad


def make_circuit(seed):
    np.random.seed(seed)
    random.seed(seed)
    return EmbeddingBuilder(
        num_vars=100, num_categories=3, sum_arity=2, prod_arity=2,
        sum_concentration=1.0, sum_reuse_probability=0.0,
        prod_reuse_probability=0.0, input_distribution="categorical", alpha=1.0,
    ).build()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("min", "max"), default="max")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    ascent = args.direction == "max"

    circuit1 = make_circuit(0)
    circuit2 = make_circuit(1)

    init = gcw_crossterm(circuit1, circuit2)
    print(f"[{args.direction}] initial GCW cross-term = {init:.8f}")

    for step in range(1, args.steps + 1):
        val, grad2 = gcw_crossterm_and_grad(circuit1, circuit2)
        apply_grads(circuit2, grad2, lr=args.lr, ascent=ascent)
        if step % 5 == 0:
            print(f"  step {step:4d}: GCW = {val:.8f}")

    final = gcw_crossterm(circuit1, circuit2)
    print(f"\nfinal GCW cross-term = {final:.8f}  (change {final - init:+.8f})")


if __name__ == "__main__":
    main()
