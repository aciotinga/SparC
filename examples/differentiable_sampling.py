"""Optimize a circuit through hard samples with the SIMPLE estimator."""

from __future__ import annotations

import numpy as np

from sparc import CategoricalInputNode, SumNode
from sparc.optim import apply_grads


def main() -> None:
    left = CategoricalInputNode(0, [0.8, 0.2])
    right = CategoricalInputNode(0, [0.3, 0.7])
    circuit = SumNode([left, right], [0.5, 0.5])
    target = np.array([0.0, 1.0])
    batch_size = 256

    for step in range(20):
        draws = circuit.sample(
            batch_size,
            seed=step,
            differentiable=True,
        )
        residual = draws.one_hot - target
        loss = float(np.mean(np.sum(residual * residual, axis=1)))

        # Gradient of mean squared error with respect to the packed one-hot
        # samples. vjp() sums exactly what it receives, so include 1 / batch.
        upstream = 2.0 * residual / batch_size
        grads = draws.vjp(upstream)
        apply_grads(circuit, grads, lr=0.05, ascent=False)

        if step % 5 == 0:
            print(
                f"step={step:02d} loss={loss:.3f} "
                f"mixture={circuit.parameters_list()}"
            )


if __name__ == "__main__":
    main()
