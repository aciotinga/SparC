"""Maximum-likelihood training of a probabilistic circuit.

Samples a dataset from a ground-truth circuit, then fits a fresh circuit of the
same structure by projected gradient ascent on the mean log-likelihood.

    python examples/mle.py
"""

import random

import numpy as np

from sparc.builders import EmbeddingBuilder
from sparc.optim import MLETrainer


def make_circuit():
    return EmbeddingBuilder(
        num_vars=8,
        num_categories=3,
        sum_arity=2,
        prod_arity=2,
        sum_concentration=1.0,
        sum_reuse_probability=0.0,
        prod_reuse_probability=0.0,
        input_distribution="categorical",
        alpha=1.0,
    ).build()


def main():
    np.random.seed(0)
    random.seed(0)

    truth = make_circuit()
    data = truth.sample(1000, seed=42)

    model = make_circuit()
    trainer = MLETrainer(model, lr=0.5, method="tangent")

    def report(epoch, mean_ll):
        if epoch % 10 == 0:
            print(f"  epoch {epoch:4d}: mean LL = {mean_ll:.6f}")

    history = trainer.fit(data, epochs=80, callback=report)
    print(f"\nstart LL = {history[0]:.6f}  ->  end LL = {history[-1]:.6f}")
    print(f"truth LL = {np.mean([truth.log_likelihood(x) for x in data]):.6f} (ceiling)")


if __name__ == "__main__":
    main()
