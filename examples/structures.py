"""Build and use the built-in circuit structures.

Demonstrates constructing each structure, sampling, batched log-likelihood,
maximum-likelihood training of a sequence model, and a Circuit-Wasserstein
distance between two circuits of the same (structurally decomposable) structure.

    python examples/structures.py
"""

import numpy as np

from sparc import cw_distance
from sparc.optim import MLETrainer
from sparc.structures import (
    HCLT,
    HMM,
    PD,
    PDHCLT,
    RAT_SPN,
    Bernoulli,
    GeneralizedHMM,
)


def main():
    np.random.seed(0)

    # A hidden Markov model: sample, score, and train.
    hmm = HMM(seq_length=8, num_latents=4, num_emits=3, seed=0)
    data = hmm.sample(400, seed=1)
    print("HMM mean LL of its own samples:",
          float(hmm.compile().log_likelihood(data).mean()))

    batch = data[:, :8]
    print("HMM batched LL shape:", hmm.compile().log_likelihood(batch).shape)

    model = HMM(seq_length=8, num_latents=4, num_emits=3, seed=99)
    history = MLETrainer(model, lr=0.3).fit(data, epochs=40)
    print(f"HMM training: start LL {history[0]:.4f} -> end LL {history[-1]:.4f}")

    # Sequence model with a custom (bernoulli) emission family.
    ghmm = GeneralizedHMM(seq_length=6, num_latents=3, input_dist=Bernoulli(), seed=0)
    print("GeneralizedHMM scope:", ghmm.scope_as_list())

    # Randomized tensorized SPN.
    rat = RAT_SPN(num_vars=8, num_latents=3, depth=2, num_repetitions=2,
                  num_cats=4, seed=1)
    print("RAT-SPN scope size:", len(rat.scope_as_list()))

    # Two hidden tree-structured circuits over the same data share a topology
    # (the tree is data-derived), so they are Circuit-Wasserstein compatible.
    rng = np.random.RandomState(0)
    tree_data = rng.randint(0, 4, size=(300, 8))
    hclt_a = HCLT(tree_data, num_latents=4, num_bins=8, num_cats=4, seed=0)
    hclt_b = HCLT(tree_data, num_latents=4, num_bins=8, num_cats=4, seed=1)
    print("CW between two HCLTs on the same data:", cw_distance(hclt_a, hclt_b))

    # Grid decompositions over a small 2-D image.
    pd = PD(data_shape=(3, 4), num_latents=3, num_cats=4, seed=0)
    print("PD scope size:", len(pd.scope_as_list()))

    img_data = rng.randint(0, 4, size=(200, 12))
    pdhclt = PDHCLT(img_data, data_shape=(3, 4), num_latents=2, num_cats=4,
                    num_bins=8, max_split_depth=1, seed=0)
    print("PDHCLT scope size:", len(pdhclt.scope_as_list()))


if __name__ == "__main__":
    main()
