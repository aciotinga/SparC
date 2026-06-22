"""Maximum-likelihood training of a hidden tree-structured PC on Plants.

Loads the binary Plants dataset from ``original_datasets/``, builds an
:data-driven :func:`~sparc.structures.HCLT` dependency tree, and fits
parameters by projected gradient ascent on mean log-likelihood.

    python examples/mle.py
    python examples/mle.py --epochs 50 --num-latents 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from sparc.optim import MLETrainer
from sparc.structures import Bernoulli, HCLT

_EXAMPLES = Path(__file__).resolve().parent
_PLANTS_DIR = _EXAMPLES / "original_datasets" / "plants"


def load_csv_data(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def rows_to_dataset(rows: np.ndarray) -> list[dict[int, int]]:
    return [dict(enumerate(row.tolist())) for row in rows]


def mean_log_likelihood(circuit, rows: np.ndarray) -> float:
    return float(circuit.compile().log_likelihood(rows).mean())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for projected gradient ascent (default: 0.1)",
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        default=4,
        help="Latent block width per tree variable (default: 4)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=2,
        help="Bins for mutual-information tree estimation (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for parameter init (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)

    train_path = _PLANTS_DIR / "plants.train.data"
    valid_path = _PLANTS_DIR / "plants.valid.data"
    for path in (train_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(f"Plants dataset file not found: {path}")

    print("Loading Plants...")
    train_rows = load_csv_data(train_path)
    valid_rows = load_csv_data(valid_path)
    n_vars = train_rows.shape[1]
    print(
        f"  train {train_rows.shape[0]} x {n_vars}, "
        f"valid {valid_rows.shape[0]}"
    )

    print("Building HCLT (dependency tree from training data)...")
    t0 = time.time()
    circuit = HCLT(
        train_rows.astype(np.float64),
        num_latents=args.num_latents,
        num_bins=args.num_bins,
        input_dist=Bernoulli(),
        seed=args.seed,
    )
    print(f"  built in {time.time() - t0:.1f}s")

    train_data = rows_to_dataset(train_rows)
    trainer = MLETrainer(circuit, lr=args.lr, method="tangent")

    print(f"Training for {args.epochs} epochs (lr={args.lr})...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_ll = trainer.step(train_data)
        train_time = time.time() - t0
        t1 = time.time()
        valid_ll = mean_log_likelihood(circuit, valid_rows)
        valid_time = time.time() - t1
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(
                f"  epoch {epoch:4d}/{args.epochs}  "
                f"train LL {train_ll:.2f}  val LL {valid_ll:.2f}  "
                f"({train_time:.1f}s train, {valid_time:.1f}s val)"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
