"""Train a hidden tree-structured PC on MNIST, sample, and plot digits.

Mirrors the basic pipeline of training a data-driven tree PC on MNIST: load
data, build an :func:`~sparc.structures.HCLT`, fit by projected gradient ascent
on mean log-likelihood, draw samples, and visualize them with matplotlib.

SparC expands each latent block into scalar nodes on CPU, so the defaults use
downsampled (14x14) binarized MNIST and modest ``num_latents``. Increase
``--image-size``, ``--num-latents``, or ``--train-size`` if you have time and
memory (full 28x28 with large latents is impractical on CPU).

Requires ``matplotlib`` for plotting (not a SparC dependency). MNIST is loaded
via ``torchvision`` when available, otherwise downloaded from the official IDX
files into ``--data-dir``.

    python examples/train_mnist.py
    python examples/train_mnist.py --epochs 50 --num-latents 12 --save samples.png
"""

from __future__ import annotations

import argparse
import gzip
import math
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from sparc.optim import MLETrainer
from sparc.structures import Bernoulli, HCLT


_MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = {
    "train": "train-images-idx3-ubyte.gz",
    "valid": "t10k-images-idx3-ubyte.gz",
}


def _download_mnist_gz(data_dir: Path, name: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = _MNIST_FILES[name]
    path = data_dir / filename
    if not path.exists():
        url = _MNIST_BASE + filename
        print(f"  downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected MNIST magic {magic} in {path}")
        buf = f.read()
    images = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return images


def _load_mnist_idx(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    train = _read_idx_images_gz(_download_mnist_gz(data_dir, "train"))
    valid = _read_idx_images_gz(_download_mnist_gz(data_dir, "valid"))
    return train, valid


def load_mnist(data_dir: Path, image_size: int):
    """Return binarized train/valid arrays of shape (N, image_size**2)."""
    try:
        import torchvision

        train_ds = torchvision.datasets.MNIST(
            root=str(data_dir), train=True, download=True
        )
        valid_ds = torchvision.datasets.MNIST(
            root=str(data_dir), train=False, download=True
        )
        train = train_ds.data.numpy()
        valid = valid_ds.data.numpy()
    except ImportError:
        train, valid = _load_mnist_idx(data_dir)
    train = _downsample_and_binarize(train.astype(np.float32), image_size)
    valid = _downsample_and_binarize(valid.astype(np.float32), image_size)
    return train, valid


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "This example requires matplotlib for plotting. "
            "Install it with: pip install matplotlib"
        ) from exc
    return plt


def _downsample_and_binarize(images: np.ndarray, image_size: int) -> np.ndarray:
    """28x28 uint8 images -> (N, image_size**2) binary {0, 1}."""
    if image_size > 28 or 28 % image_size != 0:
        raise ValueError("image_size must divide 28 and be at most 28")
    factor = 28 // image_size
    n = images.shape[0]
    imgs = images.reshape(n, 28, 28)
    if factor > 1:
        imgs = imgs.reshape(n, image_size, factor, image_size, factor).mean(axis=(2, 4))
    flat = imgs.reshape(n, image_size * image_size)
    return (flat >= 128.0).astype(np.int32)


def rows_to_dataset(rows: np.ndarray) -> list[dict[int, int]]:
    return [dict(enumerate(row.tolist())) for row in rows]


def mean_log_likelihood(circuit, rows: np.ndarray) -> float:
    """Batched mean log-likelihood over integer rows (N, D)."""
    return float(circuit.batched_log_likelihood(rows).mean())


def plot_samples(
    samples: list[dict[int, int]],
    image_size: int,
    *,
    save_path: Path | None,
    show: bool,
) -> None:
    plt = _import_matplotlib()
    n = len(samples)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, sample in zip(axes, samples):
        img = np.array([sample[v] for v in range(image_size * image_size)], dtype=np.float32)
        img = img.reshape(image_size, image_size)
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    for ax in axes[len(samples) :]:
        ax.axis("off")
    fig.suptitle("Samples from trained PC", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for MNIST download (default: data/)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=14,
        help="Downsampled side length; 14 -> 196 pixels (default: 14)",
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        default=8,
        help="Latent block width per tree variable (default: 8)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=32,
        help="Bins for mutual-information tree estimation (default: 32)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=8000,
        help="Training images used for structure + MLE (default: 8000)",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=2000,
        help="Validation images for reporting LL (default: 2000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.15,
        help="Learning rate for projected gradient ascent (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for parameter init (default: 0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of images to draw and plot (default: 16)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the sample grid PNG",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)

    print("Loading MNIST...")
    train_all, valid_all = load_mnist(args.data_dir, args.image_size)
    train_rows = train_all[: args.train_size]
    valid_rows = valid_all[: args.valid_size]
    n_vars = args.image_size * args.image_size
    print(
        f"  train {train_rows.shape[0]} x {n_vars} (binarized), "
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

    print(f"Drawing {args.num_samples} samples...")
    samples = circuit.sample(args.num_samples, seed=args.seed + 1)
    plot_samples(
        samples,
        args.image_size,
        save_path=args.save,
        show=args.show,
    )
    if args.save is None and not args.show:
        print("Pass --save path.png and/or --show to view the sample grid.")


if __name__ == "__main__":
    main(sys.argv[1:])
