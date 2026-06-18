"""Load pre-trained PCs saved under the learned_pcs/ directory layout."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from sparc.circuit import Circuit

from sparc.io.serializer import CircuitSerializer


def load_learned_pc(
    base_dir: Union[str, Path],
    structure: str,
    dataset: str,
    block_size: int,
    seed: int,
) -> Circuit:
    """Load a learned PC from the standard directory and filename convention."""
    path = (
        Path(base_dir)
        / structure
        / dataset
        / str(block_size)
        / f"{structure}_{dataset}_blocksize{block_size}_seed{seed}.json"
    )
    return Circuit(CircuitSerializer.load(path))
