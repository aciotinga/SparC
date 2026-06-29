"""Load pre-trained PCs saved under the learned_pcs/ directory layout."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from sparc.nodes import CircuitNode

from sparc.io.serializer import CircuitSerializer


def load_learned_pc(
    base_dir: Union[str, Path],
    structure: str,
    dataset: str,
    block_size: int,
    seed: int,
) -> CircuitNode:
    """Load a pre-trained circuit from the standard directory layout.

    Files are expected at::

        {base_dir}/{structure}/{dataset}/{block_size}/
            {structure}_{dataset}_blocksize{block_size}_seed{seed}.json

    Args:
        base_dir: Root directory containing structure subfolders.
        structure: Structure name (e.g. ``"hclt"``, ``"rat_spn"``).
        dataset: Dataset name (e.g. ``"adult"``, ``"plants"``).
        block_size: Block-size subdirectory name.
        seed: Random seed encoded in the filename.

    Returns:
        A :class:`~sparc.nodes.CircuitNode` loaded from the matching JSON file.

    Raises:
        FileNotFoundError: If the expected path does not exist.
    """
    path = (
        Path(base_dir)
        / structure
        / dataset
        / str(block_size)
        / f"{structure}_{dataset}_blocksize{block_size}_seed{seed}.json"
    )
    return CircuitSerializer.load(path)
