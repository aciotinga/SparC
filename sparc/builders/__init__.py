"""Circuit structure builders."""

from sparc.builders.embedding import EmbeddingBuilder, RegionEmbeddingBuilder
from sparc.builders.region_graph import Partition, RandomRegionGraph, Region

__all__ = [
    "EmbeddingBuilder",
    "RegionEmbeddingBuilder",
    "RandomRegionGraph",
    "Region",
    "Partition",
]


def __getattr__(name):
    # PyjuiceBuilder pulled in lazily; importing it does not require pyjuice itself.
    if name == "PyjuiceBuilder":
        from sparc.builders.pyjuice import PyjuiceBuilder

        return PyjuiceBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
