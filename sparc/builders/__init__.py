"""Circuit structure builders.

Two complementary construction strategies are provided:

- :class:`~sparc.builders.embedding.EmbeddingBuilder` recursively partitions a
  fixed variable scope with optional node reuse (good for random benchmarks).
- :class:`~sparc.builders.embedding.RegionEmbeddingBuilder` builds a
  block-structured PC from an explicit :class:`~sparc.builders.region_graph.Region`
  hierarchy produced by :class:`~sparc.builders.region_graph.RandomRegionGraph`.
"""

from sparc.builders.embedding import EmbeddingBuilder, RegionEmbeddingBuilder
from sparc.builders.region_graph import Partition, RandomRegionGraph, Region

__all__ = [
    "EmbeddingBuilder",
    "RegionEmbeddingBuilder",
    "RandomRegionGraph",
    "Region",
    "Partition",
]
