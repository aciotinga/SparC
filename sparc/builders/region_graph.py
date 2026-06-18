"""Random region graphs for building hierarchical probabilistic circuits."""

from __future__ import annotations

import random
from typing import FrozenSet, Iterable, Tuple


def _region_scope_key(region: "Region") -> Tuple[int, ...]:
    return tuple(sorted(region.scope))


def _scope_subset_key(subset: FrozenSet[int]) -> Tuple[int, ...]:
    return tuple(sorted(subset))


class Region:
    """A region in a hierarchical region graph.

    A region covers a set of variable indices and is decomposed into one or
    more :class:`Partition` objects when it contains more than one variable.

    Args:
        scope: Iterable of variable indices covered by this region.
    """

    def __init__(self, scope):
        self.scope = frozenset(scope)
        self.partitions = []

    def __repr__(self):
        return f"Region(scope={sorted(self.scope)})"


class Partition:
    """One admissible decomposition of a region into sub-regions.

    Args:
        sub_regions: Child regions whose scopes partition the parent scope.
    """

    def __init__(self, sub_regions: Iterable["Region"]):
        # Deterministic order across runs for a fixed RNG seed.
        self.sub_regions: Tuple[Region, ...] = tuple(
            sorted(sub_regions, key=_region_scope_key)
        )

    def __repr__(self):
        return f"Partition(sub_regions={list(self.sub_regions)})"


class RandomRegionGraph:
    """Generate a random hierarchical region graph over a variable scope.

    Recursively partitions each multi-variable region into ``partitions_per_region``
    random partitions, each with ``sub_regions_per_partition`` sub-regions,
    until regions contain a single variable.

    Args:
        starting_scope: Full set of variable indices at the root.
        partitions_per_region: Number of alternative partitions per region.
        sub_regions_per_partition: Branching factor of each partition.
    """

    def __init__(
        self,
        starting_scope: FrozenSet[int],
        partitions_per_region: int,
        sub_regions_per_partition: int,
    ):
        if len(starting_scope) == 0:
            raise ValueError("Starting scope must be non-empty")
        if partitions_per_region <= 0:
            raise ValueError("Partitions per region must be positive")
        if sub_regions_per_partition <= 0:
            raise ValueError("Sub regions per partition must be positive")
        self.starting_scope = starting_scope
        self.partitions_per_region = partitions_per_region
        self.sub_regions_per_partition = sub_regions_per_partition
        self.region_cache = {}

    def generate(self, scope: FrozenSet[int]) -> Region:
        """Build (or retrieve cached) the region graph rooted at ``scope``.

        Args:
            scope: Variable indices for the root region.

        Returns:
            A :class:`Region` whose ``partitions`` tree covers ``scope``.
        """
        if scope in self.region_cache:
            return self.region_cache[scope]
        root = Region(scope)
        if len(scope) == 1:
            return root
        partitions = []
        for _ in range(self.partitions_per_region):
            sub_region_partitions = self._balanced_random_partition(
                scope, self.sub_regions_per_partition
            )
            sub_regions = [self.generate(p) for p in sub_region_partitions]
            partitions.append(Partition(sub_regions))
        root.partitions = partitions
        self.region_cache[scope] = root
        return root

    def _balanced_random_partition(
        self, input_set: FrozenSet[int], k: int
    ) -> Tuple[FrozenSet[int], ...]:
        if k <= 0:
            raise ValueError("k must be a positive integer greater than 0.")
        items = list(input_set)
        random.shuffle(items)
        subsets = [frozenset(items[i::k]) for i in range(k)]
        non_empty = [s for s in subsets if s]
        return tuple(sorted(non_empty, key=_scope_subset_key))
