import random

from sparc.builders import RandomRegionGraph, Region


def test_leaf_region_has_no_partitions():
    random.seed(0)
    graph = RandomRegionGraph(
        frozenset({0, 1, 2}),
        partitions_per_region=2,
        sub_regions_per_partition=2,
    )
    leaf = graph.generate(frozenset({1}))
    assert isinstance(leaf, Region)
    assert leaf.scope == frozenset({1})
    assert leaf.partitions == []


def test_internal_region_has_partitions():
    random.seed(0)
    graph = RandomRegionGraph(
        frozenset({0, 1, 2, 3}),
        partitions_per_region=2,
        sub_regions_per_partition=2,
    )
    root = graph.generate(frozenset({0, 1, 2, 3}))
    assert len(root.scope) == 4
    assert len(root.partitions) == 2
    for partition in root.partitions:
        assert len(partition.sub_regions) >= 1
        for sub in partition.sub_regions:
            assert sub.scope <= root.scope


def test_generate_is_memoized():
    random.seed(1)
    graph = RandomRegionGraph(
        frozenset({0, 1, 2}),
        partitions_per_region=1,
        sub_regions_per_partition=2,
    )
    root = graph.generate(frozenset({0, 1, 2}))
    assert frozenset({0, 1, 2}) in graph.region_cache
    again = graph.generate(frozenset({0, 1, 2}))
    assert again is root
