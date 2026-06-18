import math
import random

import numpy as np
import pytest

from sparc.builders import (
    EmbeddingBuilder,
    RandomRegionGraph,
    RegionEmbeddingBuilder,
)


def _full_assignment(scope_vars, num_categories: int) -> dict[int, int]:
    return {v: 0 for v in scope_vars}


def test_region_embedding_builder_categorical():
    np.random.seed(0)
    random.seed(0)
    num_vars = 4
    block_size = 2
    region_graph = RandomRegionGraph(
        frozenset(range(num_vars)),
        partitions_per_region=2,
        sub_regions_per_partition=block_size,
    ).generate(frozenset(range(num_vars)))

    circuit = RegionEmbeddingBuilder(
        region_graph,
        num_categories=5,
        block_size=block_size,
        sum_concentration=1.0,
        input_distribution="categorical",
        alpha=1.0,
    ).build()

    scope = circuit.root.scope_as_list()
    assert len(scope) == num_vars
    assignment = _full_assignment(scope, 5)
    ll = circuit.likelihood(assignment)
    log_ll = circuit.log_likelihood(assignment)
    assert ll > 0.0
    assert math.isfinite(log_ll)
    assert log_ll == pytest.approx(math.log(ll))


def test_region_embedding_builder_binomial():
    np.random.seed(1)
    random.seed(1)
    region_graph = RandomRegionGraph(
        frozenset({0, 1}),
        partitions_per_region=1,
        sub_regions_per_partition=2,
    ).generate(frozenset({0, 1}))

    circuit = RegionEmbeddingBuilder(
        region_graph,
        num_categories=4,
        block_size=2,
        sum_concentration=1.0,
        input_distribution="binomial",
    ).build()

    scope = circuit.root.scope_as_list()
    assignment = _full_assignment(scope, 4)
    assert circuit.likelihood(assignment) > 0.0


def test_region_embedding_builder_scope_offset():
    np.random.seed(2)
    random.seed(2)
    region_graph = RandomRegionGraph(
        frozenset({0, 1}),
        partitions_per_region=1,
        sub_regions_per_partition=2,
    ).generate(frozenset({0, 1}))

    offset = 10
    circuit = RegionEmbeddingBuilder(
        region_graph,
        num_categories=3,
        block_size=2,
        sum_concentration=1.0,
        input_distribution="categorical",
        alpha=1.0,
        scope_offset=offset,
    ).build()

    assert set(circuit.root.scope_as_list()) == {offset, offset + 1}


def test_embedding_builder_categorical():
    np.random.seed(3)
    random.seed(3)
    num_vars = 5
    circuit = EmbeddingBuilder(
        num_vars=num_vars,
        num_categories=4,
        sum_arity=2,
        prod_arity=2,
        sum_concentration=1.0,
        sum_reuse_probability=0.0,
        prod_reuse_probability=0.0,
        input_distribution="categorical",
        alpha=1.0,
    ).build()

    scope = set(circuit.root.scope_as_list())
    assert scope == set(range(num_vars))
    assignment = _full_assignment(scope, 4)
    assert circuit.likelihood(assignment) > 0.0


def test_embedding_builder_with_reuse():
    np.random.seed(4)
    random.seed(4)
    circuit = EmbeddingBuilder(
        num_vars=6,
        num_categories=3,
        sum_arity=2,
        prod_arity=2,
        sum_concentration=1.0,
        sum_reuse_probability=0.5,
        prod_reuse_probability=0.5,
        input_distribution="categorical",
        alpha=1.0,
    ).build()

    assignment = _full_assignment(circuit.root.scope_as_list(), 3)
    assert math.isfinite(circuit.log_likelihood(assignment))
