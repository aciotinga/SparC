"""Random circuit builders: region-graph embeddings and recursive embeddings."""

from __future__ import annotations

import math
import random
from typing import Dict, Optional

import numpy as np

from sparc.nodes import CircuitNode, SumNode

from sparc.builders._factory import _NodeFactory
from sparc.builders.region_graph import Partition, Region

_gammaln = np.vectorize(math.lgamma)


def _binomial_pmf(n: int):
    """PMF of Binomial(n-1, p) with random p, computed in log-space."""
    p = np.random.uniform(0, 1)
    n_minus_1 = n - 1
    k = np.arange(n)
    log_comb = _gammaln(n_minus_1 + 1) - (_gammaln(k + 1) + _gammaln(n_minus_1 - k + 1))
    log_prob = k * np.log(p) + (n_minus_1 - k) * np.log1p(-p)
    return np.exp(log_comb + log_prob), p


_INPUT_DISTS = ("binomial", "categorical")


class RegionEmbeddingBuilder:
    """Build a block-structured PC from an explicit region graph.

    Each region is realized as ``block_size`` parallel sum nodes; partitions
    become product nodes over aligned sub-region blocks.

    Args:
        region_graph: Root :class:`~sparc.builders.region_graph.Region` with
            partition tree.
        num_categories: Cardinality of categorical (or binomial) leaves.
        block_size: Number of parallel sum nodes per region.
        sum_concentration: Dirichlet concentration for sum-node weights.
        input_distribution: ``"categorical"`` or ``"binomial"``.
        alpha: Dirichlet concentration for categorical leaves (required when
            ``input_distribution="categorical"``).
        scope_offset: Added to every leaf variable index.
    """

    def __init__(
        self,
        region_graph: Region,
        num_categories: int,
        block_size: int,
        sum_concentration: float,
        input_distribution: str,
        alpha: Optional[float] = None,
        scope_offset: int = 0,
    ):
        if input_distribution not in _INPUT_DISTS:
            raise ValueError(f"input_distribution must be one of {_INPUT_DISTS}")
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        if sum_concentration <= 0:
            raise ValueError("Sum concentration must be positive")
        if input_distribution == "categorical":
            if alpha is None or alpha <= 0:
                raise ValueError("alpha must be positive for categorical inputs")

        self.region_graph = region_graph
        self.num_categories = num_categories
        self.block_size = block_size
        self.sum_concentration = sum_concentration
        self.input_distribution = input_distribution
        self.alpha = alpha
        self.scope_offset = scope_offset

    def build(self) -> CircuitNode:
        """Construct and return the circuit.

        Returns:
            Root :class:`~sparc.nodes.CircuitNode` with randomly initialized
            parameters following the region graph structure.
        """
        factory = _NodeFactory()
        region_cache: Dict = {}
        children = []
        for partition in self.region_graph.partitions:
            children.extend(
                self._partition_to_product_nodes(partition, region_cache, factory)
            )
        sum_params = np.random.dirichlet(np.ones(len(children)) * self.sum_concentration)
        return factory.sum(children, sum_params)

    def _partition_to_product_nodes(self, partition: Partition, region_cache, factory):
        sub_regions = [
            self._region_to_sum_nodes(sr, region_cache, factory)
            for sr in partition.sub_regions
        ]
        return [
            factory.product([sub_regions[j][i] for j in range(len(sub_regions))])
            for i in range(self.block_size)
        ]

    def _region_to_sum_nodes(self, region: Region, region_cache, factory):
        if region.scope in region_cache:
            return region_cache[region.scope]
        if len(region.scope) == 1:
            scope_var = list(region.scope)[0] + self.scope_offset
            children = []
            for _ in range(self.block_size):
                if self.input_distribution == "binomial":
                    pmf, _ = _binomial_pmf(self.num_categories)
                else:
                    pmf = np.random.dirichlet(np.ones(self.num_categories) * self.alpha)
                children.append(factory.categorical(scope_var, pmf))
        else:
            children = []
            for partition in region.partitions:
                children.extend(
                    self._partition_to_product_nodes(partition, region_cache, factory)
                )
        sum_nodes = []
        for _ in range(self.block_size):
            sum_params = np.random.dirichlet(np.ones(len(children)) * self.sum_concentration)
            sum_nodes.append(factory.sum(children, sum_params))
        region_cache[region.scope] = sum_nodes
        return sum_nodes


class EmbeddingBuilder:
    """Build a random PC over ``num_vars`` variables with optional node reuse.

    Recursively partitions the scope into ``prod_arity`` sub-scopes at product
    nodes and mixes ``sum_arity`` children at sum nodes. Cached nodes may be
    reused according to the configured probabilities.

    Args:
        num_vars: Number of observed variables.
        num_categories: Leaf cardinality (categorical or binomial support size).
        sum_arity: Number of children per sum node.
        prod_arity: Number of sub-scopes per product partition.
        sum_concentration: Dirichlet concentration for sum-node weights.
        sum_reuse_probability: Probability of reusing a cached sum subtree.
        prod_reuse_probability: Probability of reusing a cached product subtree.
        input_distribution: ``"categorical"`` or ``"binomial"``.
        alpha: Dirichlet concentration for categorical leaves.
        scope_offset: Added to every leaf variable index.
    """

    def __init__(
        self,
        num_vars: int,
        num_categories: int,
        sum_arity: int,
        prod_arity: int,
        sum_concentration: float,
        sum_reuse_probability: float,
        prod_reuse_probability: float,
        input_distribution: str,
        alpha: Optional[float] = None,
        scope_offset: int = 0,
    ):
        if input_distribution not in _INPUT_DISTS:
            raise ValueError(f"input_distribution must be one of {_INPUT_DISTS}")
        if not 0 <= sum_reuse_probability <= 1:
            raise ValueError("sum_reuse_probability must be in [0, 1]")
        if not 0 <= prod_reuse_probability <= 1:
            raise ValueError("prod_reuse_probability must be in [0, 1]")
        if sum_concentration <= 0:
            raise ValueError("Sum concentration must be positive")
        if input_distribution == "categorical":
            if alpha is None or alpha <= 0:
                raise ValueError("alpha must be positive for categorical inputs")

        self.num_vars = num_vars
        self.num_categories = num_categories
        self.sum_arity = sum_arity
        self.prod_arity = prod_arity
        self.sum_concentration = sum_concentration
        self.sum_reuse_probability = sum_reuse_probability
        self.prod_reuse_probability = prod_reuse_probability
        self.input_distribution = input_distribution
        self.alpha = alpha
        self.scope_offset = scope_offset

    def build(self) -> CircuitNode:
        """Construct and return the circuit.

        Returns:
            Root :class:`~sparc.nodes.CircuitNode` with randomly initialized
            parameters over variables
            ``scope_offset .. scope_offset + num_vars - 1``.
        """
        factory = _NodeFactory()
        scope = frozenset(range(self.scope_offset, self.scope_offset + self.num_vars))
        root = self._build(scope, {}, {}, {}, factory)
        return root

    def _leaf_pmf(self):
        if self.input_distribution == "binomial":
            pmf, _ = _binomial_pmf(self.num_categories)
            return pmf
        return np.random.dirichlet(np.ones(self.num_categories) * self.alpha)

    def _build(self, scope, sum_cache, prod_cache, input_cache, factory):
        children = []
        for _ in range(self.sum_arity):
            if len(scope) == 1:
                scope_var = list(scope)[0]
                children.append(factory.categorical(scope_var, self._leaf_pmf()))
            else:
                prod_cache.setdefault(scope, [])
                if random.random() < self.prod_reuse_probability and prod_cache[scope]:
                    children.append(random.choice(prod_cache[scope]))
                    continue
                partition = self.partition_set(scope, self.prod_arity)
                product_children = []
                for child_scope in partition:
                    if len(child_scope) == 0:
                        continue
                    if len(sum_cache.get(child_scope, [])) == 0:
                        sum_cache[child_scope] = []
                        product_child = self._build(
                            child_scope, sum_cache, prod_cache, input_cache, factory
                        )
                        sum_cache[child_scope].append(product_child)
                        product_children.append(product_child)
                    elif random.random() < self.sum_reuse_probability and sum_cache[child_scope]:
                        product_children.append(random.choice(sum_cache[child_scope]))
                    else:
                        product_child = self._build(
                            child_scope, sum_cache, prod_cache, input_cache, factory
                        )
                        sum_cache[child_scope].append(product_child)
                        product_children.append(product_child)
                children.append(factory.product(product_children))

        target_cache = input_cache if len(scope) == 1 else prod_cache
        target_cache.setdefault(scope, [])
        target_cache[scope].extend(children)

        sum_params = np.random.dirichlet(np.ones(len(children)) * self.sum_concentration)
        sum_node = factory.sum(children, sum_params)
        sum_cache.setdefault(scope, [])
        sum_cache[scope].append(sum_node)
        return sum_node

    def partition_set(self, input_set, k):
        elements = list(input_set)
        random.shuffle(elements)
        return [frozenset(elements[i::k]) for i in range(k)]
