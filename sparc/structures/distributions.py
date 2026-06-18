"""Input-distribution specifications for circuit structures.

A structure describes *which kind of leaf* sits over each observed variable
without committing to a concrete node class or its initial parameters. An
:class:`InputDistribution` is a tiny factory that, given a node factory and a
variable index, produces a single leaf node. New leaf families can be plugged
in by adding another subclass here -- the structure code never changes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sparc.builders._factory import _NodeFactory


class InputDistribution:
    """Base spec: build one leaf node over ``scope_var`` via ``factory``."""

    def create(self, factory: _NodeFactory, scope_var: int):
        raise NotImplementedError(
            f"{type(self).__name__} must implement create()"
        )


class Categorical(InputDistribution):
    """Categorical leaf over ``num_cats`` outcomes, randomly initialized.

    Parameters are drawn from a symmetric Dirichlet with concentration
    ``alpha``.
    """

    def __init__(self, num_cats: int, alpha: float = 1.0):
        if num_cats < 2:
            raise ValueError("Categorical requires num_cats >= 2")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.num_cats = int(num_cats)
        self.alpha = float(alpha)

    def create(self, factory: _NodeFactory, scope_var: int):
        pmf = np.random.dirichlet(np.ones(self.num_cats) * self.alpha)
        return factory.categorical(scope_var, pmf)


class Bernoulli(InputDistribution):
    """Binary leaf. ``p`` is fixed if given, otherwise drawn from U(0, 1)."""

    def __init__(self, p: Optional[float] = None):
        if p is not None and not (0.0 <= p <= 1.0):
            raise ValueError("p must lie in [0, 1]")
        self.p = p

    def create(self, factory: _NodeFactory, scope_var: int):
        p = self.p if self.p is not None else float(np.random.uniform(0.0, 1.0))
        return factory.bernoulli(scope_var, p)


class Indicator(InputDistribution):
    """Deterministic leaf over ``num_cats`` states.

    ``value`` is fixed if given, otherwise a random state is chosen per leaf.
    """

    def __init__(self, num_cats: int, value: Optional[int] = None):
        if num_cats < 2:
            raise ValueError("Indicator requires num_cats >= 2")
        if value is not None and not (0 <= value < num_cats):
            raise ValueError("value must lie in [0, num_cats)")
        self.num_cats = int(num_cats)
        self.value = value

    def create(self, factory: _NodeFactory, scope_var: int):
        value = (
            self.value
            if self.value is not None
            else int(np.random.randint(self.num_cats))
        )
        return factory.indicator(scope_var, value, self.num_cats)


class Literal(InputDistribution):
    """Deterministic boolean leaf. ``value`` is fixed if given, else random."""

    def __init__(self, value: Optional[int] = None):
        if value is not None and value not in (0, 1):
            raise ValueError("value must be 0 or 1")
        self.value = value

    def create(self, factory: _NodeFactory, scope_var: int):
        value = self.value if self.value is not None else int(np.random.randint(2))
        return factory.literal(scope_var, value)


class DiscreteLogistic(InputDistribution):
    """Logistic shape discretized over ``num_cats`` integer bins.

    ``mu`` (location) and ``s`` (scale) default to random values when not given.
    """

    def __init__(
        self,
        num_cats: int,
        mu: Optional[float] = None,
        s: Optional[float] = None,
    ):
        if num_cats < 2:
            raise ValueError("DiscreteLogistic requires num_cats >= 2")
        if s is not None and s <= 0:
            raise ValueError("s must be positive")
        self.num_cats = int(num_cats)
        self.mu = mu
        self.s = s

    def create(self, factory: _NodeFactory, scope_var: int):
        mu = (
            self.mu
            if self.mu is not None
            else float(np.random.uniform(0.0, self.num_cats - 1))
        )
        s = self.s if self.s is not None else float(np.random.uniform(0.5, 2.0))
        return factory.discrete_logistic(scope_var, mu, s, self.num_cats)


def resolve_input_distribution(
    input_dist: Optional[InputDistribution], num_cats: int
) -> InputDistribution:
    """Return ``input_dist`` or a default :class:`Categorical` over ``num_cats``."""
    if input_dist is None:
        return Categorical(num_cats)
    if not isinstance(input_dist, InputDistribution):
        raise TypeError(
            "input_dist must be an InputDistribution instance, got "
            f"{type(input_dist)!r}"
        )
    return input_dist
