"""Hidden Markov model structures as probabilistic circuits.

The circuit encodes a left-to-right latent chain over a sequence of
``seq_length`` observed variables. Each time step has a latent block of
``num_latents`` states; the recurrence

    M_t(z_t) = p(x_t | z_t) * sum_{z_{t+1}} p(z_{t+1} | z_t) M_{t+1}(z_{t+1})

is built backwards from the last step, where the emission ``p(x_t | z_t)`` is an
input block over variable ``t`` and the transition mixture is a sum block. The
root mixes the first step's block under the initial-state prior.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sparc.builders._factory import _NodeFactory
from sparc.nodes import CircuitNode
from sparc.structures._blocks import (
    input_block,
    product_block,
    root_circuit,
    summate,
)
from sparc.structures.distributions import (
    Categorical,
    InputDistribution,
    resolve_input_distribution,
)


def _build_chain(
    factory: _NodeFactory,
    seq_length: int,
    num_latents: int,
    emission: InputDistribution,
    concentration: float,
) -> CircuitNode:
    if seq_length < 1:
        raise ValueError("seq_length must be >= 1")
    if num_latents < 1:
        raise ValueError("num_latents must be >= 1")

    # M for the final step is just its emission block.
    chain = input_block(factory, seq_length - 1, num_latents, emission)
    for t in range(seq_length - 2, -1, -1):
        transition = summate(factory, [chain], num_latents, concentration)
        emit = input_block(factory, t, num_latents, emission)
        chain = product_block(factory, [emit, transition])
    root = summate(factory, [chain], 1, concentration)
    return root_circuit(root)


def HMM(
    seq_length: int,
    num_latents: int,
    num_emits: int,
    *,
    sum_concentration: float = 1.0,
    seed: Optional[int] = None,
) -> CircuitNode:
    """A hidden Markov model with categorical emissions over ``num_emits`` symbols.

    Parameters
    ----------
    seq_length:
        Number of observed variables (the chain length). Variables are indexed
        ``0 .. seq_length - 1``.
    num_latents:
        Number of latent states per time step.
    num_emits:
        Number of emission symbols (categorical cardinality).
    sum_concentration:
        Dirichlet concentration for the randomly initialized transition/prior
        weights.
    seed:
        Optional RNG seed for reproducible parameter initialization.
    """
    if seed is not None:
        np.random.seed(seed)
    factory = _NodeFactory()
    emission = Categorical(num_emits)
    return _build_chain(factory, seq_length, num_latents, emission, sum_concentration)


def GeneralizedHMM(
    seq_length: int,
    num_latents: int,
    *,
    input_dist: Optional[InputDistribution] = None,
    num_cats: int = 256,
    sum_concentration: float = 1.0,
    seed: Optional[int] = None,
) -> CircuitNode:
    """A hidden Markov model with an arbitrary finite-discrete emission family.

    Same chain topology as :func:`HMM`, but emissions are produced by
    ``input_dist`` (any :class:`InputDistribution`); defaults to a categorical
    over ``num_cats`` symbols when not given.
    """
    if seed is not None:
        np.random.seed(seed)
    factory = _NodeFactory()
    emission = resolve_input_distribution(input_dist, num_cats)
    return _build_chain(factory, seq_length, num_latents, emission, sum_concentration)
