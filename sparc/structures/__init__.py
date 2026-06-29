"""Built-in probabilistic-circuit structures.

This subpackage provides ready-made circuit constructors that return a
:class:`~sparc.nodes.CircuitNode` root:

- :func:`PD` / :func:`PDHCLT` -- recursive grid decompositions.
- :func:`HCLT` -- hidden tree-structured circuits derived from data.
- :func:`HMM` / :func:`GeneralizedHMM` -- latent-chain sequence models.
- :func:`RAT_SPN` -- randomized tensorized sum-product circuits.

Structures are assembled from a small block-construction algebra
(:mod:`sparc.structures._blocks`) over pluggable leaf distributions
(:mod:`sparc.structures.distributions`), so new structures and leaf families can
be added without touching the circuit core.
"""

from sparc.structures.distributions import (
    Bernoulli,
    Categorical,
    DiscreteLogistic,
    Indicator,
    InputDistribution,
    Literal,
)
from sparc.structures.hclt import HCLT
from sparc.structures.hmm import HMM, GeneralizedHMM
from sparc.structures.pd import PD, PDHCLT
from sparc.structures.rat_spn import RAT_SPN

__all__ = [
    "PD",
    "PDHCLT",
    "HCLT",
    "HMM",
    "GeneralizedHMM",
    "RAT_SPN",
    "InputDistribution",
    "Categorical",
    "Bernoulli",
    "Indicator",
    "Literal",
    "DiscreteLogistic",
]
