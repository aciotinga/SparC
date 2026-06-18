# Structures

The [`sparc.structures`][sparc.structures] subpackage provides ready-made
circuit constructors. Each returns a [`Circuit`][sparc.circuit.Circuit].

## Built-in structures

| Constructor | Description |
|-------------|-------------|
| [`HMM`][sparc.structures.hmm.HMM] | Hidden Markov model with categorical emissions |
| [`GeneralizedHMM`][sparc.structures.hmm.GeneralizedHMM] | HMM with pluggable leaf distribution |
| [`HCLT`][sparc.structures.hclt.HCLT] | Hidden tree from data (MI + MST) |
| [`PD`][sparc.structures.pd.PD] | Recursive grid decomposition |
| [`PDHCLT`][sparc.structures.pd.PDHCLT] | PD with HCLT leaf regions |
| [`RAT_SPN`][sparc.structures.rat_spn.RAT_SPN] | Randomized tensorized sum-product network |

## Example: HCLT from data

```python
import numpy as np
from sparc.structures import HCLT

data = np.random.randint(0, 4, size=(500, 8))
circuit = HCLT(data, num_latents=4, num_bins=16, seed=0)
```

## Leaf distributions

Structures accept an optional [`InputDistribution`][sparc.structures.distributions.InputDistribution]:

| Class | Leaf type |
|-------|-----------|
| [`Categorical`][sparc.structures.distributions.Categorical] | Random categorical PMF |
| [`Bernoulli`][sparc.structures.distributions.Bernoulli] | Binary leaf |
| [`Indicator`][sparc.structures.distributions.Indicator] | Deterministic categorical |
| [`Literal`][sparc.structures.distributions.Literal] | Deterministic boolean |
| [`DiscreteLogistic`][sparc.structures.distributions.DiscreteLogistic] | Discretized logistic |

```python
from sparc.structures import GeneralizedHMM, Bernoulli

circuit = GeneralizedHMM(
    seq_length=10,
    num_latents=4,
    input_dist=Bernoulli(p=0.3),
    seed=0,
)
```

## Block algebra

Internally, structures compose [`Block`][sparc.structures._blocks.Block]
objects (lists of sum nodes) via product and sum operations. This keeps
structure code independent of the Cython node layer.

See the [architecture handbook](../handbook/architecture.md) for how builders,
structures, and the node core connect.
