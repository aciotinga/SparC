# Queries

SparC provides differentiable queries over probabilistic circuits. Pairwise
queries require **structurally compatible** circuits: equal scopes, matching
decompositions at sum and product nodes, and compatible leaf cardinalities.

## Supported queries

| Query | Math | Gradients w.r.t. |
|-------|------|------------------|
| CW | $W_p^p$ (Circuit-Wasserstein objective) | circuit2 only |
| GCW cross-term | Gromov-Circuit-Wasserstein cross-term | circuit2 only |
| Expectation | $E_Q[P(X)] = \sum_x P(x)\,Q(x)$ | both circuits |
| Log expectation | $\log E_Q[P(X)]$ | both circuits |
| ESD | $E[d(X,X')^p]$ for two i.i.d. draws | single circuit |

The CW value is the **additive** $W_p^p$ objective; take the $p$-th root for
the distance itself.

## Circuit-Wasserstein (CW)

```python
from sparc import cw_distance, cw_distance_and_grad, PNormMetric

val = cw_distance(p, q, metric=PNormMetric(p=2.0, scale=1.0))
val, grad = cw_distance_and_grad(p, q)  # grad w.r.t. q only
```

Leaf couplings use the northwest-corner plan; sum-sum couplings solve a
transport LP with duals for the subgradient.

## Gromov-Circuit-Wasserstein (GCW)

```python
from sparc import gcw_crossterm, gcw_crossterm_and_grad, gcw_coupling_circuit

cross = gcw_crossterm(c1, c2)
cross, grad = gcw_crossterm_and_grad(c1, c2)

coupling = gcw_coupling_circuit(c1, c2)
coupling.sample(1000)  # joint (x, y) pairs
```

The coupling circuit lives over `vars(c1)` plus `vars(c2)` shifted to a
disjoint range. Ancestral sampling draws joint pairs whose marginals match the
inputs.

## Expectation queries

```python
from sparc import exp_query, log_exp_query, exp_query_and_grad

val = exp_query(p, q)
log_val = log_exp_query(p, q)
val, g1, g2 = exp_query_and_grad(p, q)
```

## Expected squared distance (ESD)

```python
from sparc import expected_squared_distance, expected_squared_distance_and_grad

val = expected_squared_distance(circuit, metric_p=2.0)
val, grad = expected_squared_distance_and_grad(circuit)
```

## Structural compatibility

Queries fail at runtime when:

- Scopes differ between paired circuits.
- A sum node in one circuit pairs with a product node in the other.
- Leaf supports or cardinalities disagree on the same variable.

See [`test_query_compatibility.py`](https://github.com/aciotinga/sparc/blob/main/tests/test_query_compatibility.py)
for concrete error cases.

## Ground metrics

Pass a custom [`GroundMetric`][sparc.metrics.GroundMetric] or use the default
[`PNormMetric`][sparc.metrics.PNormMetric] with $d(i,j) = |i-j|^p / \mathrm{scale}$.

## API reference

- [`sparc.queries`][sparc.queries]
- [Query engine handbook](../handbook/query-engine.md)
