# Differentiable sampling

SparC can retain a SIMPLE gradient-estimator tape while drawing exact, hard
ancestral samples:

```python
draws = circuit.sample(128, seed=0, differentiable=True)
```

The default remains `differentiable=False`, which returns the existing
`int32` NumPy array without constructing a tape.

## Result layout

Differentiable sampling returns
[`DifferentiableSample`][sparc.sampling.DifferentiableSample]:

- `assignments` is the legacy `(n_samples, max_var + 1)` integer array.
- `one_hot` is a hard `float64` one-hot matrix packed by ascending variable
  ID. Its width is the sum of the scoped variables' cardinalities.
- `variables`, `cardinalities`, and cumulative `offsets` describe that packed
  layout.

For variables `[3, 17]` with cardinalities `[2, 3]`, `offsets` is
`[0, 2, 5]`; variable 17 occupies `one_hot[:, 2:5]`. All returned arrays are
read-only so that they cannot diverge from the frozen tape.

## Vector-Jacobian products

Pass the gradient of a scalar downstream loss with respect to `one_hot` to
`vjp`:

```python
from sparc.optim import apply_grads

draws = circuit.sample(128, seed=0, differentiable=True)
residual = draws.one_hot - target
loss = (residual * residual).sum(axis=1).mean()

# vjp() sums its input; this upstream already includes mean reduction.
upstream = 2.0 * residual / len(draws.one_hot)
grads = draws.vjp(upstream)
apply_grads(circuit, grads, lr=1e-2, ascent=False)
```

`upstream` must have exactly the same shape as `one_hot`. The returned
[`GradBundle`][sparc.grad.GradBundle] has `has_value=False` and `value=NaN`
because a VJP receives no scalar objective value. Its `sum_grads` and
`cat_grads` use the same node-ID keys as every other SparC gradient API.

SparC parameters are normalized linear probabilities. The VJP therefore
returns ambient gradients in probability coordinates; `apply_grads` projects
them onto each simplex tangent before updating parameters.

## SIMPLE estimator

Every sum-node selector and trainable categorical/Bernoulli leaf is a
one-hot (`k=1`) decision. SparC uses the APC specialization of SIMPLE:

\[
\widetilde{s} = \operatorname{stopgrad}(s-p) + p,
\]

where `s` is an exact hard categorical sample and `p` contains the
unperturbed probabilities. Forward computation sees `s`; the VJP follows
`p`.

At a sum node, SparC draws one counterfactual hard assignment from every
child, then returns the selected child assignment. If `U` is the packed
upstream and child `i` sampled category `a[i,v]` for variable `v`, its weight
gradient is

\[
G_i = \sum_{v \in \mathrm{scope(sum)}} U[\mathrm{offset}(v)+a[i,v]].
\]

Only the hard-selected child receives nested parameter gradients. At an
active trainable leaf over variable `v`,

\[
G_{\mathrm{leaf},k}=U[\mathrm{offset}(v)+k].
\]

Contributions are summed over batch rows and repeated DAG occurrences. SIMPLE
is a biased surrogate for general nonlinear losses; it is not the derivative
of the discrete hard sample itself.

## Requirements and limits

Differentiable mode requires:

- finite-discrete leaves with consistent cardinality for each variable;
- smooth sums (identical child scopes and cardinalities);
- decomposable products (pairwise-disjoint child scopes);
- an acyclic graph with unique node IDs.

Indicator, literal, and discrete-logistic leaves can appear in a sample but
do not produce leaf-parameter gradients. Continuous Gaussian circuits use
reparameterization $x=\mu+\sigma\varepsilon$ instead of SIMPLE; `vjp` takes a
gradient on `assignments` (`float64`) and writes `GradBundle.cont_grads`.
`one_hot` is empty for continuous tapes.

Version 0.7 implements unconditional differentiable sampling, matching the
existing `sample()` semantics. It does not accept evidence or implement APC's
conditional posterior reweighting.

`CompiledCircuit.sample(..., differentiable=True)` uses the compiled
parameter snapshot. Call `refresh_parameters()` after updates; cardinality
changes require recompiling. A retained sample tape owns its forward outcomes,
so its VJP remains valid after later updates or refreshes.

Sampling all immediate counterfactual children adds work and tape memory
relative to ordinary one-path ancestral sampling. Trace storage uses compact
integer assignments rather than a dense one-hot copy per child.

See the [SIMPLE paper](https://arxiv.org/abs/2210.01941) and Autoencoding
Probabilistic Circuits Appendix I for the estimator derivation.
