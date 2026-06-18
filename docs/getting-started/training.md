# Training

SparC optimizes circuit parameters with **projected gradient ascent** on the
mean log-likelihood. Sum-node weights and leaf PMFs live on probability
simplices, so each step projects back onto the feasible set.

## MLETrainer

```python
from sparc.optim import MLETrainer

trainer = MLETrainer(circuit, lr=0.5, method="tangent")
history = trainer.fit(dataset, epochs=100)
```

- `dataset`: iterable of `{var: value}` dicts (complete assignments).
- `method`: `"tangent"` (project gradient onto simplex tangent, clip,
  renormalize) or `"euclidean"` (exact Euclidean projection after the step).

## Manual gradient steps

For custom objectives (e.g. minimizing a circuit distance), compute a
`GradBundle` from a query and apply projected steps yourself:

```python
from sparc.optim import apply_grads

val, grad = cw_distance_and_grad(p, q)
apply_grads(q, grad, lr=1e-2, ascent=False)  # minimize w.r.t. q
```

## API reference

- [`MLETrainer`][sparc.optim.MLETrainer]
- [`simplex_step`][sparc.optim.simplex_step]
- [`apply_grads`][sparc.optim.apply_grads]
- [`mean_log_likelihood_and_grad`][sparc.grad.mean_log_likelihood_and_grad]

See also the [MLE example script](../examples/overview.md#mle).
