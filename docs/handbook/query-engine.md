# Query Engine

Pairwise queries (CW, GCW, expectation) share machinery in
the internal module `sparc.queries._engine`.

## CoupleContext

Each query subclasses `CoupleContext` and implements:

- **`couple_value(P, Q, side_P, side_Q)`**: recursive forward evaluation for
  a aligned pair of nodes from circuits 1 and 2.
- **Tape entries**: `TapeEntry` subclasses store forward state and implement
  `backward(ctx, g)` for reverse-mode accumulation.

## Memoization and tape

Two keying schemes are used deliberately:

| Structure | Key | Purpose |
|-----------|-----|---------|
| Coupling memo / tape | Python object identity of `(P, Q)` | Unique across two circuits |
| Returned gradients | `node.id` | Match grads to user-visible nodes |

The engine appends tape entries during forward evaluation when `recording=True`,
then replays them in reverse to populate `sum_grads*` and `cat_grads*` arrays.

## Product-child matching

When both paired nodes are products over the same scope, children are matched
by variable scope (not by pointer identity). Unmatched children raise
compatibility errors.

## Gradient semantics

| Query family | Gradients returned |
|--------------|-------------------|
| CW, GCW | w.r.t. **circuit2** only |
| Expectation, log expectation | w.r.t. **both** circuits |
| ESD | w.r.t. the single input circuit |

Apply gradients with [`apply_grads`][sparc.optim.apply_grads]; remember to
project onto simplices for probability parameters.

## Per-query specialization

| Module | Leaf coupling | Sum-sum coupling | Product-product |
|--------|---------------|------------------|-----------------|
| `cw.pyx` | NW plan | Transport LP + duals | Scope-matched product |
| `gcw.pyx` | NW + cross costs | Transport + assignment | Hungarian matching |
| `expectation.pyx` | PMF product | Weighted child recursion | Scope-matched product |

See [Solvers](solvers.md) for the underlying OT and assignment routines.
