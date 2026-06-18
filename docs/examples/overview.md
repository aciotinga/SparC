# Examples

Runnable scripts live in the [`examples/`](https://github.com/adrianciotinga/sparc/tree/main/examples)
directory. Run from the repository root:

```bash
PYTHONPATH=. python examples/<script>.py
```

## mle

Maximum-likelihood training of a random [`EmbeddingBuilder`][sparc.builders.embedding.EmbeddingBuilder]
circuit by projected gradient ascent.

```bash
PYTHONPATH=. python examples/mle.py
```

## cw_minimization

Minimize Circuit-Wasserstein distance from a fixed circuit `P` to a learnable
circuit `Q` using [`cw_distance_and_grad`][sparc.queries.cw.cw_distance_and_grad].

```bash
PYTHONPATH=. python examples/cw_minimization.py
```

## gcw_optimization

Optimize the GCW cross-term between two circuits (minimize or maximize):

```bash
PYTHONPATH=. python examples/gcw_optimization.py --direction min
PYTHONPATH=. python examples/gcw_optimization.py --direction max
```

## exp_minimization

Minimize [`log_exp_query`][sparc.queries.expectation.log_exp_query] (log
expectation) of one circuit under another.

```bash
PYTHONPATH=. python examples/exp_minimization.py
```

## dro

Distributionally robust optimization saddle-point demo on synthetic circuits.

```bash
PYTHONPATH=. python examples/dro.py
```

## robustify

DRO-style robustification on pre-trained circuits in `examples/example_pcs/`.

```bash
PYTHONPATH=. python examples/robustify.py
```

## structures

Demonstrates all built-in structures and evaluates CW distance between pairs.

```bash
PYTHONPATH=. python examples/structures.py
```

## bench_gcw

Benchmark GCW forward and gradient passes with optional profiling.

```bash
PYTHONPATH=. python examples/bench_gcw.py
```

## train_mnist

Train an HCLT circuit on downsampled MNIST (requires `matplotlib`,
`torchvision`):

```bash
PYTHONPATH=. python examples/train_mnist.py --epochs 10
```

## Pre-trained circuits

The `examples/example_pcs/` directory contains JSON circuits for datasets
including adult, plants, bbc, and others. Load them with
[`load_learned_pc`][sparc.io.learned_pc.load_learned_pc] or
[`Circuit.load`][sparc.circuit.Circuit.load].
