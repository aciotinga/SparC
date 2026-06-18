---
hide:
  - toc
---

# SparC

**Spar**se **C**ircuits — a fast, modular, CPU-only library for probabilistic
circuits written in Cython.

<div class="grid cards" markdown>

-   __Build a circuit__

    ---

    [:octicons-arrow-right-24: Quick start](getting-started/quick-start.md)

    [:octicons-arrow-right-24: Random builders](guides/builders.md)

    [:octicons-arrow-right-24: Built-in structures](guides/structures.md)

-   __Learn a circuit__

    ---

    [:octicons-arrow-right-24: Maximum likelihood training](getting-started/training.md)

    [:octicons-arrow-right-24: Example scripts](examples/overview.md)

-   __Compare and optimize circuits__

    ---

    [:octicons-arrow-right-24: Wasserstein queries](guides/queries.md)

    [:octicons-arrow-right-24: Extending the library](guides/extending.md)

</div>

## Highlights

- **Speed**: typed Cython with C++17 containers, `nogil` inner loops, and a
  compiled batched evaluation path.
- **Modularity**: leaf nodes dispatch through a C-level vtable; all pairwise
  queries share one tape/gradient engine.
- **Minimal deps**: `numpy` only at runtime.

See the [handbook](handbook/architecture.md) for internal architecture and the
[API reference](api/overview.md) for generated documentation from docstrings.
