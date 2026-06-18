"""Import PyJuice circuits into SparC."""

from __future__ import annotations

from sparc.circuit import Circuit

from sparc.builders._factory import _NodeFactory


class PyjuiceBuilder:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.scope_offset = 0

    def build(self, circuit, scope_offset: int = 0) -> Circuit:
        self.scope_offset = scope_offset
        factory = _NodeFactory()
        root = self._build(circuit.root_ns, {}, factory)[0]
        return Circuit(root)

    def _build(self, circuit, cache, factory: _NodeFactory):
        if circuit in cache:
            return cache[circuit]

        if circuit.is_sum():
            child_vals = self._build(circuit.chs[0], cache, factory)
            params = circuit.get_params().numpy()
            value = [
                factory.sum(child_vals, params[0, i, :]) for i in range(len(params[0]))
            ]
        elif circuit.is_prod():
            child_vals = [self._build(child, cache, factory) for child in circuit.chs]
            value = [
                factory.product([c[i] for c in child_vals])
                for i in range(self.block_size)
            ]
        elif circuit.is_input():
            params = circuit.get_params().numpy().reshape(self.block_size, -1)
            scope_var = list(circuit.scope)[0] + self.scope_offset
            value = [
                factory.categorical(scope_var, params[i, :])
                for i in range(self.block_size)
            ]
        else:
            raise NotImplementedError("Invalid node type")
        cache[circuit] = value
        return value
