"""Integration tests for PyjuiceBuilder.

Requires pyjuice and torch. Marked @pytest.mark.pyjuice so default CI skips.
"""

import math

import numpy as np
import pytest

juice = pytest.importorskip("pyjuice", reason="pyjuice not installed")
torch = pytest.importorskip("torch", reason="torch not installed")

from sparc.builders import PyjuiceBuilder


def _tiny_pc(device="cpu", num_vars=4, num_latents=2):
    """Build the smallest possible PyJuice PC from synthetic data."""
    rng = np.random.RandomState(0)
    data = torch.from_numpy(rng.randint(0, 3, size=(64, num_vars))).to(device)
    ns = juice.structures.HCLT(data.float(), num_latents=num_latents)
    pc = juice.compile(ns)
    return pc


@pytest.mark.pyjuice
class TestPyjuiceBuilder:
    def test_build_and_likelihood(self):
        pc = _tiny_pc()
        builder = PyjuiceBuilder(block_size=pc.root_ns.num_chs)
        circuit = builder.build(pc)
        scope = circuit.root.scope_as_list()
        assignment = {v: 0 for v in scope}
        ll = circuit.likelihood(assignment)
        log_ll = circuit.log_likelihood(assignment)
        assert ll > 0.0
        assert math.isfinite(log_ll)
        assert log_ll == pytest.approx(math.log(ll))

    def test_fake_circuit_errors(self):
        builder = PyjuiceBuilder(block_size=1)

        class FakeCircuit:
            root_ns = None

        with pytest.raises(AttributeError):
            builder.build(FakeCircuit())
