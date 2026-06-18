"""Circuit serialization and loading.

Circuits are saved as UTF-8 JSON in the ``gcw-circuit-v1`` format via
:class:`~sparc.io.serializer.CircuitSerializer`. Pre-trained circuits saved
under a standard directory layout can be loaded with
:func:`~sparc.io.learned_pc.load_learned_pc`.
"""

from sparc.io.learned_pc import load_learned_pc
from sparc.io.serializer import CircuitSerializer

__all__ = ["CircuitSerializer", "load_learned_pc"]
