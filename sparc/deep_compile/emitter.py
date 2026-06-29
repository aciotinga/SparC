"""Emit ultra or compat C glue from a CompiledCircuit codegen snapshot."""

from __future__ import annotations

from typing import Any

from sparc.deep_compile.isa import IsaPlan, resolve_isa
from sparc.deep_compile.ultra_emitter import emit_c_source as emit_ultra_c_source

NODE_SUM = 0
NODE_PRODUCT = 1
NODE_INPUT = 2

SPARC_OP_LEAF_BIN = 0
SPARC_OP_LEAF_TBL = 1
SPARC_OP_PROD_LIN = 2
SPARC_OP_PROD_LOG = 3
SPARC_OP_SUM_LIN = 4
SPARC_OP_SUM_LOG = 5


def emit_c_source(
    snap: dict[str, Any],
    *,
    isa: IsaPlan | None = None,
    parallel: bool = True,
    tile: int = 128,
    mode: str = "ultra",
) -> str:
    """Generate C source for deep compile.

    Default *mode* is ``"ultra"`` (inlined per-node SIMD kernel).
    Pass ``mode="compat"`` for the legacy SparcOp dispatch-table glue.
    """
    if mode == "compat":
        from sparc.deep_compile.compat_emitter import emit_compat_c_source

        return emit_compat_c_source(snap, parallel=parallel, tile=tile)
    plan = isa if isa is not None else resolve_isa(None)
    return emit_ultra_c_source(snap, isa=plan, parallel=parallel, tile=tile)


def leaf_var_order(snap: dict[str, Any]) -> list[int]:
    """Return variable ids in leaf_ev row order (compat helper)."""
    from sparc.deep_compile.compat_emitter import leaf_var_order as _leaf_var_order

    return _leaf_var_order(snap)
