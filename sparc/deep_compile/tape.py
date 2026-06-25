"""Parameter tape layout for deep-compiled circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NODE_INPUT = 2


@dataclass(frozen=True)
class TapeLayout:
    """Index layout for mutable linear / log parameter tapes."""

    n_leaf_pmf: int
    sum_w_base: int
    tape_len: int
    leaf_pmf_off: tuple[int, ...]
    leaf_var: tuple[int, ...]
    leaf_col: tuple[int, ...]
    n_nodes: int
    root_index: int

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> TapeLayout:
        n_nodes = snap["n_nodes"]
        n_leaf_pmf = int(snap["leaf_pmf_off"][n_nodes])
        n_edges = int(snap["child_off"][n_nodes])
        sum_w_base = n_leaf_pmf
        tape_len = n_leaf_pmf + n_edges

        leaf_pmf_off = [-1] * n_nodes
        leaf_var = [-1] * n_nodes
        leaf_col = [-1] * n_nodes
        for n in range(n_nodes):
            if snap["kinds"][n] != NODE_INPUT:
                continue
            leaf_pmf_off[n] = int(snap["leaf_pmf_off"][n])
            leaf_var[n] = int(snap["leaf_var"][n])
            leaf_col[n] = int(snap["leaf_var"][n])

        return cls(
            n_leaf_pmf=n_leaf_pmf,
            sum_w_base=sum_w_base,
            tape_len=tape_len,
            leaf_pmf_off=tuple(leaf_pmf_off),
            leaf_var=tuple(leaf_var),
            leaf_col=tuple(leaf_col),
            n_nodes=n_nodes,
            root_index=int(snap["root_index"]),
        )

    def sum_w_index(self, snap: dict[str, Any], node: int, child_slot: int) -> int:
        csr = int(snap["child_off"][node]) + child_slot
        return self.sum_w_base + csr


def fill_linear_tape(snap: dict[str, Any], out: np.ndarray) -> None:
    layout = TapeLayout.from_snapshot(snap)
    if out.shape != (layout.tape_len,):
        raise ValueError(f"expected tape length {layout.tape_len}, got {out.shape}")
    n_leaf = layout.n_leaf_pmf
    out[:n_leaf] = snap["leaf_pmf_flat"]
    out[n_leaf:] = snap["sum_w_flat"]


def fill_log_tape(snap: dict[str, Any], out: np.ndarray) -> None:
    layout = TapeLayout.from_snapshot(snap)
    if out.shape != (layout.tape_len,):
        raise ValueError(f"expected tape length {layout.tape_len}, got {out.shape}")
    n_leaf = layout.n_leaf_pmf
    out[:n_leaf] = snap["leaf_logpmf_flat"]
    out[n_leaf:] = snap["sum_logw_flat"]


def make_tape_buffers(snap: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    layout = TapeLayout.from_snapshot(snap)
    tape_lin = np.empty(layout.tape_len, dtype=np.float64)
    tape_log = np.empty(layout.tape_len, dtype=np.float64)
    fill_linear_tape(snap, tape_lin)
    fill_log_tape(snap, tape_log)
    return tape_lin, tape_log
