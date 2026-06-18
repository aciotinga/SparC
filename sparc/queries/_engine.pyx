# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Shared machinery for pairwise (two-circuit) coupling queries.

Provides the pair-keyed memo, gradient tape (append / lookup / reverse replay),
per-side gradient arrays, and structured product-child scope matching used by
the GCW, CW, and expectation queries. Each query subclasses
:class:`CoupleContext`, supplies its ``couple_value`` recursion and a handful of
:class:`TapeEntry` subclasses with their own ``backward`` math.

Two keying schemes are used deliberately: the coupling memo / tape are keyed by
Python object identity (unique across the two circuits), while the returned
gradient bundles are keyed by ``node.id`` so callers can match gradients to
nodes.
"""

from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

import numpy as np

from sparc.nodes cimport CircuitNode, ProductNode

cdef size_t NO_TAPE_IDX = <size_t>(-1)


cdef class TapeEntry:
    """Base tape entry. Subclasses store forward state + override ``backward``."""

    cdef void backward(self, object ctx, double g) except *:
        pass


cdef class CoupleContext:
    def __cinit__(self):
        self.recording = False
        self.tape = []
        self.sum_grads0 = {}
        self.cat_grads0 = {}
        self.sum_grads1 = {}
        self.cat_grads1 = {}

    cdef bint memo_get(self, CircuitNode P, CircuitNode Q, double* out) noexcept:
        cdef unordered_map[uint64_t, double].iterator it = self.couple_memo.find(
            pair_key(P, Q))
        if it == self.couple_memo.end():
            return False
        out[0] = deref(it).second
        return True

    cdef void memo_put(self, CircuitNode P, CircuitNode Q, double val) noexcept:
        self.couple_memo[pair_key(P, Q)] = val

    cdef size_t lookup_pair_tape_idx(self, CircuitNode P, CircuitNode Q) noexcept:
        cdef unordered_map[uint64_t, size_t].iterator it = self.pair_to_tape.find(
            pair_key(P, Q))
        if it == self.pair_to_tape.end():
            return NO_TAPE_IDX
        return deref(it).second

    cdef size_t append_tape(self, TapeEntry entry, CircuitNode P, CircuitNode Q) except *:
        cdef size_t idx = <size_t>len(self.tape)
        self.tape.append(entry)
        self.tape_adjoints.push_back(0.0)
        self.pair_to_tape[pair_key(P, Q)] = idx
        return idx

    cdef object sum_grad_arr(self, int side, CircuitNode node, size_t n):
        cdef dict store = self.sum_grads0 if side == 0 else self.sum_grads1
        cdef object key = node.id
        cdef object arr = store.get(key)
        if arr is None:
            arr = np.zeros(n, dtype=np.float64)
            store[key] = arr
        return arr

    cdef object cat_grad_arr(self, int side, CircuitNode node, size_t n):
        cdef dict store = self.cat_grads0 if side == 0 else self.cat_grads1
        cdef object key = node.id
        cdef object arr = store.get(key)
        if arr is None:
            arr = np.zeros(n, dtype=np.float64)
            store[key] = arr
        return arr

    cdef void run_backward(self) except *:
        cdef ssize_t k
        cdef double g
        cdef TapeEntry entry
        for k in range(<ssize_t>len(self.tape) - 1, -1, -1):
            g = self.tape_adjoints[<size_t>k]
            if g == 0.0:
                continue
            entry = <TapeEntry>self.tape[k]
            entry.backward(self, g)

    cdef void reset_base(self):
        self.couple_memo.clear()
        self.tape = []
        self.tape_adjoints.clear()
        self.pair_to_tape.clear()
        self.sum_grads0 = {}
        self.cat_grads0 = {}
        self.sum_grads1 = {}
        self.cat_grads1 = {}


cdef inline uint64_t _scope_sig(CircuitNode node) noexcept:
    # Order-independent 64-bit signature over the node's (C-level) scope set:
    # XOR of a bit-mixed hash per variable, plus the cardinality. Collisions are
    # possible but rare and always resolved by an exact set comparison below, so
    # the signature is only a fast bucket key (no frozenset allocation).
    cdef uint64_t sig = 0
    cdef uint64_t h
    cdef unordered_set[int].iterator it = node.scope.begin()
    while it != node.scope.end():
        h = <uint64_t><unsigned int>deref(it)
        # SplitMix64-style finalizer for good avalanche.
        h = (h ^ (h >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
        h = (h ^ (h >> 27)) * <uint64_t>0x94d049bb133111eb
        h = h ^ (h >> 31)
        sig ^= h
        inc(it)
    return sig ^ (<uint64_t>node.scope.size() * <uint64_t>0x9e3779b97f4a7c15)


cdef inline bint _scope_eq(CircuitNode a, CircuitNode b) noexcept:
    if a.scope.size() != b.scope.size():
        return False
    cdef unordered_set[int].iterator it = a.scope.begin()
    while it != a.scope.end():
        if b.scope.find(deref(it)) == b.scope.end():
            return False
        inc(it)
    return True


cdef void match_prod_children(
    ProductNode P,
    ProductNode Q,
    vector[int]& row_ind,
    vector[int]& col_ind,
    str query_name,
) except *:
    """Match product children by scope into a bijection (structured decomp.).

    Fills ``row_ind[i] = i`` and ``col_ind[i] = matched Q-child index``. Matching
    is done by an integer scope signature (computed from the C-level scope set)
    with exact set-comparison fallback, avoiding per-call Python frozenset/dict
    construction while preserving the original incompatibility errors.
    """
    cdef size_t n = P.num_children()
    cdef size_t m = Q.num_children()
    cdef size_t i
    cdef size_t j
    cdef CircuitNode q_child
    cdef CircuitNode p_child
    cdef uint64_t sig
    cdef int matched
    cdef size_t remaining
    # Per-signature buckets of (still-unmatched) Q child indices.
    cdef unordered_map[uint64_t, vector[int]] buckets
    cdef unordered_map[uint64_t, vector[int]].iterator bit
    cdef vector[int]* bucket
    cdef list q_children = []
    cdef size_t b

    if n != m:
        raise ValueError(
            f"{query_name} incompatible: product nodes have different numbers "
            f"of children ({n} vs {m})"
        )
    row_ind.assign(n, 0)
    col_ind.assign(n, 0)
    for j in range(m):
        q_child = Q.child_at(j)
        q_children.append(q_child)
        sig = _scope_sig(q_child)
        # Reject duplicate Q scopes (same error as before): a signature match
        # plus an exact scope match means a true duplicate.
        bit = buckets.find(sig)
        if bit != buckets.end():
            bucket = &deref(bit).second
            for b in range(bucket.size()):
                if _scope_eq(<CircuitNode>q_children[deref(bucket)[b]], q_child):
                    raise ValueError(
                        f"{query_name} incompatible: duplicate child scope among "
                        "Q product children"
                    )
        buckets[sig].push_back(<int>j)
    remaining = m
    for i in range(n):
        p_child = P.child_at(i)
        sig = _scope_sig(p_child)
        matched = -1
        bit = buckets.find(sig)
        if bit != buckets.end():
            bucket = &deref(bit).second
            for b in range(bucket.size()):
                if deref(bucket)[b] >= 0 and _scope_eq(
                    <CircuitNode>q_children[deref(bucket)[b]], p_child
                ):
                    matched = deref(bucket)[b]
                    deref(bucket)[b] = -1  # consume this Q child
                    break
        if matched < 0:
            raise ValueError(
                f"{query_name} incompatible: no Q product child with scope "
                f"matching P child at index {i}"
            )
        row_ind[i] = <int>i
        col_ind[i] = matched
        remaining -= 1
    if remaining != 0:
        raise ValueError(
            f"{query_name} incompatible: Q product children with unmatched scopes"
        )
